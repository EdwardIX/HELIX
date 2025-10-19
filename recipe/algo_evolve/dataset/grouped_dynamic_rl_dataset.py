from verl.utils.dataset.rl_dataset import RLHFDataset
import datasets
import xxhash
import yaml
import json
import pickle
import os
import re
import ray
import numpy as np
from typing import Set

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from recipe.algo_evolve.dataset.prompt_sampler import ComsolProcessor, MathProgramProcessor, AIProgramProcessor, MathSymbolicRegressionProcessor
from recipe.algo_evolve.dataset.sample_strategy import get_solution_from_dataframe, RandomSampleStrategy, TopKSampleStrategy, NSGASampleStrategy, TopKDivSampleStrategy
from src.simplify import normalize_yaml, normalize_python_code

PROMPT_SAMPLER_CLASSES = {
    "comsol_*": ComsolProcessor,
    "math_SR_*": MathSymbolicRegressionProcessor,
    "math_*": MathProgramProcessor,
    "ai_*": AIProgramProcessor,
}

datasets.disable_progress_bars()

class GroupedDynamicRLHFDataset(RLHFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = self.config.custom_cls.init_kwargs.is_train
        if not self.enabled: # Behave just like RLHFDataset
            return
        
        self._lock = threading.Lock()
        self.log_path = self.config.custom_cls.init_kwargs.log_path
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)

        # Record indices grouped by data_source
        self.max_trial_per_ds = self.config.custom_cls.init_kwargs.max_trial_per_ds if self.config.custom_cls.init_kwargs.max_trial_per_ds != -1 else np.inf
        self.max_bsz_per_ds = min(self.config.custom_cls.init_kwargs.max_bsz_per_ds, self.config.train_batch_size) if self.config.custom_cls.init_kwargs.max_bsz_per_ds != -1 else self.config.train_batch_size # For each ds, only the first max_bsz_per_ds points are sampled
        self.source_to_indices = {}
        for i, item in enumerate(self.dataframe):
            ds = item["data_source"]
            self.source_to_indices.setdefault(ds, []).append(i)
        self.sources_uid = {k: i for i, k in enumerate(self.source_to_indices.keys())}
        self.sources_count = np.array([len(indices) for indices in self.source_to_indices.values()])
        self.trials_count = np.zeros_like(self.sources_count)

        self.records = {}
        for ds, indices in self.source_to_indices.items():
            self.records[ds] = {
                "sampled_count": [0] * len(indices),
                "source_index": [-1] * len(indices),
            }

        # Init Configs
        prompt_sampler_config = dict(self.config.get('dynamic_prompt_sampler', {}))
        sample_strategy_config = dict(self.config.get('dynamic_sample_strategy', {'type': "Random"}))
        self.filter_invalid_responses = self.config.get('filter_invalid_responses', True)
        self.filter_overlong_prompts_strategy = self.config.get('filter_overlong_prompts_strategy', 'Truncate')
        self.filter_duplicate_responses = self.config.get('filter_duplicate_responses', True)
        if self.filter_duplicate_responses:
            self.seen_hashes = {ds: set() for ds in self.source_to_indices.keys()}

        self.resp_normalize_func = {
            'python': normalize_python_code,
            'yaml': normalize_yaml
        }[self.config.custom_cls.init_kwargs.language]

        # Create independent prompt_sampler and sample_strategy for each data_source
        self.prompt_samplers = {}
        self.sample_strategies = {}
        sample_strategy_type = sample_strategy_config.pop('type')

        def init_single_datasource(ds, indices):
            for pattern, prompt_sampler_cls in PROMPT_SAMPLER_CLASSES.items():
                if re.match(pattern, ds):
                    prompt_sampler = prompt_sampler_cls(**prompt_sampler_config)
                    break
            else:
                raise ValueError(f"No matching prompt sampler found for data source: {ds}")

            sub_df = self.dataframe.select(indices)

            if sample_strategy_type == "Random":
                strategy = RandomSampleStrategy(sub_df, self.config.train_batch_size, **sample_strategy_config)
            elif sample_strategy_type == "TopK":
                strategy = TopKSampleStrategy(sub_df, self.config.train_batch_size, **sample_strategy_config)
            elif sample_strategy_type == "NSGA":
                strategy = NSGASampleStrategy(sub_df, self.config.train_batch_size, **sample_strategy_config)
            elif sample_strategy_type == "TopKDiv":
                strategy = TopKDivSampleStrategy(sub_df, self.config.train_batch_size, **sample_strategy_config)

            with self._lock:
                self.prompt_samplers[ds] = prompt_sampler
                self.sample_strategies[ds] = strategy
                if self.filter_duplicate_responses:
                    self._update_hashes(sub_df, ds)

        with ThreadPoolExecutor(max_workers=len(self.source_to_indices)) as executor:
            futures = [
                executor.submit(init_single_datasource, ds, indices)
                for ds, indices in self.source_to_indices.items()
            ]
            for future in as_completed(futures):
                future.result()
        # for ds, indices in self.source_to_indices.items():
        #     init_single_datasource(ds, indices)

    def __getitem__(self, idx):
        if not self.enabled:
            return super().__getitem__(idx)
        sources_prob = np.where(self.trials_count < self.max_trial_per_ds, self.sources_count, 0)
        if sources_prob.sum() == 0:
            raise ValueError(f"All the tasks reached max trial count {self.max_trial_per_ds}, exit program.")
        ds = np.random.choice(list(self.source_to_indices.keys()), p=sources_prob / sources_prob.sum())
        mapped_idx = int(idx / self.config.train_batch_size * self.max_bsz_per_ds)
        local_idx = self.sample_strategies[ds](mapped_idx)
        local_indices = self.source_to_indices[ds]
        global_idx = local_indices[local_idx]
        result = super().__getitem__(global_idx)
        result['extra_info']['source_index'] = global_idx
        return result
    
    def __len__(self):
        if not self.enabled:
            return super().__len__()
        return self.config.train_batch_size

    def _update_hashes(self, dataframe, ds):
        """Update the seen hashes with the new dataframe"""
        new_idx = []
        solutions = get_solution_from_dataframe(dataframe)
        for i, solution in enumerate(solutions):
            # Try to calculate Hash for single solution:
            solution = self.resp_normalize_func(solution)
            h = xxhash.xxh64(solution).intdigest()
            target_hashes = self.seen_hashes[ds]
            if h not in target_hashes:
                new_idx.append(i)
                target_hashes.add(h)
        return new_idx

    def log(self):
        if not self.log_path:
            return
        for ds in self.source_to_indices.keys():
            os.makedirs(os.path.join(self.log_path, ds), exist_ok=True)
            self.sample_strategies[ds].log(os.path.join(self.log_path, ds))
            with open(os.path.join(self.log_path, ds, f'rl_dataset_{len(self.source_to_indices[ds])}.txt'), 'w') as f:
                counts = self.records[ds]['sampled_count']
                f.write(f"DataSource: {ds}\n")
                f.write(f"Mapping from local index to global index: {self.source_to_indices[ds]}\n")
                f.write(f"Sampled Count: {counts}\n")
                f.write(f"Total samples: {sum(counts)}\n")
                f.write(f"Max samples for an item: {max(counts)}\n")
                sorted_counts = sorted(counts)
                for p in range(10, 100, 10):
                    idx = int(len(sorted_counts) * p / 100)
                    f.write(f"{p}% percentile: {sorted_counts[idx]}\n")
                f.write(f"Source Index: {self.records[ds]['source_index']}\n")

    def extend(self, data):
        if not self.enabled:
            raise ValueError("Cannot extend dataset when disabled")

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Group new data by data_source
        grouped_indices = {}
        for i in range(len(data)):
            ds = data.non_tensor_batch["data_source"][i]
            grouped_indices.setdefault(ds, []).append(i)
            self.trials_count[self.sources_uid[ds]] += 1

        def extend_single_datasource(ds, idx_list):
            source_index, responses = [], []
            for i in idx_list:
                source_index.append(data.non_tensor_batch["extra_info"][i]['source_index'])
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:valid_response_lengths[i].item()],
                    skip_special_tokens=True
                )
                responses.append(response_str)

            # Call corresponding prompt_sampler
            new_dataframe = self.prompt_samplers[ds](
                data=data[idx_list],
                src_dataframe=self.dataframe.select(source_index),
                responses=responses,
            )
            source_index = np.array(source_index, dtype=np.int64)

            # Filter invalid responses
            if self.filter_invalid_responses:
                validity = data.non_tensor_batch["validity"]
                valid_idx = [i for i in range(len(idx_list)) if validity[idx_list[i]] > 0]
                new_dataframe = new_dataframe.select(valid_idx)
                source_index = source_index[valid_idx]
                print(f"\n[{ds}] filter illegal responses, len: {len(new_dataframe)}")

            # Filter overlong prompts
            if self.filter_overlong_prompts:
                tokenizer = self.tokenizer
                prompt_key = self.prompt_key
                if self.filter_overlong_prompts_strategy == 'Filter':
                    valid_idx = []
                    for i, doc in enumerate(new_dataframe):
                        if len(tokenizer.apply_chat_template(
                                doc[prompt_key],
                                add_generation_prompt=True)) <= self.max_prompt_length:
                            valid_idx.append(i)
                    new_dataframe = new_dataframe.select(valid_idx)
                    source_index = source_index[valid_idx]
                    print(f"\n[{ds}] filter overlong prompts, len: {len(new_dataframe)}")

                elif self.filter_overlong_prompts_strategy == 'Truncate':
                    def truncate_prompt(doc):
                        messages = doc[prompt_key]
                        truncated_messages = []
                        total_tokens = 20  # Reserve some tokens for generation prompt
                        for msg in messages:
                            msg_tokens = tokenizer.apply_chat_template(
                                [msg], add_generation_prompt=False)
                            token_count = len(msg_tokens)
                            if total_tokens + token_count <= self.max_prompt_length:
                                truncated_messages.append(msg)
                                total_tokens += token_count
                            else:
                                remaining_tokens = self.max_prompt_length - total_tokens
                                if remaining_tokens > 0:
                                    content_tokens = tokenizer.encode(msg["content"])
                                    truncated_content_tokens = content_tokens[:remaining_tokens]
                                    truncated_content = tokenizer.decode(truncated_content_tokens)
                                    truncated_messages.append({
                                        "role": msg["role"],
                                        "content": truncated_content + "[truncated due to length limit]"
                                    })
                                    total_tokens = self.max_prompt_length
                                break
                        doc[prompt_key] = truncated_messages
                        return doc

                    new_dataframe = new_dataframe.map(
                        truncate_prompt,
                        # num_proc=self.num_workers,
                        desc=f"Truncating prompts to max {self.max_prompt_length} tokens",
                    )
                else:
                    raise ValueError(
                        f"Unknown filter_overlong_prompts_strategy: {self.filter_overlong_prompts_strategy}")

            # Deduplicate
            if self.filter_duplicate_responses:
                valid_idx = self._update_hashes(new_dataframe, ds)
                new_dataframe = new_dataframe.select(valid_idx)
                source_index = source_index[valid_idx]
                print(f"\n[{ds}] Filter duplicate responses len: {len(new_dataframe)}")

            # Update sample_strategy 
            new_dataframe = self.sample_strategies[ds].extend(new_dataframe)

            # Lock and update shared state
            with self._lock:
                self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])
                # Update records
                self.records[ds]['sampled_count'].extend([0] * len(new_dataframe))
                self.records[ds]['source_index'].extend(source_index.tolist())
                for i in range(len(new_dataframe)):
                    local_idx = self.source_to_indices[ds].index(source_index[i])
                    self.records[ds]['sampled_count'][local_idx] += 1
                self.source_to_indices[ds].extend(
                    range(len(self.dataframe) - len(new_dataframe), len(self.dataframe))
                )

        with ThreadPoolExecutor(max_workers=len(grouped_indices)) as executor:
            futures = [
                executor.submit(extend_single_datasource, ds, idx_list)
                for ds, idx_list in grouped_indices.items()
            ]

            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    raise e

        # Log
        self.log()

    def on_batch_end(self, batch):
        self.extend(batch)
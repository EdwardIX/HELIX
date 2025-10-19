from verl.utils.dataset.rl_dataset import RLHFDataset
import datasets
import xxhash
import yaml
import json
import pickle
import os
import re
import numpy as np
from typing import Set

from recipe.algo_evolve.dataset.prompt_sampler import ComsolProcessor, MathProgramProcessor, AIProgramProcessor, MathSymbolicRegressionProcessor
from recipe.algo_evolve.dataset.sample_strategy import get_solution_from_dataframe, RandomSampleStrategy, TopKSampleStrategy, NSGASampleStrategy, TopKDivSampleStrategy
from src.simplify import normalize_yaml, normalize_python_code

PROMPT_SAMPLER_CLASSES = {
    "comsol_*": ComsolProcessor,
    "math_SR_*": MathSymbolicRegressionProcessor,
    "math_*": MathProgramProcessor,
    "ai_*": AIProgramProcessor,
}

class DynamicRLHFDataset(RLHFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = self.config.custom_cls.init_kwargs.is_train
        if not self.enabled: # Behave just like RLHFDataset
            return
        
        self.log_path = self.config.custom_cls.init_kwargs.log_path
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)

        self.records = {
            "sampled_count": [0] * len(self.dataframe),
            "source_index": [-1] * len(self.dataframe),  # To track the original index of the data item, -1 for initial samples
        }

        self.resp_normalize_func = {
            'python': normalize_python_code,
            'yaml': normalize_yaml
        }[self.config.custom_cls.init_kwargs.language]

        prompt_sampler_config = dict(self.config.get('dynamic_prompt_sampler', {}))
        sample_strategy_config = dict(self.config.get('dynamic_sample_strategy', {'type': "Random"}))

        # This instance now handles a single data_source, determined from the first item.
        data_source = self.dataframe[0]['data_source']

        if prompt_sampler_config:
            # Determine the appropriate prompt sampler class based on data source
            for pattern, prompt_sampler_cls in PROMPT_SAMPLER_CLASSES.items():
                if re.match(pattern, data_source):
                    self.prompt_sampler = prompt_sampler_cls(**prompt_sampler_config)
                    break
            else:
                raise ValueError(f"No matching prompt sampler found for data source: {data_source}")
        else:
            self.prompt_sampler = (lambda batch:datasets.Dataset.from_dict(batch))

        sample_strategy_type = sample_strategy_config.pop('type')
        if sample_strategy_type == "Random":
            self.sample_strategy = RandomSampleStrategy(self.dataframe, self.config.train_batch_size, 
                                                        **sample_strategy_config)
        elif sample_strategy_type == "TopK":
            self.sample_strategy = TopKSampleStrategy(self.dataframe, self.config.train_batch_size,
                                                        **sample_strategy_config)
        elif sample_strategy_type == "NSGA":
            self.sample_strategy = NSGASampleStrategy(self.dataframe, self.config.train_batch_size,
                                                        **sample_strategy_config)
        elif sample_strategy_type == "TopKDiv":
            self.sample_strategy = TopKDivSampleStrategy(self.dataframe, self.config.train_batch_size,
                                                        **sample_strategy_config)

        self.filter_invalid_responses = self.config.get('filter_invalid_responses', True)
        self.filter_overlong_prompts_strategy = self.config.get('filter_overlong_prompts_strategy', 'Truncate')
        self.filter_duplicate_responses = self.config.get('filter_duplicate_responses', True)
        if self.filter_duplicate_responses:
            self.seen_hashes: Set[int] = set()
            self._update_hashes(self.dataframe)

    def __getitem__(self, idx):
        if not self.enabled: return super().__getitem__(idx)
        interpreted_idx = self.sample_strategy(idx)
        result = super().__getitem__(interpreted_idx)
        result['extra_info']['source_index'] = interpreted_idx # Add source index for retrival of original data item it based on
        return result
    
    def __len__(self):
        if not self.enabled: return super().__len__()
        return self.config.train_batch_size
    
    def _update_hashes(self, dataframe):
        """Update the seen hashes with the new dataframe"""
        new_idx = []
        solutions = get_solution_from_dataframe(dataframe)
        for i, solution in enumerate(solutions):
            # Try to calculate Hash for single solution:
            solution = self.resp_normalize_func(solution)
            h = xxhash.xxh64(solution).intdigest()
            
            # Update Dict
            if h not in self.seen_hashes:
                new_idx.append(i)
                self.seen_hashes.add(h)
        return new_idx

    def log(self):
        if not self.log_path:
            return
        
        self.sample_strategy.log(self.log_path)

        with open(os.path.join(self.log_path, f'rl_dataset_{len(self.dataframe)}.txt'), 'w') as f:
            f.write(f"Sampled Count: {self.records['sampled_count']}\n")
            f.write(f"Total samples: {sum(self.records['sampled_count'])}\n")
            f.write(f"Max samples for an item: {max(self.records['sampled_count'])}\n")
            sorted_counts = sorted(self.records['sampled_count'])
            for p in range(10, 100, 10):
                idx = int(len(sorted_counts) * p / 100)
                f.write(f"{p}% percentile: {sorted_counts[idx]}\n")
            f.write(f"Source Index: {self.records['source_index']}\n")

    def extend(self, data):
        if not self.enabled: raise ValueError("Cannot extend dataset when disabled")
        
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Generate New DataFrame
        source_index, responses = [], []
        for i in range(len(data)):
            source_index.append(data.non_tensor_batch["extra_info"][i]['source_index'])
            response_str = self.tokenizer.decode(data.batch["responses"][i][:valid_response_lengths[i].item()], skip_special_tokens=True)
            responses.append(response_str)
        new_dataframe = self.prompt_sampler(data=data, src_dataframe=self.dataframe.select(source_index), responses=responses)
        source_index = np.array(source_index, dtype=np.int64)
        
        # filter out responses with zero format reward
        if self.filter_invalid_responses:
            validity = data.non_tensor_batch.get("validity", [1.0] * len(data))
            valid_idx = [i for i,f in enumerate(validity) if f > 0]
            new_dataframe = new_dataframe.select(valid_idx)
            source_index = source_index[valid_idx]
            print(f"\nfilter illegal responses, len: {len(new_dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            if self.filter_overlong_prompts_strategy == 'Filter': # Erase the prompts that are too long
                valid_idx = []
                for i, doc in enumerate(new_dataframe):
                    if len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length:
                        valid_idx.append(i)
                new_dataframe = new_dataframe.select(valid_idx)
                source_index = source_index[valid_idx]
                print(f"\nfilter overlong prompts, len: {len(new_dataframe)}")
            elif self.filter_overlong_prompts_strategy == 'Truncate': # Truncate the prompts to max length
                def truncate_prompt(doc):
                    messages = doc[prompt_key]  # [{"role": ..., "content": ...}, ...]
                    truncated_messages = []
                    total_tokens = 20 # Reserve some tokens for the generation prompt

                    for msg in messages:
                        # First calculate the token count of the current message
                        msg_tokens = tokenizer.apply_chat_template([msg], add_generation_prompt=False)
                        token_count = len(msg_tokens)

                        if total_tokens + token_count <= self.max_prompt_length:
                            # Add the entire message directly
                            truncated_messages.append(msg)
                            total_tokens += token_count
                        else:
                            # Only truncate part of the content
                            remaining_tokens = self.max_prompt_length - total_tokens
                            if remaining_tokens > 0:
                                # Truncate content by tokens
                                content_tokens = tokenizer.encode(msg["content"])
                                truncated_content_tokens = content_tokens[:remaining_tokens]
                                truncated_content = tokenizer.decode(truncated_content_tokens)
                                truncated_messages.append({
                                    "role": msg["role"],
                                    "content": truncated_content + "[truncated due to length limit]"
                                })
                                total_tokens = self.max_prompt_length
                            break  # Already reached max length, no more additions

                    doc[prompt_key] = truncated_messages
                    return doc
                
                new_dataframe = new_dataframe.map(
                    truncate_prompt,
                    num_proc=self.num_workers,
                    desc=f"Truncating prompts to max {self.max_prompt_length} tokens",
                )
            else:
                raise ValueError(f"Unknown filter_overlong_prompts_strategy: {self.filter_overlong_prompts_strategy}")
        
        # filter out duplicate responses
        if self.filter_duplicate_responses:
            valid_idx = self._update_hashes(new_dataframe)
            new_dataframe = new_dataframe.select(valid_idx)
            source_index = source_index[valid_idx]
            print(f"\nFilter duplicate responses len: {len(new_dataframe)}")

        # Extend the sample strategy and dataframe
        new_dataframe = self.sample_strategy.extend(new_dataframe)
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])
        self.records['sampled_count'].extend([0] * len(new_dataframe))
        self.records['source_index'].extend(source_index.tolist())
        for i in range(len(new_dataframe)):
            self.records['sampled_count'][source_index[i]] += 1

        # Log 
        self.log()
    
    def on_batch_end(self, batch):
        self.extend(batch)
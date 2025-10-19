#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
Math Program Rewards Base Class

This module defines a base reward evaluation class that can be inherited
to execute code modifications, submit jobs, parse outputs, and calculate rewards.
Different tasks can subclass this base class and override:
    - post_process: preprocess generated code
    - calculate_reward: calculate reward score and extra info
"""

import os
import re
import json
import time
import pickle
import base64
from typing import List, Tuple
from ..utils.cli import JobClient

class MathProgramRewards:
    """
    Base class for math program reward evaluation.

    Subclasses must override:
        - post_process
        - calculate_reward
    """

    def __init__(self, timeout=2, use_slurm=True):
        """
        Initialize the reward evaluator.

        Args:
            timeout (int): Timeout for each task in seconds.
            use_slurm (bool): Whether to submit jobs via SLURM.
        """
        self.timeout = timeout
        self.use_slurm = use_slurm
        if use_slurm:
            print("Using slurm to submit jobs to servers")
            self.job_client = JobClient()
            self.job_batchnum = 0

        self.__name__ = self.__class__.__name__

    # ================= Methods to override in subclasses =================

    def post_process(self, completions: List[str], names: List[str]) -> List[str]:
        """
        Preprocess generated code (default: return as is).

        Args:
            completions (List[str]): List of generated code strings.

        Returns:
            List[str]: Processed code list.
        """
        return completions

    def calculate_reward(self, stdout: str, success_replace_count: int):
        """
        Parse Final Reward, Message, and Validity from stdout.
        Requirements:
        - Must match exactly at the beginning of a line (no indentation)
        - Must be a single line match
        - Fallback to default values if any field is missing
        Returns:
        final_reward (float), {"message": str, "validity": float}
        """
        if not success_replace_count:
            return 0.0, {"message": "No valid modification detected", "validity": 0.0}

        timeout_pat = re.compile(r"CANCELLED AT .* DUE TO TIME LIMIT")
        if timeout_pat.search(stdout):
            return 0.0, {"message": "Execution time out", "validity": 0.0}
        
        number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

        reward_match = re.search(rf"^Final Reward:\s*({number_pattern})\s*$", stdout, re.MULTILINE)
        message_match = re.search(r"^Message:\s*(.*?)\s*$", stdout, re.MULTILINE)
        validity_match = re.search(rf"^Validity:\s*({number_pattern}|True|False)\s*$", stdout, re.MULTILINE)


        # Fallback to default values if not found
        final_reward = float(reward_match.group(1)) if reward_match else 0.0
        message = message_match.group(1).strip() if message_match else ""
        if validity_match:
            validity = validity_match.group(1)
            if validity in ["True", "False"]:
                validity = 1.0 if validity == "True" else 0.0
            else:
                validity = float(validity)
        else:
            validity = 0.0

        return final_reward, {"message": message, "validity": validity}
    # ======================================================================

    def get_results(self, code_snippets: List[str], names: List[str]):
        """
        Submit tasks and get execution results.
        """
        self.job_batchnum += 1
        if self.use_slurm:
            results = self.job_client.submit_batch(
                batch_id=f"batch{self.job_batchnum}_pid{os.getpid()}",
                tasks=[{"id": names[i] + f"_{i}", "code": code_snippets[i]} for i in range(len(names))],
                timeout=self.timeout
            )
        else:
            results = {}
        return results['results']

    def apply_diff(self, diff_blocks: List[Tuple[str, str]], original_code: str):
        """
        Apply SEARCH/REPLACE diff to the original code.

        Args:
            diff_blocks (str): Diff block in SEARCH/REPLACE format.
            original_code (str): Original source code.

        Returns:
            str: Modified code.
            success_replace_count (int): Number of successful replacements.
        """
        original_lines = original_code.split("\n")
        result_lines = original_lines.copy()
        success_replace_count = 0

        for search_text, replace_text in diff_blocks:
            search_lines = search_text.split("\n")
            replace_lines = replace_text.split("\n")

            for i in range(len(result_lines) - len(search_lines) + 1):
                if result_lines[i: i + len(search_lines)] == search_lines:
                    result_lines[i: i + len(search_lines)] = replace_lines
                    success_replace_count += 1
                    break

        return "\n".join(result_lines), success_replace_count

    def extract_diffs(self, diff_text: str) -> List[Tuple[str, str]]:
        """
        Extract (SEARCH, REPLACE) blocks from a diff string. Exclude text between <think> and </think>.
        """
        # Remove text between <think> and </think>
        diff_text = re.sub(r"<think>.*?</think>", "", diff_text, flags=re.DOTALL)
        # Extract diff blocks
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
        return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]

    def __call__(self, data_sources, solution_strs, prompt_strs, solution_ids, ground_truths, extra_infos, proj_base_path, **kwargs):
        """
        Main execution entry.
        """
        names = [info["name"] for info in extra_infos]
        orig_code = [info["solution"] for info in extra_infos]

        diff_blocks = [self.extract_diffs(sol) for sol in solution_strs]
        diff_results = [self.apply_diff(dif, orig) for dif, orig in zip(diff_blocks, orig_code)]
        code_snippets = [result[0] for result in diff_results]
        success_replace_counts = [result[1] for result in diff_results]
        code_processed = self.post_process(completions=code_snippets, names=names)
        batch_results = self.get_results(code_processed, names)
        stdouts = [out["stdout"] for out in batch_results]

        rewards = []
        for i, stdout in enumerate(stdouts):
            reward, extra_info = self.calculate_reward(stdout=stdout, success_replace_count=success_replace_counts[i])
            rewards.append({
                'score': float(reward),
                'diff': base64.b64encode(pickle.dumps(diff_blocks[i])).decode('ascii'), # Avoid try to convert to numpy array
                'code': code_snippets[i],
                'success_replace_count': success_replace_counts[i],
                **extra_info
            })

        self.log_to_files(names, prompt_strs, solution_strs, code_processed, stdouts, proj_base_path,
            [{'score': r['score'], 'validity': r['validity'], 'success_replace_count': r['success_replace_count']} for r in rewards])
        return rewards
    
    def log_to_files(self, names, prompts, completions, code_processed, stdouts, proj_base_path, rewards):
        log_path = f"{proj_base_path}/running_jobs/{time.strftime('%y%m%d_%H%M%S')}_batch{self.job_batchnum}_pid{os.getpid()}"
        os.makedirs(log_path, exist_ok=True)
        for i in range(len(prompts)):
            log_path_i = os.path.join(log_path, names[i] + f'{i}')
            os.makedirs(log_path_i, exist_ok=True)
            with open(os.path.join(log_path_i, 'prompt.txt'), 'w') as fp:
                fp.write(prompts[i])
            with open(os.path.join(log_path_i, 'completion.txt'), 'w') as fp:
                fp.write(completions[i])
            with open(os.path.join(log_path_i, 'code_processed.py'),'w') as fp:
                fp.write(code_processed[i])
            with open(os.path.join(log_path_i, 'stdout.txt'), 'w') as fp:
                fp.write(stdouts[i])
            with open(os.path.join(log_path_i, 'reward.json'), 'w') as fp:
                json.dump(rewards[i], fp, indent=4)
        with open(os.path.join(log_path, "reward_summary.csv"), "w") as fp:
            fp.write("Index\tScore\tValidity\tSuccess Replace Count\n")
            for i in range(len(names)):
                fp.write(f"{i}\t{rewards[i]['score']}\t{rewards[i]['validity']}\t{rewards[i]['success_replace_count']}\n")
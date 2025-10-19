import numpy as np
import re
import os
import time
import json

from typing import List, Tuple

from ..utils.cli import JobClient
from .math_rewards import MathProgramRewards

class SymbolicRegressionRewardsDiff(MathProgramRewards):
    """
    Reward class for Symbolic Regression task.
    """

    def __init__(self, problem_base_path, absolute_path_to_problems_in_cpu_cluster, **kwargs):
        super().__init__(**kwargs)
        self.problem_base_path = problem_base_path
        self.absolute_path_to_problems_in_cpu_cluster = absolute_path_to_problems_in_cpu_cluster

    def post_process(self, completions, names):
        """
        Preprocess generated code.
        Here: read the evaluation code and initial code. append the evaluation code to the back.
        """
        results = []
        for c, name in zip(completions, names):
            if match := re.match(r"math_SR_([-_\w]+)_(\w+)", name):
                problem_path = os.path.join(self.problem_base_path, match.group(1), match.group(2))
            else:
                raise ValueError(f"Invalid problem name format: {name}")

            with open(os.path.join(problem_path, 'initial_program.py')) as fp:
                all_code = fp.read()
                init_code_1 = all_code.split("# EVOLVE-BLOCK-START")[0].strip()
                init_code_2 = all_code.split("# EVOLVE-BLOCK-END")[1].strip()
            
            with open(os.path.join(problem_path, 'evaluator.py')) as fp:
                eval_code = fp.read().strip()
                keys_to_sub = ['X_TRAIN_EVAL_PATH', 'Y_TRAIN_EVAL_PATH', 'X_TEST_EVAL_PATH', 'Y_TEST_EVAL_PATH', 'X_OOD_TEST_EVAL_PATH', 'Y_OOD_TEST_EVAL_PATH']
                for key in keys_to_sub:
                    eval_code = re.sub(
                        rf"({key}\s*=\s*r'\.)(/problems/[^']+')",
                        rf"{key} = r'" + self.absolute_path_to_problems_in_cpu_cluster + r"\2",
                        eval_code
                    )

            results.append(init_code_1 + "\n" + c + "\n" + init_code_2 + '\n' + eval_code)
            
        return results

    def calculate_reward(self, stdout, success_replace_count):
        """
        Parse stdout to extract reward and evaluation metrics.
        Returns reward and dict with metrics including error_message as "messages".
        """
        # Define expected float keys with default values
        float_keys = {
            "can_run": 0.0,
            "optimization_success": 0.0,
            "nmse_train_score": 0.0,
            'nmse_train': 1e9,
            'acc_train': 0.0,
            'nmse_test': 1e9,
            'acc_test': 0.0,
            'nmse_ood_test': 1e9,
            'acc_ood_test': 0.0,
        }
        
        # Initialize result dictionary with default values
        result = {**float_keys, "validity": 0.0, "messages": None}
        
        if not success_replace_count:
            return -5.0, {**result, "messages": "No valid modification detected"}

        # Step 1: Extract the Evaluation Results section
        eval_match = re.search(r"Evaluation Results:\s*([\s\S]+?)(?=\n\S|\Z)", stdout)
        if not eval_match:
            return -5.0, {**result, "messages": "Evaluation Results not found"}
        
        eval_block = eval_match.group(1).strip()
        
        # Reset messages for success case
        result["messages"] = "Success"
        
        # Step 2: Parse key-value pairs from the evaluation block
        for line in eval_block.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Special handling for error_message
            if key == "error_message":
                result["messages"] = value
                continue
                
            # Handle float values
            if key in float_keys:
                try:
                    result[key] = float(value)
                except ValueError:
                    # Keep default value if parsing fails
                    pass
        
        # Step 3: Calculate validity
        result["validity"] = 1.0 if (
            result["can_run"] == 1.0 
            # and result["optimization_success"] == 1.0
        ) else 0.0
        
        # Step 4: Determine reward (-5.0 if invalid)
        if re.search(r"raw_mse_train: inf", stdout): # Optimization Failed
            reward = -5.0
        else:
            reward = max(-5.0, result["nmse_train_score"]) if result["validity"] == 1.0 else -5.0
        
        return reward, result

class SymbolicRegressionRewards:

    def __init__(self, problem_base_path, absolute_path_to_problems_in_cpu_cluster, timeout=2, use_slurm=True, find_last_match=True):
        """
        Initialize the reward evaluator.

        Args:
            timeout (int): Timeout for each task in seconds.
            use_slurm (bool): Whether to submit jobs via SLURM.
        """
        self.timeout = timeout
        self.use_slurm = use_slurm
        self.find_last_match=find_last_match
        if use_slurm:
            print("Using slurm to submit jobs to servers")
            self.job_client = JobClient()
            self.job_batchnum = 0
        
        self.problem_base_path = problem_base_path
        self.absolute_path_to_problems_in_cpu_cluster = absolute_path_to_problems_in_cpu_cluster
    
    def post_process(self, completions, names):
        """
        Preprocess generated code.
        Here: read the evaluation code and initial code. append the evaluation code to the back.
        """
        results = []
        for c, name in zip(completions, names):
            if match := re.match(r"math_SR_([-_\w]+)_(\w+)", name):
                problem_path = os.path.join(self.problem_base_path, match.group(1), match.group(2))
            else:
                raise ValueError(f"Invalid problem name format: {name}")

            with open(os.path.join(problem_path, 'initial_program.py')) as fp:
                all_code = fp.read()
                init_code_1 = all_code.split("# EVOLVE-BLOCK-START")[0].strip()
                init_code_2 = all_code.split("# EVOLVE-BLOCK-END")[1].strip()
            
            with open(os.path.join(problem_path, 'evaluator.py')) as fp:
                eval_code = fp.read().strip()
                keys_to_sub = ['X_TRAIN_EVAL_PATH', 'Y_TRAIN_EVAL_PATH', 'X_TEST_EVAL_PATH', 'Y_TEST_EVAL_PATH', 'X_OOD_TEST_EVAL_PATH', 'Y_OOD_TEST_EVAL_PATH']
                for key in keys_to_sub:
                    eval_code = re.sub(
                        rf"({key}\s*=\s*r'\.)(/problems/[^']+')",
                        rf"{key} = r'" + self.absolute_path_to_problems_in_cpu_cluster + r"\2",
                        eval_code
                    )

            results.append(init_code_1 + "\n" + c + "\n" + init_code_2 + '\n' + eval_code)
            
        return results

    def calculate_reward(self, stdout):
        """
        Parse stdout to extract reward and evaluation metrics.
        Returns reward and dict with metrics including error_message as "messages".
        """
        # Define expected float keys with default values
        float_keys = {
            "can_run": 0.0,
            "optimization_success": 0.0,
            "nmse_train_score": 0.0,
            'nmse_train': 1e9,
            'acc_train': 0.0,
            'nmse_test': 1e9,
            'acc_test': 0.0,
            'nmse_ood_test': 1e9,
            'acc_ood_test': 0.0,
        }
        
        # Initialize result dictionary with default values
        result = {**float_keys, "validity": 0.0, "messages": None}
        
        # Step 1: Extract the Evaluation Results section
        eval_match = re.search(r"Evaluation Results:\s*([\s\S]+?)(?=\n\S|\Z)", stdout)
        if not eval_match:
            return -5.0, {**result, "messages": "Evaluation Results not found"}
        
        eval_block = eval_match.group(1).strip()
        
        # Reset messages for success case
        result["messages"] = "Success"
        
        # Step 2: Parse key-value pairs from the evaluation block
        for line in eval_block.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            
            # Special handling for error_message
            if key == "error_message":
                result["messages"] = value
                continue
                
            # Handle float values
            if key in float_keys:
                try:
                    result[key] = float(value)
                except ValueError:
                    # Keep default value if parsing fails
                    pass
        
        # Step 3: Calculate validity
        result["validity"] = 1.0 if (
            result["can_run"] == 1.0 
            # and result["optimization_success"] == 1.0
        ) else 0.0
        
        # Step 4: Determine reward (-5.0 if invalid)
        if re.search(r"raw_mse_train: inf", stdout): # Optimization Failed
            reward = -5.0
        else:
            reward = max(-5.0, result["nmse_train_score"]) if result["validity"] == 1.0 else -5.0
        
        return reward, result
    
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

    def extract_code(self, completion, code_format='python'):
        pattern = re.compile(rf"```{code_format}\n(.*?)```", re.DOTALL)

        matches = pattern.findall(completion)
        extracted_answer = ""
        if len(matches) >= 1:
            extracted_answer = matches[-1] if self.find_last_match else matches[0]
        return extracted_answer

    def __call__(self, data_sources, solution_strs, prompt_strs, solution_ids, ground_truths, extra_infos, proj_base_path, **kwargs):
        """
        Main execution entry.
        """
        names = [info["name"] for info in extra_infos]

        code_snippets = [self.extract_code(s, code_format='python') for s in solution_strs]
        code_processed = self.post_process(completions=code_snippets, names=names)
        batch_results = self.get_results(code_processed, names)
        stdouts = [out["stdout"] for out in batch_results]

        rewards = []
        for i, stdout in enumerate(stdouts):
            reward, extra_info = self.calculate_reward(stdout=stdout)
            rewards.append({
                'score': float(reward),
                'code': code_snippets[i],
                **extra_info
            })

        self.log_to_files(names, prompt_strs, solution_strs, code_processed, stdouts, proj_base_path,
            [{'score': r['score'], 'validity': r['validity']} for r in rewards])
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
            fp.write("Index\tScore\tValidity\n")
            for i in range(len(names)):
                fp.write(f"{i}\t{rewards[i]['score']}\t{rewards[i]['validity']}\n")

import os
import re
import copy
import yaml
import pickle
import base64
import pathlib
from datasets import Dataset

class ComsolProcessor:
    def __init__(self, code_format='python', max_history_length=5):
        self.code_format = code_format
        self.max_history_length = max_history_length
    
    def generate_new_prompt(self, original_problem, history):
        """Generate the full prompt including historical trials"""
        # System prompt and original problem sections remain unchanged
        base_prompt = original_problem
        
        # Add historical trials section
        history_section = "\n\n## Historical Trials (Reference, Most recent first)"
        history_section += "\nSelect a trial below as a starting point. Your new trial **MUST include modifications**."
        history_section += "\nYour answer **Cannot be identical** to any historical trial, or 0 reward will be given."
        
        # Add each historical trial
        for idx, trial in enumerate(reversed(history[-self.max_history_length:]), start=1):
            history_section += f"\n\n### Trial {idx}"
            history_section += f"\nReward: {trial['reward']}"
            if trial.get('message'):
                history_section += f"\nMessage: {trial['message']}"
            history_section += f"\nResponse: ```{self.code_format}\n{trial['solution']}\n```"
        
        return base_prompt + history_section

    def make_new_data(self, src_item, extracted_code, reward, message=""):
        """Create new data item with updated fields"""

        new_item = copy.deepcopy(src_item)
        
        # Extract original problem description
        # Assume user message content contains the original problem description
        original_problem = new_item['prompt'][-1]['content'].split('\n## Historical Trials')[0].strip()
        
        # Prepare current trial data
        new_trial = {
            "reward": reward,
            "message": message,
            "solution": extracted_code
        }
        
        # Update history record - add new trial
        current_history = new_item["extra_info"].get("history", [])
        updated_history = current_history + [new_trial]
        
        # Update new item fields
        new_item['extra_info'].update(new_trial) # Update reward, message and solution
        new_item["extra_info"]["history"] = updated_history
        
        # Generate new prompt content containing history
        new_prompt_content = self.generate_new_prompt(original_problem, updated_history)
        
        # Build new prompt structure
        new_item['prompt'] = [
            {"role": "system", "content": new_item['prompt'][0]['content']},
            {"role": "user", "content": new_prompt_content}
        ]
        
        return new_item

    def __call__(self, data, src_dataframe, responses):
        """Process batch of data to create new dataset with history"""
        
        rewards = data.batch['token_level_scores'].sum(-1).cpu().tolist()
        messages = data.non_tensor_batch.get('messages', [''] * len(data))
        extracted_codes = data.non_tensor_batch.get('code_extracted', [''] * len(data))

        # Prepare new dataset
        new_data_list = []
        for i in range(len(data)):
            new_data = self.make_new_data(
                src_item=src_dataframe[i],
                extracted_code=extracted_codes[i],
                reward=rewards[i],
                message=messages[i]
            )
            new_data_list.append(new_data)
        
        return Dataset.from_list(new_data_list, features=src_dataframe.features)

class MathProgramProcessor:
    def __init__(self, max_history_length=5):
        """
        Args:
            max_history_length (int): Maximum number of history trials to include.
            problem_base_path (str): Base directory where original .md problems are stored.
        """
        self.max_history_length = max_history_length
        self.problem_base_path = (pathlib.Path(__file__).parent / "../../../data/mathprogram").resolve()
        self.history_section = "\n\n## Historical Trials (Reference, Most recent first)"
        self.history_section += "\nYour answer **Cannot be identical** to any historical trial, or 0 reward will be given."
        self.history_section += "\nMethods different from the historical trials are encouraged. Try out new ideas!"
    
    def generate_new_prompt(self, original_problem, history):
        """Generate the full prompt including historical trials"""
        base_prompt = original_problem
        if not history:
            return base_prompt
        
        history_section = self.history_section

        for idx, trial in enumerate(reversed(history[-self.max_history_length:]), start=1):
            history_section += f"\n\n### Trial {idx}"
            history_section += f"\nReward: {trial['reward']}"
            if trial.get('message'):
                history_section += f"\nMessage: {trial['message']}"
            # Force language to python
            history_section += f"\nResponse: ```python\n{trial['solution']}\n```"
        
        return base_prompt + history_section

    def make_new_data(self, src_item, extracted_code, reward, message="", diff=[]):
        """Create new data item with updated fields"""
        new_item = copy.deepcopy(src_item)

        # Load original problem from .md file using extra_info.name
        problem_name = new_item["extra_info"]["name"]
        problem_path = os.path.join(self.problem_base_path, f"{problem_name}.md")
        if not os.path.exists(problem_path):
            raise FileNotFoundError(f"Problem file not found: {problem_path}")
        
        with open(problem_path, "r", encoding="utf-8") as f:
            original_problem = f.read().strip()
        
        new_trial = {
            "reward": reward,
            "message": message,
            "solution": extracted_code,
            "diffs": diff if diff else []
        }
        
        current_history = new_item["extra_info"].get("history", [])
        updated_history = current_history + [new_trial]
        
        new_item['extra_info'].update(new_trial)
        new_item["extra_info"]["history"] = updated_history
        
        original_problem = original_problem.format(current_program=extracted_code, current_status=f"Reward: {reward}, Message: {message}")
        new_prompt_content = self.generate_new_prompt(original_problem, current_history)
        
        new_item['prompt'] = [
            {"role": "system", "content": new_item['prompt'][0]['content']},
            {"role": "user", "content": new_prompt_content}
        ]
        
        return new_item

    def __call__(self, data, src_dataframe, responses):
        """Process batch of data to create new dataset with history"""
        
        messages = data.non_tensor_batch.get('messages', [''] * len(data))
        extracted_codes = data.non_tensor_batch.get('code', [''] * len(data))
        diffs = data.non_tensor_batch.get('diff', [None] * len(data))
        rewards = data.batch['token_level_scores'].sum(-1).cpu().tolist()
        
        new_data_list = []
        for i in range(len(data)):
            new_data = self.make_new_data(
                src_item=src_dataframe[i],
                extracted_code=extracted_codes[i],
                reward=rewards[i],
                message=messages[i],
                diff=pickle.loads(base64.b64decode(diffs[i].encode('ascii'))) if diffs[i] else []
            )
            new_data_list.append(new_data)
        
        return Dataset.from_list(new_data_list, features=src_dataframe.features)

class MathSymbolicRegressionProcessor(MathProgramProcessor):
    def __init__(self, max_history_length=5):
        """
        Args:
            max_history_length (int): Maximum number of history trials to include.
        """
        super().__init__(max_history_length=max_history_length)
        self.problem_base_path = (pathlib.Path(__file__).parent / "../../../data/mathprogram/symbolic_regression/problems").resolve()
    
    def make_new_data(self, src_item, extracted_code, reward, message="", diff=[]):
        """Create new data item with updated fields"""
        new_item = copy.deepcopy(src_item)

        # Load original problem from .md file using extra_info.name
        problem_name = new_item["extra_info"]["name"]
        if match := re.match(r"math_SR_([-_\w]+)_(\w+)", problem_name):
            problem_path = os.path.join(self.problem_base_path, match.group(1), match.group(2), 'config.yaml')
        else:
            raise ValueError(f"Invalid problem name format: {problem_name}")
        
        if not os.path.exists(problem_path):
            raise FileNotFoundError(f"Problem file not found: {problem_path}")
        
        with open(problem_path, "r", encoding="utf-8") as f:
            original_problem = yaml.safe_load(f)['prompt']['system_message'].replace("\\n", "\n").strip()
        
        new_trial = {
            "reward": reward,
            "message": message,
            "solution": extracted_code,
            "diffs": diff if diff else []
        }
        
        current_history = new_item["extra_info"].get("history", [])
        updated_history = current_history + [new_trial]
        
        new_item['extra_info'].update(new_trial)
        new_item["extra_info"]["history"] = updated_history
        
        original_problem = original_problem + \
            f"\n## Current Program\nStatus: Reward: {reward}, Message: {message}\n```python\n{extracted_code}\n```"
        new_prompt_content = self.generate_new_prompt(original_problem, current_history)
        
        new_item['prompt'] = [
            {"role": "system", "content": new_item['prompt'][0]['content']},
            {"role": "user", "content": new_prompt_content}
        ]
        
        return new_item

class AIProgramProcessor(MathProgramProcessor):
    def __init__(self, max_history_length=5):
        """
        Args:
            max_history_length (int): Maximum number of history trials to include.
            problem_base_path (str): Base directory where original .md problems are stored.
        """
        super().__init__(max_history_length=max_history_length)
        self.problem_base_path = (pathlib.Path(__file__).parent / "../../../data/ai_coding").resolve()
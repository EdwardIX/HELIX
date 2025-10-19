import os
import sys
import time
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from openai import OpenAI
import traceback
import subprocess
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.rewards.reward import custom_reward_fn

output_dir = ""
# ==========================
# Configuration
# ==========================
MODEL_NAME = "gpt-4o"  # "gpt-4o" | "qwen-14b" | "qwen-32b"
CONFIG = {
    "MODEL_NAME": MODEL_NAME,  # "gpt-4o" | "qwen-14b" | "qwen-32b"
    "API_URLS": {
        "gpt-4o": "https://xiaoai.plus/v1",
        "qwen-14b": "http://0.0.0.0:8081/v1",
        "qwen-32b": "http://0.0.0.0:8080/v1",
    },
    "API_KEY": {
        "gpt-4o": "Your Key Here",
        "qwen-14b": "",
        "qwen-32b": "",
    },
    "REAL_MODELNAME": {
        'gpt-4o': 'gpt-4o',
        "qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    },
    "MAX_RETRIES": 5,
    "OUTPUT_DIR": f"/workspace/data/models/direct_prompt_{MODEL_NAME}",
    "LOG_DIR": f"/workspace/data/models/direct_prompt_{MODEL_NAME}/logs",
    "PARALLEL": True,
    "MAX_WORKERS": 8,
    "Best Of N": 64,
}

TOKENIZER = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")


# ==========================
# Utility functions
# ==========================
def retry_with_backoff(func):
    def wrapper(*args, **kwargs):
        delay = 1
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == CONFIG["MAX_RETRIES"] - 1:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapper


# ==========================
# LLM Client (OpenAI SDK)
# ==========================
class LLMClient:
    def __init__(self):
        self.model_name = CONFIG["MODEL_NAME"]
        self.base_url = CONFIG["API_URLS"][self.model_name]
        self.api_key = CONFIG["API_KEY"][self.model_name]

        # Initialize OpenAI client
        if not self.api_key:
            self.client = OpenAI(api_key="dummy-key", base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @retry_with_backoff
    def generate(self, system_prompt: str, user_prompt: str):
        if not system_prompt or not user_prompt:
            raise ValueError("system_prompt and user_prompt cannot be empty")

        completion = self.client.chat.completions.create(
            model=CONFIG['REAL_MODELNAME'][self.model_name],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=16384
        )

        text = completion.choices[0].message.content

        return text


# ==========================
# File processing logic
# ==========================
def query_with_evaluate(llm: LLMClient, system_prompt: str, user_prompt: str, task_name: str, evaluator: callable):
    """Process single task, supporting Best Of N and parallel execution"""
    output_dir = Path(CONFIG["OUTPUT_DIR"]) / task_name / "raw_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [None] * CONFIG["Best Of N"]
    rewards = [None] * CONFIG["Best Of N"]

    # ====== Step 1: Check/generate outputs ======
    def run_single(idx):
        result_file = output_dir / f"output_{idx}.txt"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return f.read()
        else:
            result = llm.generate(system_prompt, user_prompt)
            with open(result_file, "w", encoding="utf-8") as f:
                f.write(result)
            return result

    if CONFIG["PARALLEL"]:
        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            futures = {executor.submit(run_single, i): i for i in range(CONFIG["Best Of N"])}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    print(f" Output ready: {task_name} trial {idx}")
                except Exception:
                    print(f" Error generating output {task_name} trial {idx}")
                    traceback.print_exc()
    else:
        for i in range(CONFIG["Best Of N"]):
            try:
                results[i] = run_single(i)
                print(f" Output ready: {task_name} trial {i}")
            except Exception:
                print(f" Error generating output {task_name} trial {i}")
                traceback.print_exc()

    # ====== Step 2: Check/generate rewards ======
    missing_eval_indices = []
    for i in range(CONFIG["Best Of N"]):
        reward_file = output_dir / f"reward_{i}.json"
        if reward_file.exists():
            with open(reward_file, "r", encoding="utf-8") as f:
                rewards[i] = json.load(f)
        else:
            missing_eval_indices.append(i)

    if missing_eval_indices:
        # Batch evaluate
        to_eval_results = [results[i] for i in missing_eval_indices]
        eval_rewards = evaluator(to_eval_results)

        # evaluator may return list/dict, here we assume it returns list[dict]
        for j, idx in enumerate(missing_eval_indices):
            rewards[idx] = eval_rewards[j]
            with open(output_dir / f"reward_{idx}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(rewards[idx], ensure_ascii=False, indent=2))

    # ====== Step 3: Return all scores ======
    scores = [r["score"] for r in rewards]
    return scores


def process_comsol():
    system_prompt = """You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. 
The final answer is a yaml file, respond in the following format and do not include any other text in the answer, an example is: <answer>\n```yaml\n...\n```</answer>."""

    data_source = Path("/workspace/data/comsol/pymodels/v6.3design")
    tasks = ["inductor", "demultiplexer_2d", "beam_bending_2d", "periodic_heat_3d", "magnetic_torque_2d"]
    # tasks = ["inductor"]

    llm = LLMClient()
    for task_name in tasks:
        md_file = data_source / f"{task_name}.md"
        if not md_file.exists():
            print(f" File not found: {md_file}")
            continue

        with open(md_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()

        def evaluator(batch_results):
            return custom_reward_fn(
                data_sources=[f"comsol_{task_name}"] * len(batch_results),
                solution_strs=batch_results,
                prompt_strs=[system_prompt + '\n' + user_prompt] * len(batch_results),
                solution_ids=[None] * len(batch_results),
                ground_truths=[None] * len(batch_results),
                extra_infos=[{"name": f"{task_name}"}] * len(batch_results),
                proj_base_path=CONFIG['LOG_DIR']
            )

        scores = query_with_evaluate(llm, system_prompt, user_prompt, f"comsol_{task_name}", evaluator)

        with open(f"{CONFIG['OUTPUT_DIR']}/comsol_{task_name}/best_scores.txt", "w", encoding="utf-8") as f:
            f.write(f"{md_file}: {max(scores)}\n")
            for i, s in enumerate(scores):
                f.write(f"  Trial {i}: {s}\n")
        print(f"Best Score of {md_file}: ", max(scores))

def extract_example_from_code(content):
    """
    Extract code blocks from prompt text that are enclosed by # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END
    """
    pattern = r"# EVOLVE-BLOCK-START(.*?)# EVOLVE-BLOCK-END"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        code_block = match.group(1).strip()
        return code_block
    else:
        raise ValueError("No code block found between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END")

def process_ai():
    system_prompt = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>.
IMPORTANT NOTICE: Always Use Search & Replace to modify the existing code! Don't just output a new code block."""

    data_source = Path("/workspace/data/ai_coding")
    tasks = ["adult_income", "bank_additional", "boston_housing"]
    # tasks = ["boston_housing"]

    llm = LLMClient()
    for task_name in tasks:
        md_file = data_source / f"{task_name}.md"
        py_file = data_source / f"{task_name}.py"
        if not md_file.exists() or not py_file.exists():
            print(f" File not found: {md_file} or {py_file}")
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            init_content = extract_example_from_code(f.read())

        with open(md_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()
            user_prompt = user_prompt.format(
                current_program=init_content, 
                current_status=f"Reward: Initial Program, Message: Success"
            )
            if task_name == 'bank_additional' and MODEL_NAME == 'qwen-14b':
                user_prompt += '\n\nTry Not to use pipelines or transformers, write all things step by step.'

        def evaluator(batch_results):
            return custom_reward_fn(
                data_sources=[f"ai_{task_name}"] * len(batch_results),
                solution_strs=batch_results,
                prompt_strs=[system_prompt + '\n' + user_prompt] * len(batch_results),
                solution_ids=[None] * len(batch_results),
                ground_truths=[None] * len(batch_results),
                extra_infos=[{"name": f"{task_name}", "solution": init_content}] * len(batch_results),
                proj_base_path=CONFIG['LOG_DIR']
            )

        scores = query_with_evaluate(llm, system_prompt, user_prompt, f"ai_{task_name}", evaluator)

        with open(f"{CONFIG['OUTPUT_DIR']}/ai_{task_name}/best_scores.txt", "w", encoding="utf-8") as f:
            f.write(f"{md_file}: {max(scores)}\n")
            for i, s in enumerate(scores):
                f.write(f"  Trial {i}: {s}\n")
        print(f"Best Score of {md_file}: ", max(scores))

def process_circle():
    system_prompt = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. 
IMPORTANT NOTICE: Always Use Search & Replace to modify the existing code! Don't just output a new code block."""

    data_source = Path("/workspace/data/mathprogram")
    tasks = ["circle_packing_disk", "circle_packing"]

    llm = LLMClient()
    for task_name in tasks:
        md_file = data_source / f"{task_name}.md"
        py_file = data_source / f"{task_name}.py"
        if not md_file.exists() or not py_file.exists():
            print(f" File not found: {md_file} or {py_file}")
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            init_content = extract_example_from_code(f.read())

        with open(md_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()
            user_prompt = user_prompt.format(
                current_program=init_content, 
                current_status=f"Reward: Initial Program, Message: Success"
            )

        def evaluator(batch_results):
            return custom_reward_fn(
                data_sources=[f"math_{task_name}"] * len(batch_results),
                solution_strs=batch_results,
                prompt_strs=[system_prompt + '\n' + user_prompt] * len(batch_results),
                solution_ids=[None] * len(batch_results),
                ground_truths=[None] * len(batch_results),
                extra_infos=[{"name": f"{task_name}", "solution": init_content}] * len(batch_results),
                proj_base_path=CONFIG['LOG_DIR']
            )

        scores = query_with_evaluate(llm, system_prompt, user_prompt, f"math_{task_name}", evaluator)

        with open(f"{CONFIG['OUTPUT_DIR']}/math_{task_name}/best_scores.txt", "w", encoding="utf-8") as f:
            f.write(f"{md_file}: {max(scores)}\n")
            for i, s in enumerate(scores):
                f.write(f"  Trial {i}: {s}\n")
        print(f"Best Score of {md_file}: ", max(scores))

def process_func():
    system_prompt = """You are an expert software developer tasked with iteratively improving a codebase.
Your job is to analyze the current program and suggest improvements based on feedback from previous attempts.
Focus on making targeted changes that will increase the program's performance metrics.
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. 
IMPORTANT NOTICE: Always Use Search & Replace to modify the existing code! Don't just output a new code block."""

    data_source = Path("/workspace/data/mathprogram")
    tasks = ["eggholder_function", "mishras_bird_function", "keanes_bump_10d", "keanes_bump_20d", "keanes_bump_30d"]

    llm = LLMClient()
    for task_name in tasks:
        subprocess.run(['python', 'make_data.py', task_name], cwd=str(data_source / "minimize_func"), check=True)
        md_file = data_source / f"minimize_func.md"
        py_file = data_source / f"minimize_func.py"
        if not md_file.exists() or not py_file.exists():
            print(f" File not found: {md_file} or {py_file}")
            continue

        with open(py_file, "r", encoding="utf-8") as f:
            init_content = extract_example_from_code(f.read())

        with open(md_file, "r", encoding="utf-8") as f:
            user_prompt = f.read()
            user_prompt = user_prompt.format(
                current_program=init_content, 
                current_status=f"Reward: Initial Program, Message: Success"
            )

        def evaluator(batch_results):
            return custom_reward_fn(
                data_sources=[f"math_minimize_func"] * len(batch_results),
                solution_strs=batch_results,
                prompt_strs=[system_prompt + '\n' + user_prompt] * len(batch_results),
                solution_ids=[None] * len(batch_results),
                ground_truths=[None] * len(batch_results),
                extra_infos=[{"name": task_name, "solution": init_content}] * len(batch_results),
                proj_base_path=CONFIG['LOG_DIR']
            )

        scores = query_with_evaluate(llm, system_prompt, user_prompt, f"math_{task_name}", evaluator)

        with open(f"{CONFIG['OUTPUT_DIR']}/math_{task_name}/best_scores.txt", "w", encoding="utf-8") as f:
            f.write(f"{md_file}: {max(scores)}\n")
            for i, s in enumerate(scores):
                f.write(f"  Trial {i}: {s}\n")
        print(f"Best Score of {md_file}: ", max(scores))

def process_symbolic():
    system_prompt = """You are an expert software developer. Your job is to write a Python function based on feedback from previous attempts.
Write your code in exactly the following format:
```python
# your code
```
Your code's execution time is limited, so pay attention to runtime efficiency!
If you use new packages, please import them.
Ensure the program still contains the func() function and produces the same outputs; other functions can be added, deleted, or modified freely.
IMPORTANT: The current task is a symbolic regression problem. Write a Python expression in func() where parameter scales are as similar as possible (use linear scaling or translation if needed). This helps later optimization when all parameters are initialized randomly in [0,1].
Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>. """
    current_program_prompt = """## Current Program
Status: {current_status}
```python
{current_program}
```"""

    data_source = Path("/workspace/data/mathprogram/symbolic_regression")
    categories = ['phys_osc', "matsci", "bio_pop_growth", "chem_react"]

    llm = LLMClient()
    for category in categories:
        category_path = data_source / "problems" / category

        for i, name in enumerate(os.listdir(category_path)):
            task_path = category_path / name
            if not task_path.is_dir():
                continue
            
            yaml_file = task_path / f"config.yaml"
            py_file = task_path / f"initial_program.py"
            if not yaml_file.exists() or not py_file.exists():
                print(f" File not found: {yaml_file} or {py_file}")
                continue

            with open(py_file, "r", encoding="utf-8") as f:
                init_content = extract_example_from_code(f.read())

            with open(yaml_file, "r", encoding="utf-8") as f:
                user_prompt = yaml.safe_load(f)['prompt']['system_message']
                user_prompt = user_prompt.replace("\\n", "\n") + '\n' + current_program_prompt.format(
                    current_program=init_content,
                    current_status=f"Initial Program"
                )

            def evaluator(batch_results):
                return custom_reward_fn(
                    data_sources=[f'math_SR_{category}_{name}'] * len(batch_results),
                    solution_strs=batch_results,
                    prompt_strs=[system_prompt + '\n' + user_prompt] * len(batch_results),
                    solution_ids=[None] * len(batch_results),
                    ground_truths=[None] * len(batch_results),
                    extra_infos=[{"name": f'math_SR_{category}_{name}', "solution": init_content}] * len(batch_results),
                    proj_base_path=CONFIG['LOG_DIR']
                )

            scores = query_with_evaluate(llm, system_prompt, user_prompt, f"math_SR_{category}_{name}", evaluator)

            with open(f"{CONFIG['OUTPUT_DIR']}/math_SR_{category}_{name}/best_scores.txt", "w", encoding="utf-8") as f:
                f.write(f"{category}_{name}: {max(scores)}\n")
                for i, s in enumerate(scores):
                    f.write(f"  Trial {i}: {s}\n")
            print(f"Best Score of {category}_{name}: ", max(scores))

if __name__ == "__main__":
    # process_comsol()
    # process_ai()
    # process_circle()
    # process_func()
    process_symbolic()
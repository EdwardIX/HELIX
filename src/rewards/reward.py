import ray
import re
import pathlib

from src.rewards.comsol.comsol_reward_design import ComsolRewardDesign
from src.rewards.math.external_evaluator_rewards import ExternalEvaluatorRewards
from src.rewards.math.symbolic_regression import SymbolicRegressionRewards
from src.rewards.math.ai_coding_rewards import AICodingRewards
from src.rewards.utils.statistics import RewardMetricsHistActor

data_dir = (pathlib.Path(__file__).parent / "../../data").resolve()
# data_dir_cpu_cluster = "/workspace/data/SciLLM"
data_dir_cpu_cluster = data_dir

DATA_SOURCE_TO_REWARD_MAPPING = {
    r"comsol.*": (ComsolRewardDesign, {
        # "timeout": 120,
        "timeout": 300,
        "probes_path": f"{data_dir}/comsol/pymodels/v6.3design/design_reward_ref.json",
        "find_last_match": True
    }),
    r"math_circle_packing.*": (ExternalEvaluatorRewards, {
        "timeout": 60,
        "program_base_path": f"{data_dir}/mathprogram"
    }),
    r"math_minimize_func": (ExternalEvaluatorRewards, {
        "timeout": 120, # Timeout for evaluator, not single code execution
        "program_path": f"{data_dir}/mathprogram/minimize_func.py"
    }),
    r"math_SR_.*": (SymbolicRegressionRewards, {
        "timeout": 60, # Timeout for evaluator, not single code execution
        "problem_base_path": f"{data_dir}/mathprogram/symbolic_regression/problems",
        "absolute_path_to_problems_in_cpu_cluster": f"{data_dir_cpu_cluster}/mathprogram/symbolic_regression"
    }),
    r"ai_*": (AICodingRewards, {
        "timeout": 120, # Timeout for evaluator, not single code execution
        "program_base_path": f"{data_dir}/ai_coding",
        "absolute_path_to_datasets_in_cpu_cluster": f"{data_dir_cpu_cluster}/ai_coding/datasets",
    }),
}

def _find_reward_mapping(data_source: str):
    """
    Find the corresponding reward calculation class and initialization parameters based on data source name.
    
    Args:
        data_source: Data source name string.
    
    Returns:
        (reward_class, init_kwargs): Reward calculation class and initialization parameter dictionary.
    """
    for pattern, (reward_class, init_kwargs) in DATA_SOURCE_TO_REWARD_MAPPING.items():
        if re.match(pattern, data_source):
            return reward_class, init_kwargs
    raise ValueError(f"No reward mapping found for data source: {data_source}")

# Define Actor class to encapsulate reward calculator
@ray.remote
class RewardCalculatorActor:
    def __init__(self, reward_class, init_kwargs):
        self.calculator = reward_class(**init_kwargs)
    
    def compute_batch(self, data_sources, solution_strs, prompt_strs, solution_ids, ground_truths, extra_infos, proj_base_path, **kwargs):
        return self.calculator(
            data_sources=data_sources,
            solution_strs=solution_strs,
            prompt_strs=prompt_strs,
            solution_ids=solution_ids,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            proj_base_path=proj_base_path,
            **kwargs
        )

# Global cache references
reward_actor_cache = {}
hist_actor = None

def custom_reward_fn(data_sources, solution_strs, prompt_strs, solution_ids, ground_truths, extra_infos, proj_base_path, **kwargs):
    """
    Custom reward function that computes rewards based on the data sources and solution strings.
    Uses Ray for parallel computation and caches reward calculator instances.
    
    Args:
        data_sources: List of data sources. Different source of data will be mapped to different reward function classes.
        solution_strs: List of solution strings. Passed to reward function
        prompt_strs: List of prompt strings (Added)
        solution_ids: List of solution tokens (Added)
        ground_truths: List of ground truth strings. Passed to reward function
        extra_infos: List of extra information dictionaries. Passed to reward function
        proj_base_path: The base log path of the project
        **kwargs: Additional keyword arguments for specific reward computation.

    Returns:
        List of computed rewards (dict with extra info).
    """
    global reward_actor_cache
    global hist_actor
    if hist_actor is None:
        hist_actor = RewardMetricsHistActor.remote()
    
    # Group data: group by data source
    source_groups = {}
    for idx, source in enumerate(data_sources):
        if source not in source_groups:
            source_groups[source] = {
                "indices": [],
                "solution_strs": [],
                "prompt_strs": [],
                "solution_ids": [],
                "ground_truths": [],
                "extra_infos": []
            }
        group = source_groups[source]
        group["indices"].append(idx)
        group["solution_strs"].append(solution_strs[idx])
        group["prompt_strs"].append(prompt_strs[idx])
        group["solution_ids"].append(solution_ids[idx])
        group["ground_truths"].append(ground_truths[idx])
        group["extra_infos"].append(extra_infos[idx])
    
    # Initialize or get cached Actor
    for source in source_groups.keys():
        if source not in reward_actor_cache:
            reward_class, init_kwargs = _find_reward_mapping(source)
            reward_actor_cache[source] = RewardCalculatorActor.remote(
                reward_class, init_kwargs
            )
    
    # Compute rewards for all groups in parallel
    futures = {}
    for source, group_data in source_groups.items():
        actor = reward_actor_cache[source]
        future = actor.compute_batch.remote(
            data_sources=[source] * len(group_data["indices"]),
            solution_strs=group_data["solution_strs"],
            prompt_strs=group_data["prompt_strs"],
            solution_ids=group_data["solution_ids"],
            ground_truths=group_data["ground_truths"],
            extra_infos=group_data["extra_infos"],
            proj_base_path=proj_base_path,
            **kwargs
        )
        futures[source] = future
    
    # Collect results
    results = {}
    for source, future in futures.items():
        group_results = ray.get(future)
        results[source] = {
            "indices": source_groups[source]["indices"],
            "rewards": group_results
        }
    
    # Merge results and maintain original order
    final_rewards = [None] * len(data_sources)
    for source, result_data in results.items():
        for idx, reward in zip(result_data["indices"], result_data["rewards"]):
            final_rewards[idx] = {"data_source": source, **reward} # Add data_source to extra info

    # Handle reward metrics visualization
    hist_actor.process_reward_metrics.remote(data_sources=data_sources, 
                                             sample_inputs=solution_strs, 
                                             infos_dict={k: [final_rewards[idx][k] for idx in range(len(final_rewards))] 
                                                         for k in set.intersection(*[set(r.keys()) for r in final_rewards])},
                                             proj_base_path=proj_base_path)

    return final_rewards
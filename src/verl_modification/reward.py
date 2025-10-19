# Original module: verl.trainer.ppo.reward
# Modified to add new reward manager

import multiprocessing
from functools import partial

from verl import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn, default_compute_score

def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    elif reward_manager_name == "custom_batch": # MODIFIED: Added new reward manager
        from src.rewards.utils.rm_custom_batch import CustomBatchRewardManager

        reward_manager_cls = CustomBatchRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    if reward_manager_name == 'custom_batch': # MODIFIED: Added Log Path to reward manager
        return reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=final_compute_score,
            reward_fn_key=config.data.reward_fn_key,
            proj_base_path=config.trainer.default_local_dir,
            **reward_kwargs,
        )
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
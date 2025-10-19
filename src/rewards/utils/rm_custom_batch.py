# Modified from verl.workers.reward_manager

from verl.workers.reward_manager import BatchRewardManager

class CustomBatchRewardManager(BatchRewardManager):
    """
    Custom batch reward manager that extends the BatchRewardManager.
    Current function added:
    1. add prompts_str and valid_response_ids to verify function.
    """

    def __init__(self, proj_base_path, **kwargs):
        super().__init__(**kwargs)
        self.proj_base_path = proj_base_path

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        prompts_str = []
        responses_str = []
        solution_ids = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            solution_ids.append(valid_response_ids)
            responses_str.append(response_str)
            prompts_str.append(prompt_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            prompt_strs=prompts_str,
            solution_ids=solution_ids,
            ground_truths=ground_truths,
            extra_infos=extras,
            proj_base_path=self.proj_base_path,
            **self.reward_kwargs,
        )

        return scores
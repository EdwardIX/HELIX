import re

from .external_evaluator_rewards import ExternalEvaluatorRewards

class AICodingRewards(ExternalEvaluatorRewards):
    def __init__(self, absolute_path_to_datasets_in_cpu_cluster, **kwargs):
        super().__init__(**kwargs)
        self.absolute_path_to_datasets_in_cpu_cluster = absolute_path_to_datasets_in_cpu_cluster

    def calculate_reward(self, stdout: str, success_replace_count: int):
        final_reward, info = super().calculate_reward(stdout, success_replace_count)

        number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        acc_match = re.search(rf"^Accuracy:\s*({number_pattern})\s*$", stdout, re.MULTILINE)
        prec_match = re.search(rf"^Precision:\s*({number_pattern})\s*$", stdout, re.MULTILINE)
        rec_match = re.search(rf"^Recall:\s*({number_pattern})\s*$", stdout, re.MULTILINE)

        info['Accuracy'] = float(acc_match.group(1)) if acc_match else 0.0
        info['Precision'] = float(prec_match.group(1)) if prec_match else 0.0
        info['Recall'] = float(rec_match.group(1)) if rec_match else 0

        return final_reward, info

    def post_process(self, completions, names, **kwargs):
        results = super().post_process(completions, names, **kwargs)
        for i, res in enumerate(results):
            res = re.sub(
                r'DATASET_PATH\s*=\s*["\'].*?["\']',
                f'DATASET_PATH = "{self.absolute_path_to_datasets_in_cpu_cluster}"',
                res
            )
            results[i] = res
        
        return results
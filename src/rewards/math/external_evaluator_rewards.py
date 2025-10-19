#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os

from .math_rewards import MathProgramRewards

class ExternalEvaluatorRewards(MathProgramRewards):
    """
    Reward class for Circle Packing task.
    """

    def __init__(self, program_path=None, program_base_path=None, **kwargs):
        super().__init__(**kwargs)
        if program_path is None and program_base_path is None:
            raise ValueError("program_path or program_base_path must be provided")
        self.program_path = program_path
        self.program_base_path = program_base_path

    def post_process(self, completions, names, **kwargs):
        """
        Preprocess generated code.
        Here: append the evaluation code to the front and the back
        """
        if self.program_path is not None:
            with open(self.program_path) as fp:
                all_code = fp.read()
                eval_code_1 = all_code.split("# EVOLVE-BLOCK-START")[0].strip()
                eval_code_2 = all_code.split("# EVOLVE-BLOCK-END")[1].strip()
            
            results = []
            for c in completions:
                results.append(eval_code_1 + "\n" + c + "\n" + eval_code_2)
            return results
        else:
            results = []
            for c, name in zip(completions, names):
                with open(os.path.join(self.program_base_path, f"{name}.py")) as fp:
                    all_code = fp.read()
                    eval_code_1 = all_code.split("# EVOLVE-BLOCK-START")[0].strip()
                    eval_code_2 = all_code.split("# EVOLVE-BLOCK-END")[1].strip()
                
                results.append(eval_code_1 + "\n" + c + "\n" + eval_code_2)
            return results
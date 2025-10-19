#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import re
import os
import json
import time

from ..utils.cli import JobClient
from .post_process import PostProcessor, DictToComsolTransformer
from .err_msg import extract_error_info

class ComsolRewardDesign():
    def __init__(self, config=None, timeout=120, probes_path=None, materials_path=None, find_last_match=True, msg_buffer_size=4, enable_msg_logging=False):
        self.config = config
        self.timeout = timeout
        self.job_client = JobClient()
        self.job_batchnum = 0
        self.find_last_match = find_last_match
        self.msg_buffer = {}
        self.msg_buffer_size = msg_buffer_size
        self.enable_msg_logging = enable_msg_logging

        self.__name__ = "comsol_reward_design"
        if probes_path:
            self.probes_path = probes_path
            with open(self.probes_path, 'r') as f:
                self.probe_dict = json.load(f)
                self.msg_buffer = {k: [] for k in self.probe_dict.keys()}

        if materials_path:
            self.materials_path = materials_path
            with open(self.materials_path, 'r') as f:
                self.materials_dict = json.load(f)

        self.post_processor = PostProcessor({"remove_lines": False, "add_probe": False, "add_materials": False})
        self.dict_to_comsol = DictToComsolTransformer()
        self.code_format = 'yaml'

    def extract_code(self, completion, code_format='python'):
        pattern = re.compile(rf"```{code_format}\n(.*?)```", re.DOTALL)

        matches = pattern.findall(completion)
        extracted_answer = ""
        if len(matches) >= 1:
            extracted_answer = matches[-1] if self.find_last_match else matches[0]
        return extracted_answer

    def extract_runtime_message(self, stdouts):
        """TracebackDesign is InvalidSuccess"""
        pattern_design_invalid = re.compile(r'Design is [Ii]nvalid')

        messages = []
        for stdout in stdouts:
            lines = stdout.split('\n')
            err_info = extract_error_info(lines)

            if err_info.get('timeout'):
                messages.append('Timeout')
            elif err_info.get('err_msg'):
                messages.append(f"Error: {err_info['err_msg']}")
            else:
                for line in reversed(lines):
                    if pattern_design_invalid.search(line):
                        messages.append(line.strip())
                        break
                else:
                    messages.append('Success')

        return messages

    def parse_integral_probes_from_stdout(self, stdout: str) -> float:
        """stdout"""
        pattern = re.compile(
            r'Final Reward: ([+-]?[\d\.Ee+-]+)'
        )
        for line in stdout.split('\n'):
            if match := pattern.search(line):
                value = float(match.group(1))

                return value

        return 0

    def reward_format(self, code_snippets):
        reward_list = []
        
        for c in code_snippets:
            valid = False
            if not c.strip(): # Empty Code
                reward_list.append(0.0)
                continue
            
            # YAML
            if self.code_format == 'yaml':
                import yaml
                try:
                    yaml.safe_load(c)
                    valid = True
                except Exception:
                    pass
            
            # JSON
            elif self.code_format == 'json':
                try:
                    json.loads(c)
                    valid = True
                except json.JSONDecodeError:
                    pass
            
            # XMLXXE
            elif self.code_format == 'xml':
                import xml.etree.ElementTree as ET
                from io import StringIO
                try:
                    parser = ET.XMLParser()
                    parser.entity = {}  # 
                    ET.parse(StringIO(c), parser=parser)
                    valid = True
                except ET.ParseError:
                    pass
            
            # 0
            reward_list.append(1.0 if valid else 0.0)
        
        return reward_list

    def reward_probe(self, stdouts, names):

        reward_list = []
        for stdout, name in zip(stdouts, names):
            ref_val = self.probe_dict[name]

            reward = self.parse_integral_probes_from_stdout(stdout)
            reward = reward / ref_val

            reward_list.append(reward)

        return reward_list

    def query_err_msg_to_prompt(self, names):
        output_msgs = []
        for i, name in enumerate(names):
            err_msg_list = self.msg_buffer[name]
            if len(err_msg_list) and self.enable_msg_logging:
                # err_prompt = f'Notice: You should correct the following code becaue you encoutered an error with msgs last time when you try to solve it. Details are:\n'

                for i, msg in enumerate(err_msg_list):
                    # err_prompt+=f'Case {i}: err code: {msg[0]}, err msg: {msg[1]}\t'
                    output_msgs.append((name, msg[0], msg[1]))

            else:
                output_msgs.append((name, '', ''))
        return output_msgs

    def reward_errline(self, stdouts, names, code_processed):
        reward_list = []
        for name, stdout, code in zip(names, stdouts, code_processed):
            if 'create_model(model)' not in code:  # Exclude non-comsol code (e.g. empty code)
                reward_list.append(0.)
                continue

            err_info = extract_error_info(stdout.split('\n'))

            if err_info.get('err_msg') is None:  # Bug1: Wrong User Name
                if err_info.get('timeout'):
                    reward_list.append(0.)  # Timeout
                    # self.msg_buffer[name].append('timeout')
                else:
                    reward_list.append(1.)  # Correct: No Exception Found
                    # self.msg_buffer[name].append('correct')
                # print(self.msg_buffer[name])

                continue
            err_line = err_info.get('err_line', None)
            reward_list.append(min(err_line / len(code.split('\n')), 1.) if err_line is not None else 0.)
            self.msg_buffer[name].append([err_info['err_code'], err_info['err_msg']])
            if len(self.msg_buffer[name]) > self.msg_buffer_size:
                self.msg_buffer[name].pop(0)
            # print('GGGGGGGGGGGGGGGGG', self.msg_buffer[name])
        return reward_list

    def post_process(self, code_snippets, names):
        code_processed = []
        for code, name in zip(code_snippets, names):
            try:
                with open(os.path.join(os.path.dirname(self.probes_path), f"{name}.py"), 'r') as fp:
                    code_template = fp.read()
                    code_template_lines = code_template.split('\n')

                    # If YAML_STRING defined, replace it.
                    if any(line.strip().startswith('YAML_STRING') for line in code_template_lines):
                        start_index, end_index = None, None
                        for i, code_line in enumerate(code_template_lines):
                            if code_line.strip().startswith('YAML_STRING'):
                                start_index = i
                            elif code_line.strip().startswith(('"""', "'''")):
                                end_index = i
                                break
                        assert start_index is not None and end_index is not None, f"Error In Template File {name}, Wrong YAML_STRING Format"
                        code = '\n'.join(code_template_lines[:start_index] + [f'YAML_STRING = """\n{code}\n"""'] + code_template_lines[end_index+1:])
                    else:
                        # Insert the code snippet into the template (between # Start From Here and # End Here)
                        if self.code_format in ('yaml', 'json'): # Transform to comsol code
                            code = self.dict_to_comsol(code)
                        
                        start_index, end_index = None, None
                        for i, code_line in enumerate(code_template_lines):
                            if code_line.strip().startswith('# Start From Here'):
                                start_index = i
                            elif code_line.strip().startswith('# End Here'):
                                end_index = i
                        assert start_index is not None and end_index is not None, f"Error In Template File {name}, No Start/End Found"
                        
                        # Adjust code snippet indent to match the template
                        code_lines = code.split('\n')
                        code_lines = [f"{' ' * 4}{line.strip()}" for line in code_lines]

                        # Merge to one code
                        code = "\n".join(code_template_lines[:start_index+1] + code_lines + code_template_lines[end_index:])
                        # Post-process the code snippet
                        code = self.post_processor(code, name)

                code_processed.append(code)
            except Exception:
                code_processed.append(code)
        return code_processed

    def get_results(self, code_snippets, names):
        self.job_batchnum += 1
        results = self.job_client.submit_batch(
            batch_id=f"batch{self.job_batchnum}_pid{os.getpid()}",
            tasks=[{"id": names[i] + f'_{i}', "code": code_snippets[i]} for i in range(len(names))],
            timeout=self.timeout,
        )
        if results.get("error"):
            print(f"Error in batch submission: {results['error']}")
        return results['results']

    def __call__(self, data_sources, solution_strs, prompt_strs, solution_ids, ground_truths, extra_infos, proj_base_path, **kwargs):
        completions = solution_strs
        names = [info["name"] for info in extra_infos]

        code_snippets = [self.extract_code(c, code_format=self.code_format) for c in completions]
        code_processed = self.post_process(code_snippets, names)
        batch_results = self.get_results(code_processed, names)
        stdouts = [out['stdout'] for out in batch_results]
        probe_rewards = self.reward_probe(stdouts, names)
        format_rewards = self.reward_format(code_snippets)
        errline_rewards = self.reward_errline(stdouts, names, code_processed)
        messages = self.extract_runtime_message(stdouts)

        self.log_to_files(names, prompt_strs, completions, code_processed, stdouts, proj_base_path,
                          [{"probe": r_p, "format": r_f, "errline": r_e, "validity": float(r_f > 0 and r_e > 0.99)} for r_p, r_f, r_e in zip(probe_rewards, format_rewards, errline_rewards)])

        
        return [{
            'score': r_p + 0.0 * r_f + 0.0 * r_e,
            'messages': m,
            'validity': float(r_f > 0 and r_e > 0.99),
            'probe_rewards': r_p,
            'format_rewards': r_f,
            'errline_rewards': r_e,
            "code_extracted": c 
        } for r_p, r_f, r_e, m, c in zip(probe_rewards, format_rewards, errline_rewards, messages, code_snippets)]

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
            fp.write("Index\tProbe Reward\tErrline Reward\tFormat Reward\n")
            for i in range(len(names)):
                fp.write(f"{i}\t{rewards[i]['probe']}\t{rewards[i]['errline']}\t{rewards[i]['format']}\n")

    def __name__(self):
        return "comsol_reward_design"
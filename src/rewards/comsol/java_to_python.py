import os
import re
import ast

from typing import List

from src.rewards.comsol.gather_materials import process_materials

def process_float(match:re.Match):
    s = match.group()
    return f"{float(s):.3E}" if len(s) > 9 else s

# 
CONFIG = {
    "functions": {
        "remove_lines": True,
        "replace_lines": True,
        "filter_material": True,
        "remove_empty_lines": True,
    },
    "input_folder": "/data/SciLLM/comsol/javamodels/v6.3design",
    "output_folder": "/data/SciLLM/comsol/pymodels/v6.3design",
    "output_suffix": ".py",
    # remove if the line contains any of the patterns in keywords_to_remove
    "keywords_to_remove": [
        # r"model\.result\(.+\)\.run\(\);", # Exclude all plotting command
        r"model\.result", # Exclude all model.result, include all plotting command
        r"model\.description\(",
        r"model\.title\(",
        r"model\.modelPath\(",
        r"descr\(",
        r'"descr"',
        r'result.*set\("x.*label"',
        r'result.*set\("y.*label"',
        r'model.*material.*\.setPropertyInfo\(', # Remove material setPropertyInfo
        r'^(?!.*material\("[^"]*"\)\.label).*label\(', # Keep material label, remove other labels
        r'model\.view\("[^"]+"\)',
        r'model\.result\("[^"]+"\)\.set\("view", "[^"]+"\)', # Remove All views
        r'clearMesh\(\)',
        r'clearMeshes\(\)',
        r'clearSolution\(\)',
        r'clearSolutionData\(\)',
        r"^/\*.*\*/$", # Multiline Comment
        r"^//.*$", # Singleline Comment
    ],
    "keywords_to_replace": [
        (r"(?<![\d.])(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)", process_float), # Process Float
        (r'model\.param\((.*)\)\.set\((.*,.*), ".+"\);', r"model.param(\1).set(\2);"), # Remove param comment
        (r'Double\.POSITIVE_INFINITY', r"math.inf"),
        (r'Double\.NEGATIVE_INFINITY', r"-math.inf"),
        (r'Double\.NaN', r"math.nan"),
    ]
}

PYTHON_TEMPLATE = """import mph
from jpype import JArray, JInt, JDouble, JString
import math

def create_model(model):
{code_of_create_model}

if __name__ == "__main__":
    client = mph.start(cores=4)
    model = client.create("{file_name}").java
    create_model(model)
"""

def process_java_files(config: dict) -> None:
    """Java"""
    for root, _, files in os.walk(config["input_folder"]):
        java_files = [f for f in files if f.endswith(".java")]
        if not java_files:
            continue

        print(f"Processing directory: {root}")
        for java_file in java_files:
            input_path = os.path.join(root, java_file)
            output_path = build_output_path(config, root, java_file)
            
            print(f"  Processing file: {java_file}")
            process_single_file(input_path, output_path)

def build_output_path(config: dict, root: str, java_file: str) -> str:
    """"""
    relative_path = os.path.relpath(root, config["input_folder"])
    output_dir = os.path.join(config["output_folder"], relative_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(java_file)[0]
    return os.path.join(output_dir, base_name + config["output_suffix"])

def process_single_file(input_path: str, output_path: str) -> None:
    """"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"    : {e}")
        return

    # 1. 
    lines = merge_lines(lines)

    # 2. 
    if CONFIG["functions"]["remove_lines"]:
        lines = remove_lines(lines)

    # 3. 
    if CONFIG["functions"]["replace_lines"]:
        lines = replace_lines(lines)

    # 4. Convert to Python
    lines = convert_to_python(lines)

    # 5. 
    if CONFIG["functions"]["filter_material"]:
        mat, lines = process_materials(lines, input_path)

    # 6. Remove Empty
    if CONFIG['functions']['remove_empty_lines']:
        lines = remove_empty_lines(lines)

    # 7. 
    save_content(output_path, lines)

def merge_lines(lines: List[str]):
    ret_lines = []
    skip_to = -1
    for i in range(0, len(lines)):
        if i <= skip_to:
            continue
        
        if not lines[i].strip().endswith(';') and lines[i].strip().startswith('model'):
            res = lines[i]
            for j in range(i+1, len(lines)):
                res = res.rstrip() + lines[j].lstrip()
                skip_to = j
                if lines[j].strip().endswith((';')):
                    break
            ret_lines.append(res)
        else:
            ret_lines.append(lines[i])
    return ret_lines

def remove_lines(lines: List[str]):
    ret_lines = []
    for line in lines:
        if not any(re.search(keyword, line.strip()) for keyword in CONFIG['keywords_to_remove']):
            ret_lines.append(line)
    return ret_lines

def replace_lines(lines: List[str]):
    for keyword, replacement in CONFIG['keywords_to_replace']:
        for i, line in enumerate(lines):
            lines[i] = re.sub(keyword, replacement, line)
    return lines

def remove_empty_lines(lines: List[str]): 
    """Remove adjacent empty lines"""
    ret_lines = []
    if lines[0].strip():
        ret_lines.append(lines[0])
    for i in range(1, len(lines)-1):
        if not lines[i].strip() and not lines[i+1].strip():
            continue
        ret_lines.append(lines[i])
    if lines[-1].strip():
        ret_lines.append(lines[-1])
    return ret_lines

def replace_java_keywords(java_code):
    # 
    pattern = re.compile(
        r'("(?:[^"\\]|\\.)*")'  # 
        r"|('(?:[^'\\]|\\.)*')"  # 
        r'|([^"\']+)',           # 
        re.DOTALL
    )
    
    processed = []
    for match in pattern.finditer(java_code):
        dq_str, sq_str, non_str = match.groups()
        if dq_str:
            processed.append(dq_str)  # 
        elif sq_str:
            processed.append(sq_str)  # 
        elif non_str:
            # 
            replaced = re.sub(r'\btrue\b', 'True', non_str)
            replaced = re.sub(r'\bfalse\b', 'False', replaced)
            replaced = re.sub(r'\bnull\b', 'None', replaced)
            processed.append(replaced)
    
    return ''.join(processed)

def convert_to_python_line(line):
    """"""
    # 
    processed = line.rstrip('\r\n').rstrip(' ;\t')
    
    # 
    processed = replace_java_keywords(processed)
    
    # 
    if re.search(r'new\s+([a-zA-Z0-9_]+)\s*((?:\[\s*\]\s*)+)\s*', processed):
        processed = re.sub(
            r'new\s+([a-zA-Z0-9_]+)\s*((?:\[\s*\]\s*)+)\s*',
            '',
            processed
        )
        processed = re.sub(r'{', '[', processed)
        processed = re.sub(r'}', ']', processed)
    
    # 
    processed = re.sub(
        r'new\s+double\s*\[\]\s*\{([^}]+)\}',
        lambda m: f'[{m.group(1)}]',
        processed
    )

    # ast
    space = len(processed) - len(processed.lstrip())
    tree = ast.parse(processed.lstrip())
    processed = ' ' * space + ast.unparse(tree)
    
    return processed

def convert_to_python(lines: List[str]):
    ret_lines = []
    for line in lines:
        sline = line.strip()
        if sline and (not sline.startswith('model.') or not sline.endswith(';')):
            continue
        
        ret_lines.append(convert_to_python_line(line) + '\n')
    return ret_lines

def save_content(output_path: str, lines: List[str]) -> None:
    """"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            code_of_create_model = "".join(lines)
            file_name = os.path.basename(output_path).replace(".py", ".mph")
            f.write(PYTHON_TEMPLATE.format(file_name=file_name, code_of_create_model=code_of_create_model))
        print(f"    : {output_path}")
    except Exception as e:
        print(f"    : {e}")
        raise e

if __name__ == "__main__":
    process_java_files(CONFIG)
    print("")
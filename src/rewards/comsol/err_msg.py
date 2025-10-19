import re
import os
import json

def is_error_line(line):
    error_pattern = re.compile(
        r'^(\w+Error\b|com\.|\b(Exception|Error|Warning)\b)',
        re.IGNORECASE | re.UNICODE
    )
    return bool(error_pattern.match(line.strip()))

def extract_line_number(log_line):
    """
    
    - 'File "file.py", line 123'
    - 'File ".../file.py", line 45, in <module>'
    - '  File "file.py", line 89'
    """
    #  "line" 
    pattern = r'File ".*?", line (\d+)'
    match = re.search(pattern, log_line)
    return int(match.group(1)) if match else None

def extract_error_info(lines):
    # 
    result = {
        'err_code': None, # 
        'err_msg': None, # 
        'err_line': None, # 
        'timeout': False,
    }

    # 
    tracebacks = []
    error_blocks = []
    in_error_block = False
    current_error = []

    for i, line in enumerate(lines):
        # Traceback
        if line.startswith("Traceback"):
            tracebacks.append(i)
            in_error_block = True
            current_error = []
        # 
        elif is_error_line(line):
            if not in_error_block:  # 
                error_blocks.append({'start': i, 'lines': [line]})
            else:  # Traceback
                current_error.append(line)
                error_blocks.append({
                    'start': i,
                    'lines': current_error
                })
                in_error_block = False
        # 
        elif in_error_block:
            current_error.append(line)
        # 
        if 'TIME LIMIT' in line:
            result['timeout'] = True

    # 
    def extract_from_block(block):
        """"""
        code_line = None
        error_lines = []
        last_line_number = None
        
        # 
        for i in range(block['start'], max(-1, block['start']-5), -1):
            if lines[i].startswith(('    ', '        ')) and ('File "' not in lines[i]) and ('^' not in lines[i]):
                code_line = lines[i].strip()
                break
        for line in block['lines']:
            line_num = extract_line_number(line)
            if line_num is not None:
                last_line_number = line_num
            if is_error_line(line):
                error_lines.append(line)
        
        return code_line, '\n'.join(error_lines), last_line_number

    # 
    for block in error_blocks[-3:]:  # 
        code, msg, line = extract_from_block(block)
        if code:
            result['err_code'] = code
        if msg:
            result['err_msg'] = msg
        if line:
            result['err_line'] = line

    return result

# def process_stdout_file(stdout_path, error_path):
#     try:
#         with open(stdout_path, 'r') as f:
#             input_lines = [line.rstrip('\n') for line in f]
#     except FileNotFoundError:
#         return

#     code, msg, timeout, err_line, create_model_line = extract_error_info(input_lines)
    
#     if err_line:
#         err_line_content = input_lines[err_line] if err_line < len(input_lines) else ""
#         err_line_num = extract_line_number(err_line_content)

#     if create_model_line:
#         create_model_line_content = input_lines[create_model_line] if create_model_line < len(input_lines) else ""
#         create_model_line_num = extract_line_number(create_model_line_content)
    
#     err_msg = ""
#     if code or msg:
#         code_snippet = code if code else msg.split(':')[0]
#         err_msg += f"Notice: Avoid code like ```{code_snippet}```, since there is error: ```{msg}```\n"
#     if timeout:
#         err_msg += "Notice: Optimize code to avoid timeout.\n"
    
#     if err_line:
#         err_msg += f"Stopped at line {err_line_num} (error)\n"

#     if err_line and create_model_line:
#         err_msg += f"Stopped at line {err_line_num} (error) / line {create_model_line_num} (create_model)\n"
    
#     with open(error_path, 'w') as f:
#         f.write(err_msg)

def process_stdout_file(stdout_path, error_path):
    try:
        with open(stdout_path, 'r') as f:
            input_lines = [line.rstrip('\n') for line in f]
    except FileNotFoundError:
        return

    result = extract_error_info(input_lines)
    
    with open(error_path, 'w') as f:
        json.dump(result, f, indent=4)

def main():
    root_dir = '/data/SciLLM/comsol/results/run_rltasks'
    for root, dirs, files in os.walk(root_dir):
        if 'stdout.txt' in files:
            stdout_file = os.path.join(root, 'stdout.txt')
            error_file = os.path.join(root, 'error.json')
            process_stdout_file(stdout_file, error_file)
            print(f"Processed: {stdout_file} -> {error_file}")

if __name__ == "__main__":
    main()
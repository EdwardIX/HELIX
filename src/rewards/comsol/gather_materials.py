import os
import re

from typing import List

from src.rewards.comsol.material_lib import MaterialLibSetDict #, MaterialLibTree, MaterialLibPropGroup

# 
CONFIG = {
    "input_folder": "/data/SciLLM/comsol/javamodels/v6.3",
    "material_lib_path": "/data/SciLLM/comsol/pymodels/mlib_temp.json",
    "material_pattern": [
        r"model.*material\(",
    ],
    "material_pattern_exclude": [
        r"model.*material.*\.comments\(",
        r'model.*material.*\.setPropertyInfo\(',
    ],
    "material_create": r"""material\(\)\.create\(['"]([^,]*)['"], ['"]([^,]*)['"].*\)""",
    "material_ops": r"""material\(\)\.(move|remove)\(['"]([^,]*)['"].*\)""",
    "material_with_name": r"""material\(['"]{material_name}['"]\).*""",
    "material_label": r"""material\(['"]{material_name}['"]\)\.label\(['"](.*)['"]\)""",
    "material_select": r"""material\(['"]{material_name}['"]\)*\.selection\(""",
    "material_properties": r"""^.*material\(['"]{material_name}['"]\)(\..*)$""",
    "material_label_basestr": r"""(.*)\.label\(['"].*['"]\)$""",
}

def process_java_files(config: dict) -> None:
    """Java"""
    lib = MaterialLibSetDict()
    for root, _, files in os.walk(config["input_folder"]):
        java_files = [f for f in files if f.endswith(".java")]
        if not java_files:
            continue

        print(f"Processing directory: {root}")
        for java_file in java_files:
            input_path = os.path.join(root, java_file)
            relative_path = os.path.join(os.path.relpath(root, config["input_folder"]), java_file)

            print(f"  Processing file: {java_file}")
            mats = process_single_file(input_path, relative_path, config)
            lib.update(mats, relative_path)
    
    lib.save(CONFIG['material_lib_path'])

def process_single_file(input_path: str, relative_path: str, config: dict) -> None:
    """"""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"    : {e}")
        return

    # 1. 
    from java_to_python import merge_lines, remove_lines, convert_to_python
    lines = merge_lines(lines)
    lines = remove_lines(lines)
    lines = convert_to_python(lines)

    # 2. materialkey-properties
    mats, lines_without_props = process_materials(lines, relative_path)

    return mats

def is_material_line(line: str):
    for pattern in CONFIG["material_pattern_exclude"]:
        if re.match(pattern, line.strip()):
            break
    else:
        for pattern in CONFIG["material_pattern"]:
            if re.match(pattern, line.strip()):
                return True
    return False

def process_materials(lines: List[str], relative_path:str):
    """materialkey-properties"""
    mats = {} # 
    ret_lines = [] # propertiesmats

    for line in lines: # Gather Names
        if not is_material_line(line):
            continue

        m_create = re.findall(CONFIG['material_create'], line.strip())
        if m_create:
            assert len(m_create) == 1
            material_name = m_create[0][0].strip()
            material_type = m_create[0][1].strip()
            assert material_name not in mats
            mats[material_name] = {"label": "No Label: " + line.strip(), "type": material_type, "prop": []}
    
    for material_name in mats.keys(): # Gather Labels
        for line in lines:
            m_label = re.findall(CONFIG['material_label'].format(material_name=material_name), line.strip())
            if m_label:
                mats[material_name]["label"] = m_label[0].strip()
                break # Only the first label is recoreded

    label_occured = {material_name: False for material_name in mats.keys()}
    for line in lines: # Process Other Lines
        ret_lines.append(line)
        if not is_material_line(line):
            continue
        
        line = line.strip()
        matched_names = [material_name for material_name in mats.keys()
                         if re.search(CONFIG['material_with_name'].format(material_name=material_name), line)]

        assert len(matched_names) <= 1
        if not matched_names: # Not belongs to any existing materials
            if not re.search(CONFIG['material_create'], line) and not re.search(CONFIG['material_ops'], line):
                with open("gather_materials_error_log.txt", "a") as f: # Not an material create/remove op
                    print(relative_path + "\n\t" + line, file=f)
            continue
        material_name = matched_names[0]

        m_label = re.search(CONFIG['material_label'].format(material_name=material_name), line)
        m_select = re.search(CONFIG['material_select'].format(material_name=material_name), line)
        
        if m_select:
            # Ignore selection lines 
            pass
        elif m_label: 
            # Ignore label lines (recoreded)
            # However, if multiple labels occured, record only the first one, delete the rest
            if not label_occured[material_name]:
                label_occured[material_name] = True
            else:
                ret_lines.pop()
        else:
            m_prop = re.fullmatch(CONFIG['material_properties'].format(material_name=material_name), line)
            if m_prop:
                mats[material_name]["prop"].append(m_prop.group(1))
            else:
                raise ValueError(f"    : {line}")
            
            if not mats[material_name]['label'].startswith("No Label") and mats[material_name]['type'] == "Common": # Keep Self-defined Materials(no label / non Common type)
                ret_lines.pop() # Remove this property line

    return mats, ret_lines

def add_materials(code, filename, lib):
    if not lib:
        return code
    
    if filename in lib['files']:
        f_entry = lib['files'][filename]
    else:
        f_entry = None
    
    lines = code.split('\n')
    mat_types = {}
    ret_lines = []
    for line in lines:
        ret_lines.append(line)
        if not is_material_line(line):
            continue

        m_create = re.findall(CONFIG['material_create'], line.strip()) # Create Options
        if m_create:
            assert len(m_create) == 1
            material_name = m_create[0][0].strip()
            material_type = m_create[0][1].strip()
            mat_types[material_name] = material_type
            continue

        matched_names = [name for name in mat_types.keys()
                         if re.findall(CONFIG['material_with_name'].format(material_name=name), line.strip())]
        assert len(matched_names) <= 1
        if not matched_names or mat_types.get(matched_names[0], None) != "Common": # Skip not defined / Not common materials
            continue
        material_name = matched_names[0]

        m_label = re.findall(CONFIG['material_label'].format(material_name=material_name), line.strip()) # Label found, insert!
        if m_label:
            material_label = m_label[0].strip()
            m_basestr = re.match(CONFIG['material_label_basestr'], line)
            if not m_basestr:
                continue
            basestr = m_basestr.group(1)
            
            # Insert props from lib
            if f_entry and material_name in f_entry:
                for prop in f_entry[material_name]["prop"]:
                    ret_lines.append(basestr + prop)
            elif material_label in lib['gather']:
                for prop, attr in lib['gather'][material_label]["properties"].items():
                    if isinstance(attr, dict): # Select the most frequent one
                        mx_item = max(attr.items(), key=lambda x: (x[1], x[0]))
                        prop = f"{prop}, {mx_item[0]})"
                    ret_lines.append(basestr + prop)
            else:
                pass # Material Not Found
    
    return "\n".join(ret_lines)

if __name__ == "__main__":
    process_java_files(CONFIG)
    print("")
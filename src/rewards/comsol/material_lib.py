import re
import os
from collections import defaultdict
import json

CONFIG = {
    'set': r"""(.*\.(set|setIndex|setEntry)\(['"][^'"]*['"]),\s*(.*)\)""",
    'create': r"""\.propertyGroup\(\)\.create\((['"][^'"]*['"]),\s*(['"][^'"]*['"])\)""",
}

class MaterialLibSetDict:
    def __init__(self):
        self.mlib = {
            'files': {},
            'gather': {},
        }
    
    def update(self, mats: dict, filename: str) -> None:
        """"""
        self.mlib['files'][os.path.basename(filename).replace('.java', '')] = mats
        mlib = self.mlib['gather']

        for mat in mats.values():
            label = mat["label"]
            mat_type = mat["type"]
            if mat_type != "Common":
                continue
            
            # 
            if label not in mlib:
                mlib[label] = {
                    "type": mat_type,
                    "properties": defaultdict(int),
                }
            
            # 
            for prop in mat["prop"]:
                prop = prop.strip()
                set_match = re.match(CONFIG['set'], prop)
                create_group_match = re.match(CONFIG['create'], prop)

                if set_match:
                    # 
                    prop_key, _, prop_value = set_match.groups()
                    if prop_key not in mlib[label]["properties"]:
                        mlib[label]["properties"][prop_key] = defaultdict(int)
                    mlib[label]["properties"][prop_key][prop_value] += 1
                elif create_group_match:
                    group_name, descr = create_group_match.groups()
                    prop = f".propertyGroup().create({group_name}, {group_name}, {descr})"
                    mlib[label]["properties"][prop] += 1
                else:
                    # 
                    mlib[label]["properties"][prop] += 1
    
    def save(self, filepath: str) -> None:
        """"""
        with open(filepath, "w") as f:
            json.dump(self.mlib, f, indent=4, ensure_ascii=False)

# 
if __name__ == "__main__":
    lib = MaterialLibSetDict()
    lib.update({
        "mat1": {
            "label": "Copper",
            "type": "Common",
            "prop": [
                ".propertyGroup().create(\"Enu\", \"Enu\", \"Young's modulus and Poisson's ratio\")",
                ".propertyGroup().create(\"linzRes\", \"linzRes\", \"Linearized resistivity\")",
                ".set(\"family\", \"copper\")",
                ".propertyGroup(\"def\").set(\"relpermeability\", [\"1\", \"0\", \"0\", \"0\", \"1\", \"0\", \"0\", \"0\", \"1\"])",
                ".propertyGroup(\"def\").set(\"electricconductivity\", [\"5.998e7[S/m]\", \"0\", \"0\", \"0\", \"5.998e7[S/m]\", \"0\", \"0\", \"0\", \"5.998e7[S/m]\"])",
                ".propertyGroup(\"def\").set(\"thermalexpansioncoefficient\", [\"17e-6[1/K]\", \"0\", \"0\", \"0\", \"17e-6[1/K]\", \"0\", \"0\", \"0\", \"17e-6[1/K]\"])",
                ".propertyGroup(\"def\").set(\"heatcapacity\", \"385[J/(kg*K)]\")",
                ".propertyGroup(\"def\").set(\"relpermittivity\", [\"1\", \"0\", \"0\", \"0\", \"1\", \"0\", \"0\", \"0\", \"1\"])",
                ".propertyGroup(\"def\").set(\"density\", \"8960[kg/m^3]\")",
                ".propertyGroup(\"def\").set(\"thermalconductivity\", [\"400[W/(m*K)]\", \"0\", \"0\", \"0\", \"400[W/(m*K)]\", \"0\", \"0\", \"0\", \"400[W/(m*K)]\"])",
                ".propertyGroup(\"Enu\").set(\"E\", \"110[GPa]\")",
                ".propertyGroup(\"Enu\").set(\"nu\", \"0.35\")",
                ".propertyGroup(\"linzRes\").set(\"rho0\", \"1.72e-8[ohm*m]\")",
                ".propertyGroup(\"linzRes\").set(\"alpha\", \"0.0039[1/K]\")",
                ".propertyGroup(\"linzRes\").set(\"Tref\", \"298[K]\")",
                ".propertyGroup(\"linzRes\").addInput(\"temperature\")"
            ]
        },
    }, "path/to/file.java")
    lib.save("material_lib.json")
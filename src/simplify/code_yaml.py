import yaml
from typing import Any, Optional

def normalize_yaml(yaml_content: str, indent: int = 2, sort_keys: bool = False) -> str:
    """
    Standardize YAML content:
    1. Delete all comments (PyYAML itself doesn't preserve comments)
    2. Unify indentation
    3. Optionally sort dictionary keys
    """
    try:
        # Load YAML as Python object
        data: Any = yaml.safe_load(yaml_content)

        # Re-dump YAML
        normalized = yaml.dump(
            data,
            indent=indent,
            default_flow_style=False,
            sort_keys=sort_keys,
            allow_unicode=True,
        )

        # Ensure trailing newline
        if not normalized.endswith("\n"):
            normalized += "\n"
        return normalized
    except Exception as e:
        print(f"Warning: Error in Normalizing Yaml: {type(e)}: {str(e)}")
        return yaml_content


# --- Example ---
if __name__ == "__main__":
    messy_yaml = """
# Top comment
person:
    name: Alice   # Internal comment
    age: 30
    address:
        city: Wonderland
        zip: 12345
"""
    print(normalize_yaml(messy_yaml))

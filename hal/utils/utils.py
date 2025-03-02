import re
import os
import json
from typing import Any

def move_merge_dirs(source_root, dest_root):
    for path, dirs, files in os.walk(source_root, topdown=False):
        dest_dir = os.path.join(
            dest_root,
            os.path.relpath(path, source_root)
        )
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for filename in files:
            os.rename(
                os.path.join(path, filename),
                os.path.join(dest_dir, filename)
            )
        for dirname in dirs:
            os.rmdir(os.path.join(path, dirname))
    os.rmdir(source_root)

def safe_filename(input_string):
    # Replace spaces with underscores
    transformed_string = input_string.replace(' ', '_')
    # Remove or replace any characters that are not safe for file names
    transformed_string = re.sub(r'[^\w\-\.]', '', transformed_string)
    return transformed_string


def make_json_serializable(obj: Any) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith('{') and obj.endswith('}')) or (obj.startswith('[') and obj.endswith(']')):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        # For custom objects, convert their __dict__ to a serializable format
        return {
            '_type': obj.__class__.__name__,
            **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}
        }
    else:
        # For any other type, convert to string
        return str(obj)
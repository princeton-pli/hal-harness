import re
import os

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
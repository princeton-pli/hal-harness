import os
import ast


def check_subprocess_usage(path):
    def check_file(file_path):
        with open(file_path, "r") as file:
            try:
                tree = ast.parse(file.read())
            except SyntaxError:
                print(f"Syntax error in file: {file_path}")
                return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess":
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module == "subprocess":
                    return True
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in [
                        "Popen",
                        "call",
                        "check_call",
                        "check_output",
                        "run",
                    ]:
                        return True

        return False

    def check_directory(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if check_file(file_path):
                        return file_path

            for dir in dirs:
                dir_path = os.path.join(root, dir)
                result = check_directory(dir_path)
                if result:
                    return result

        return None

    if os.path.isfile(path):
        return path if check_file(path) else None
    elif os.path.isdir(path):
        return check_directory(path)
    else:
        print(f"Invalid path: {path}")
        return None

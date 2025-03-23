# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.
from typing import Optional, List

import requests

# from smolagents.agents import ToolCallingAgent
from smolagents import ToolCallingAgent, tool, LiteLLMModel, DuckDuckGoSearchTool, CodeAgent, Tool
import subprocess
from pathlib import Path
import re

from prompts import USACO_PROMPT, PATCH_EXAMPLE


@tool
def execute_bash(command: str) -> str:
    """
    Description: Execute a bash command and return its output.
    Will not execute commands requiring internet access.
    Common linux and python packages are available via apt and pip.
    Args:
        command: The bash command to execute
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return f"Exit Code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def search_wikipedia(query: str) -> str:
    """
    Fetches a summary of a Wikipedia page for a given query.
    Args:
        query: The search term to look up on Wikipedia.
    Returns:
        str: A summary of the Wikipedia page if successful, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        title = data["title"]
        extract = data["extract"]

        return f"Summary for {title}: {extract}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia data: {str(e)}"
    
    
@tool
def edit_file(command: str, path: str, content: Optional[str] = None, 
              line_number: Optional[int] = None, old_str: Optional[str] = None,
              new_str: Optional[str] = None) -> str:
    """
    Edit files in the project with various operations.
    Args:
        command: One of 'view', 'create', 'str_replace', 'insert', 'delete'
        path: Path to the file to edit
        content: Content for create/insert operations
        line_number: Line number for insert/delete operations
        old_str: String to replace when using str_replace
        new_str: New string for replacement
    """
    path = Path(path)
    try:
        if command == "view":
            if path.is_file():
                with open(path, 'r') as f:
                    content = f.read()
                    return content[:1000] + ('...' if len(content) > 1000 else '')
            elif path.is_dir():
                files = list(path.rglob('*'))[:100]
                return "\n".join(str(f.relative_to(path)) for f in files)
            return f"Path {path} does not exist"

        elif command == "create":
            if path.exists():
                return f"Error: {path} already exists"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content or "")
            return f"Created file {path}"

        elif command == "str_replace":
            if not path.is_file():
                return f"Error: {path} is not a file"
            with open(path, 'r') as f:
                file_content = f.read()
            if old_str not in file_content:
                return f"Error: Could not find exact match for replacement string"
            new_content = file_content.replace(old_str, new_str)
            with open(path, 'w') as f:
                f.write(new_content)
            return f"Successfully replaced content in {path}"

        elif command == "insert":
            if not path.is_file():
                return f"Error: {path} is not a file"
            if line_number is None:
                return f"Error: Line number is required for insert operation"
            with open(path, 'r') as f:
                lines = f.readlines()
            if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines) + 1:
                return f"Error: Invalid line number {line_number}"
            lines.insert(line_number - 1, content + '\n')
            with open(path, 'w') as f:
                f.writelines(lines)
            return f"Inserted content at line {line_number} in {path}"
        
        elif command == "delete":
            if not path.is_file():
                return f"Error: {path} is not a file"
            if line_number is None:
                return f"Error: Line number is required for delete operation"
            with open(path, 'r') as f:
                lines = f.readlines()
            if not isinstance(line_number, int) or line_number < 1 or line_number > len(lines):
                return f"Error: Invalid line number {line_number}"
            del lines[line_number - 1]
            with open(path, 'w') as f:
                f.writelines(lines)
            return f"Deleted line {line_number} from {path}"

    except Exception as e:
        return f"Error performing {command} operation: {str(e)}"

@tool
def file_search(query: str, exclude_pattern: Optional[str] = "*.pyc,*.git*,__pycache__,*.bin,*.exe,*.dll,*.so") -> str:
    """
    Search files in the current directory and subdirectories for specific content.
    Args:
        query: The search term or regex pattern to look for
        exclude_pattern: Comma-separated file patterns to exclude from search (default: binaries and cache files)
    Returns:
        str: Matching passages with file paths and line numbers
    """
    if not query.strip():
        return "Error: Empty search pattern. Please provide a valid search term."
        
    results = []
    matches_found = 0
    files_searched = 0
    
    context_lines = 100
    max_matches = 10
    max_files = 50
    
    exclude_patterns = exclude_pattern.split(',') if exclude_pattern else []
    
    try:
        all_files = list(Path('.').rglob('*'))
        
        files_to_search = []
        for file_path in all_files:
            if not file_path.is_file():
                continue
                
            # Skip excluded patterns
            skip = False
            for pattern in exclude_patterns:
                pattern = pattern.strip()
                if file_path.match(pattern):
                    skip = True
                    break
            
            if not skip:
                files_to_search.append(file_path)
        
        # Limit to max_files
        files_to_search = files_to_search[:max_files]
        
        for file_path in files_to_search:
            if matches_found >= max_matches:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                files_searched += 1
                
                for i, line in enumerate(lines):
                    if matches_found >= max_matches:
                        break
                        
                    if re.search(query, line, re.IGNORECASE):
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        context = ''.join(lines[start:end])
                        results.append(f"File: {file_path} (line {i+1}):\n{context}\n---")
                        matches_found += 1
                        
            except (UnicodeDecodeError, IOError):
                # Skip binary or unreadable files
                continue
    
        if not results:
            return f"No matches found for '{query}' in {files_searched} files."
        
        summary = f"Found {matches_found} matches for '{query}' in {files_searched} files.\n\n"
        return summary + "\n".join(results)
    
    except Exception as e:
        return f"Error searching files: {str(e)}"


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    
    task_id, task = list(input.items())[0]
    
    results = {}
    
    model = LiteLLMModel(model_id=kwargs['model_name'])

    agent = ToolCallingAgent(
    tools=[
        DuckDuckGoSearchTool(),
        search_wikipedia,
        execute_bash,
        edit_file,
        file_search,
    ],
    add_base_tools=True,
    model=model)

    if kwargs['benchmark_name'] == 'usaco':
        prompt = USACO_PROMPT.format(task['description'])
        response = agent.run(prompt)
        
        # extract code from response
        if '```python' in response:
            response = response.split('```python')[1].split('```')[0]
    elif kwargs['benchmark_name'] == 'swebench_verified':
        pass
    elif kwargs['benchmark_name'] == 'swebench_verified_mini':
        process = subprocess.run(['git', 'clone', f'https://github.com/{task["repo"]}.git'], capture_output=True, text=True)
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to clone repository: {process.stderr}")
        
        process = subprocess.run(['git', 'reset', '--hard', task['base_commit']], capture_output=True, text=True)
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to reset repository: {process.stderr}")
        
        
        
        response = agent.run(
            f"""I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply.
            
Patch example:

{PATCH_EXAMPLE}
            
Problem: {task['problem_statement']}
            
The code of the project is cloned to your current directory. Please respond with a single patch file. Please do not include any other text or comments."""
        )
        
    elif kwargs['benchmark_name'] == 'appworld_test_normal':
        pass
    elif kwargs['benchmark_name'] == 'appworld_test_challenge':
        pass
    elif kwargs['benchmark_name'] == 'taubench_retail':
        pass
    elif kwargs['benchmark_name'] == 'taubench_airline':
        pass
    elif kwargs['benchmark_name'] == 'inspect_evals/gaia':
        pass
    elif kwargs['benchmark_name'] == 'inspect_evals/cybench':
        pass
    elif kwargs['benchmark_name'] == 'inspect_evals/appworld':
        pass
    elif kwargs['benchmark_name'] == 'inspect_evals/agentharm':
        pass
    elif kwargs['benchmark_name'] == 'inspect_evals/agentharm_benign':
        pass
    
    results[task_id] = response
        
    return results
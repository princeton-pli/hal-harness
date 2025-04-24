
from typing import Optional, List, Dict, Any
from functools import partial
import tiktoken

import subprocess
from pathlib import Path
import re

import json
import os

from typing import Optional

from hal.utils.weave_utils import MODEL_PRICES_DICT

from smolagents import CodeAgent, tool, LiteLLMModel, DuckDuckGoSearchTool, CodeAgent, Tool, PythonInterpreterTool, VisitWebpageTool, GoogleSearchTool
from smolagents.models import MessageRole, Model
from smolagents.agents import ActionStep

from mdconvert import MarkdownConverter

AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]


def save_agent_steps(agent, kwargs, response, sample):
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            step.agent_memory = None
    intermediate_steps = str(agent.memory.steps)
    with open("steps.json", "w") as f:
        json.dump({
            "agent_args": kwargs,
            "intermediate_steps": intermediate_steps,
            "response": str(response),
            "sample": sample
            }, f, indent=2)
        
def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]

def check_budget_exceeded(agent: CodeAgent, budget: float, model_name: str) -> bool:
    total_input_tokens = agent.monitor.total_input_token_count
    total_output_tokens = agent.monitor.total_input_token_count
    
    cost = MODEL_PRICES_DICT[model_name]["prompt_tokens"] * total_input_tokens + MODEL_PRICES_DICT[model_name]["completion_tokens"] * total_output_tokens
    
    print(f"Current cost: {cost}")
    if cost >= budget:
        return True
    return False

class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".htm", ".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    def __init__(self, model: Model, text_limit: int):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def forward_initial_exam_mode(self, file_path, question):
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self.model(messages).content

    def forward(self, file_path, question: Optional[str] = None) -> str:
        result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content

        if not question:
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        return self.model(messages).content


@tool
def query_vision_language_model(query: str, image_path: str) -> str:
    """
    Query a vision language model with text and an image.
    Args:
        query: The text query or question to ask about the image
        image_path: Path to the image file to analyze
    Returns:
        str: The vision language model's response about the image
    """
    try:
        import base64
        from litellm import completion
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            base64_image = base64.b64encode(image_content).decode("utf-8")
        
        # Create the message with text and image
        response = completion(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        
        # Return the model's response
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error processing vision query: {str(e)}"

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
        output = f"Exit Code: {result.returncode}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        
        # Limit output to 1000 tokens
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(output)
        if len(tokens) > 1000:
            output = encoding.decode(tokens[:1000]) + "\n... (output truncated to 1000 tokens)"
        
        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"
    
    
    
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
            if not path.exists():
                return f"Path {path} does not exist"
            elif path.is_dir():
                return f"Path {path} is a directory"
            else:
                with open(path, 'r') as f:
                    content = f.read()
                    return content[:5000] + ('...' if len(content) > 5000 else '')

        elif command == "create":
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
def file_content_search(query: str, exclude_pattern: Optional[str] = "*.pyc,*.git*,__pycache__,*.bin,*.exe,*.dll,*.so") -> str:
    """
    Search files in the current directory and subdirectories for specific content. This will only search the content of the files, not the files themselves.
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
    
    context_lines = 3  # Reduced from 100 to keep output manageable
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
                
            # skip input.json, agent_args.json, and steps.json
            if file_path.name in ["input.json", "agent_args.json", "steps.json", "main.py"]:
                skip = True
            
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
                        # Truncate very long contexts
                        if len(context) > 1000:
                            context = context[:500] + "\n... (truncated) ...\n" + context[-500:]
                            
                        results.append(f"File: {file_path} (line {i+1}):\n{context}\n---")
                        matches_found += 1
                        
            except (UnicodeDecodeError, IOError):
                # Skip binary or unreadable files
                continue
    
        if not results:
            return f"No matches found for '{query}' in {files_searched} files."
        
        summary = f"Found {matches_found} matches for '{query}' in {files_searched} files.\n\n"
        full_output = summary + "\n".join(results)
        
        return full_output
    
    except Exception as e:
        return f"Error searching files: {str(e)}"


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    assert 'budget' in kwargs, 'budget is required'
    
    BUDGET = kwargs['budget']
    
    import litellm
    litellm.drop_params = True
    model_params = {}
    model_params['model_id'] = kwargs['model_name']
    if 'reasoning_effort' in kwargs:
        model_params['reasoning_effort'] = kwargs['reasoning_effort']
    if 'temperature' in kwargs:
        model_params['temperature'] = kwargs['temperature']
        
    if 'gemini' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('gemini/', 'openai/')
        model_params['api_key'] = os.getenv('GEMINI_API_KEY')
        model_params['api_base'] = "https://generativelanguage.googleapis.com/v1beta/openai/"
        
    if 'together_ai' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('together_ai/', 'openai/')
        model_params['api_key'] = os.environ.get("TOGETHERAI_API_KEY")
        model_params['api_base'] = "https://api.together.xyz/v1"
        
    task_id, task = list(input.items())[0]
    
    results = {}
    
    model = LiteLLMModel(**model_params)
    
    CORE_TOOLS = [
        # DuckDuckGoSearchTool(),
        GoogleSearchTool(provider='serpapi'),
        VisitWebpageTool(),
        PythonInterpreterTool(),
        execute_bash,
        TextInspectorTool(model=model, text_limit=5000),
        edit_file,
        file_content_search,
        query_vision_language_model,
    ]

    agent = CodeAgent(
    tools=CORE_TOOLS,
    planning_interval=4,
    max_steps=200,
    additional_authorized_imports=AUTHORIZED_IMPORTS,
    budget_exceeded_callback=partial(check_budget_exceeded, budget=BUDGET, model_name=kwargs['model_name']),
    model=model)
    agent.python_executor.state["__name__"] = "__main__"

    if kwargs['benchmark_name'] == 'usaco':
        USACO_PROMPT = """Please reply with a Python 3 solution to the below problem. Make sure
to wrap your code in '```python' and '```' Markdown delimiters, and
include exactly one block of code with the entire solution.
No outside libraries are allowed.

[BEGIN PROBLEM]
{}
[END PROBLEM]
"""
        prompt = USACO_PROMPT.format(task['description'])
        response = agent.run(prompt)
        save_agent_steps(agent, kwargs, response, task)
        
        # extract code from response
        response = str(response)
        if '```python' in response:
            response = response.split('```python')[1].split('```')[0]
            
        return {task_id: response}
            
    elif kwargs['benchmark_name'] == 'corebench_easy':
        
        response = agent.run(task['prompt'])
        save_agent_steps(agent, kwargs, response, task)
        return {task_id: response}

    elif kwargs['benchmark_name'] == 'corebench_medium':
        response = agent.run(task['prompt'])
        save_agent_steps(agent, kwargs, response, task)
        return {task_id: response}
    
    elif kwargs['benchmark_name'] == 'corebench_hard':
        response = agent.run(task['prompt'])
        save_agent_steps(agent, kwargs, response, task)
        return {task_id: response}
    
    elif kwargs['benchmark_name'] == 'scienceagentbench':
        
        DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""
        
        def format_task_dict(example):
            task = {
                "instance_id": example["instance_id"],
                "task_inst": example["task_inst"],
                "dataset_path": "benchmark/datasets/" + example["dataset_folder_tree"].split("\n")[0][4:],
                "dataset_folder_tree": example["dataset_folder_tree"],
                "dataset_preview": example["dataset_preview"],
                "output_fname": example["output_fname"],
                "domain_knowledge": example["domain_knowledge"],
                "gold_program_name": example["gold_program_name"],
            }

            return task
        
        def get_sys_msg(task):

            sys_msg = (
                """Please reply with a Python 3 solution to the below problem. Make sure
to wrap your code in '```python' and '```' Markdown delimiters, and
include exactly one block of code with the entire solution.""",
                "\n" + task["task_inst"] + 
                ("\n" + str(task["domain_knowledge"]))
            )

            sys_msg += (
                "\n" +
                DATA_INFO_PROMPT.format(
                    dataset_path = task['dataset_path'],
                    dataset_folder_tree = task['dataset_folder_tree'],
                    dataset_preview = task["dataset_preview"]
                )
            )


            return sys_msg
        
        
        task_id = list(input.keys())[0]
        task = format_task_dict(list(input.values())[0])
        sys_msg = get_sys_msg(task)
        
        response = str(agent.run(sys_msg))
        save_agent_steps(agent, kwargs, response, task)
        
        return {task_id: {"history": [{"role": "assistant", "content": f"```python{response}```"}]}}
        
        
        
    elif kwargs['benchmark_name'] == 'swebench_verified':
        pass
    elif kwargs['benchmark_name'] == 'swebench_verified_mini':
        process = subprocess.run(['git', 'clone', f'https://github.com/{task["repo"]}.git'], capture_output=True, text=True)
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to clone repository: {process.stderr}")
        
        process = subprocess.run(
            f"cd {task['repo'].split('/')[-1]} && git reset --hard {task['base_commit']}", shell=True, capture_output=True, text=True)
        print(process.stdout)
        if process.returncode != 0:
            raise Exception(f"Failed to reset repository: {process.stderr}")
        
        response = agent.run(
            f"""I need you to solve this issue by generating a single patch that I can apply directly to this repository using git apply.
            
Problem: {task['problem_statement']}
            
The code of the project is cloned to {task['repo'].split('/')[-1]}. After you are done, please return the content of the patch as your final answer."""
        )
        save_agent_steps(agent, kwargs, response, task)
        
        model_patch = extract_diff(response)
        return {task_id: model_patch}
        
    elif kwargs['benchmark_name'] == 'appworld_test_normal':
        from appworld.task import Task
        
        def tool_generator(name_: str, description_: str, inputs_: dict, output_type_: dict, function: callable):
            class GeneratedTool(Tool):
                name = name_
                description = description_
                inputs = inputs_
                output_type = output_type_

                def forward(self, *args, **kwargs):
                    return world.apis.function(*args, **kwargs)

            GeneratedTool.__name__ = f"{name_.title()}Tool"
            return GeneratedTool

        def get_smolagents_tools(task):
            tools: list[Tool] = []
            for api in task.api_docs.keys():
                for api_func in task.api_docs[api].keys():
                    tool = tool_generator(
                    name_=api_func,
                    description_=task.api_docs[api][api_func]["description"],
                    inputs_=task.api_docs[api][api_func]["parameters"],
                    output_type_=task.api_docs[api][api_func]["response_schemas"]["success"],
                    function=api_func
                    )
                    tools.append(tool)
            return tools
        
        from appworld import AppWorld
    
        with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8001") as world:
            instruction = world.task.instruction # To see task instruction.
            supervisor = world.task.supervisor
            tools = get_smolagents_tools(world.task)
            
            prompt = f"""Using the available APIs you can interact with on my behalf through the "interact_with_apis" tool, generate code to solve the following task.
            
Here are three key APIs that you need to know to get more information
            
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Now please generate code to solve the following task:

My name is: {supervisor.first_name} {supervisor.last_name}. My personal email is {supervisor.email} and phone number is {supervisor.phone_number}.

Task:

{instruction}
"""
            
            @tool
            def execute_code_to_interact_with_apis(code: str) -> str:
                """
                Execute code to interact with the APIs. You can access variables from previous code blocks you executed. Code is executed in a python REPL environment.
                Args:
                    code: The code to execute
                Returns:
                    str: The terminal output of the code
                    str: Whether the task is completed
                """
                return world.execute(code), "Task completed" if world.task_completed() else "Task not yet completed"
            
            agent = CodeAgent(
                tools=CORE_TOOLS + get_smolagents_tools(world.task),
                planning_interval=4,
                max_steps=200,
                additional_authorized_imports=AUTHORIZED_IMPORTS,
                budget_exceeded_callback=partial(check_budget_exceeded, budget=BUDGET, model_name=kwargs['model_name']),
                model=model)
            agent.python_executor.state["__name__"] = "__main__"
            
            response = agent.run(prompt)
            world.post_execute()
            world.save()
            save_agent_steps(agent, kwargs, response, task)
            
        return {task_id: "Completed"}
    
    
    elif kwargs['benchmark_name'] == 'appworld_test_challenge':
        from appworld import AppWorld
        
        @tool
        def execute_code_to_interact_with_apis(code: str) -> str:
            """
            Execute code to interact with the APIs. You can access variables from previous code blocks you executed. Code is executed in a python REPL environment.
            Args:
                code: The code to execute
            Returns:
                str: The terminal output of the code
                str: Whether the task is completed
            """
            return world.execute(code), "Task completed" if world.task_completed() else "Task not yet completed"
        
        agent = CodeAgent(
            tools=CORE_TOOLS + [execute_code_to_interact_with_apis],
            planning_interval=4,
            max_steps=200,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            budget_exceeded_callback=partial(check_budget_exceeded, budget=BUDGET, model_name=kwargs['model_name']),
            model=model)
        agent.python_executor.state["__name__"] = "__main__"
        
        with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8001") as world:
            instruction = world.task.instruction # To see task instruction.
            supervisor = world.task.supervisor
            
            prompt = f"""Using the available APIs you can interact with on my behalf through the "interact_with_apis" tool, generate code to solve the following task.
            
Here are three key APIs that you need to know to get more information
            
# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Now please generate code to solve the following task:

My name is: {supervisor.first_name} {supervisor.last_name}. My personal email is {supervisor.email} and phone number is {supervisor.phone_number}.

Task:

{instruction}
"""
            
            response = agent.run(prompt)
            world.post_execute()
            world.save()
            save_agent_steps(agent, kwargs, response, task)
        
        return {task_id: "Completed"}
            
            
    elif kwargs['benchmark_name'] == 'gaia':
        prompt = f"""Please answer the question below. You should:                                                                                                                   
                                                                                                                                                                 
- Return only your answer, which should be a number, or a short phrase with as few words as possible, or a comma separated list of numbers and/or strings.      
- If the answer is a number, return only the number without any units unless specified otherwise.                                                               
- If the answer is a string, don't include articles, and don't use abbreviations (e.g. for states).                                                             
- If the answer is a comma separated list, apply the above rules to each element in the list.                                                                                                                                                                                                                    
                                                                                                                                                                 
Here is the question and attached files are stored in your current directory:

{task['Question']}"""
        response = agent.run(prompt)
        save_agent_steps(agent, kwargs, response, task)
        return {task_id: response}
    
    
    elif kwargs['benchmark_name'] == 'taubench_airline':
        from tau_bench.envs import get_env
        from tau_bench.types import Action
        
        ### ENV SETUP (usually this should be untouched) ###
        isolated_env = get_env(
            input[task_id]['env'],
            input[task_id]['user_strategy'],
            input[task_id]['user_model'],
            input[task_id]['task_split'],
            input[task_id]['user_provider'],
            input[task_id]['task_index']
        )
        
        ## taubench airline tools
        @tool
        def book_reservation(
            user_id: str,
            origin: str,
            destination: str,
            flight_type: str,
            cabin: str,
            flights: List[Dict[str, str]],
            passengers: List[Dict[str, str]],
            payment_methods: List[Dict[str, str]],
            total_baggages: int,
            nonfree_baggages: int,
            insurance: str,
        ) -> str:
            """
            Book a reservation.

            Args:
                user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
                origin: The IATA code for the origin city, such as 'SFO'.
                destination: The IATA code for the destination city, such as 'JFK'.
                flight_type: Type of the trip ('one_way' or 'round_trip').
                cabin: Cabin class for the reservation ('basic_economy', 'economy', 'business').
                flights: An array of objects containing details about each piece of flight.
                        Each flight should have 'flight_number' (such as 'HAT001') and 
                        'date' (in the format 'YYYY-MM-DD', such as '2024-05-01').
                passengers: An array of objects containing details about each passenger.
                        Each passenger should have 'first_name' (such as 'Noah'),
                        'last_name' (such as 'Brown'), and 'dob' (date of birth in the 
                        format 'YYYY-MM-DD', such as '1990-01-01').
                payment_methods: An array of objects containing details about each payment method.
                                Each payment method should have 'payment_id' (such as 'credit_card_7815826',
                                'gift_card_7815826', 'certificate_7815826') and 'amount' (the amount to be paid).
                total_baggages: The total number of baggage items included in the reservation.
                nonfree_baggages: The number of non-free baggage items included in the reservation.
                insurance: Indicates whether travel insurance is added ('yes' or 'no').

            Returns:
                str: A JSON string of the reservation details if booking is successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            
            action = Action(
                name='book_reservation',
                kwargs={
                    'user_id': user_id,
                    'origin': origin,
                    'destination': destination,
                    'flight_type': flight_type,
                    'cabin': cabin,
                    'flights': flights,
                    'passengers': passengers,
                    'payment_methods': payment_methods,
                    'total_baggages': total_baggages,
                    'nonfree_baggages': nonfree_baggages,
                    'insurance': insurance
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"
        
        @tool
        def calculate(expression: str) -> str:
            """
            Calculate the result of a mathematical expression.
            
            Args:
                expression: The mathematical expression to calculate, such as '2 + 2'. 
                        The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.
            
            Returns:
                str: The result of the calculation or an error message if the calculation fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='calculate',
                kwargs={
                    'expression': expression
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def cancel_reservation(reservation_id: str) -> str:
            """
            Cancel the whole reservation.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
            
            Returns:
                str: Confirmation message if cancellation is successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='cancel_reservation',
                kwargs={
                    'reservation_id': reservation_id
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def get_reservation_details(reservation_id: str) -> str:
            """
            Get the details of a reservation.
            
            Args:
                reservation_id: The reservation id, such as '8JX2WO'.
            
            Returns:
                str: A JSON string of the reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='get_reservation_details',
                kwargs={
                    'reservation_id': reservation_id
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def get_user_details(user_id: str) -> str:
            """
            Get the details of an user, including their reservations.
            
            Args:
                user_id: The user id, such as 'sara_doe_496'.
            
            Returns:
                str: A JSON string of the user details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='get_user_details',
                kwargs={
                    'user_id': user_id
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def list_all_airports() -> str:
            """
            List all airports and their cities.
            
            Returns:
                str: A JSON string containing all airports and their cities if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='list_all_airports',
                kwargs={}
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def search_direct_flight(origin: str, destination: str, date: str) -> str:
            """
            Search direct flights between two cities on a specific date.
            
            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.
            
            Returns:
                str: A JSON string of available direct flights if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='search_direct_flight',
                kwargs={
                    'origin': origin,
                    'destination': destination,
                    'date': date
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def search_onestop_flight(origin: str, destination: str, date: str) -> str:
            """
            Search direct flights between two cities on a specific date.
            
            Args:
                origin: The origin city airport in three letters, such as 'JFK'.
                destination: The destination city airport in three letters, such as 'LAX'.
                date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.
            
            Returns:
                str: A JSON string of available one-stop flights if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='search_onestop_flight',
                kwargs={
                    'origin': origin,
                    'destination': destination,
                    'date': date
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def send_certificate(user_id: str, amount: float) -> str:
            """
            Send a certificate to a user. Be careful!
            
            Args:
                user_id: The ID of the user to book the reservation, such as 'sara_doe_496'.
                amount: Certificate amount to send.
            
            Returns:
                str: Confirmation message if the certificate is sent successfully, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='send_certificate',
                kwargs={
                    'user_id': user_id,
                    'amount': amount
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def think(thought: str) -> str:
            """
            Use the tool to think about something. It will not obtain new information or change the database, 
            but just append the thought to the log. Use it when complex reasoning is needed.
            
            Args:
                thought: A thought to think about.
            
            Returns:
                str: Confirmation that the thought was logged.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='think',
                kwargs={
                    'thought': thought
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def transfer_to_human_agents(summary: str) -> str:
            """
            Transfer the user to a human agent, with a summary of the user's issue. 
            Only transfer if the user explicitly asks for a human agent, or if the user's issue 
            cannot be resolved by the agent with the available tools.
            
            Args:
                summary: A summary of the user's issue.
            
            Returns:
                str: Confirmation that the user was transferred to a human agent.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='transfer_to_human_agents',
                kwargs={
                    'summary': summary
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def update_reservation_baggages(reservation_id: str, total_baggages: int, nonfree_baggages: int, payment_id: str) -> str:
            """
            Update the baggage information of a reservation.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                total_baggages: The updated total number of baggage items included in the reservation.
                nonfree_baggages: The updated number of non-free baggage items included in the reservation.
                payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.
            
            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='update_reservation_baggages',
                kwargs={
                    'reservation_id': reservation_id,
                    'total_baggages': total_baggages,
                    'nonfree_baggages': nonfree_baggages,
                    'payment_id': payment_id
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def update_reservation_flights(reservation_id: str, cabin: str, flights: List[Dict[str, str]], payment_id: str) -> str:
            """
            Update the flight information of a reservation.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                cabin: Cabin class for the reservation ('basic_economy', 'economy', 'business').
                flights: An array of objects containing details about each piece of flight in the ENTIRE new reservation. 
                        Even if the a flight segment is not changed, it should still be included in the array.
                payment_id: The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.
            
            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='update_reservation_flights',
                kwargs={
                    'reservation_id': reservation_id,
                    'cabin': cabin,
                    'flights': flights,
                    'payment_id': payment_id
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"

        @tool
        def update_reservation_passengers(reservation_id: str, passengers: List[Dict[str, str]]) -> str:
            """
            Update the passenger information of a reservation.
            
            Args:
                reservation_id: The reservation ID, such as 'ZFA04Y'.
                passengers: An array of objects containing details about each passenger including 'first_name', 'last_name', and 'dob'.
            
            Returns:
                str: Updated reservation details if successful, or an error message if it fails.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='update_reservation_passengers',
                kwargs={
                    'reservation_id': reservation_id,
                    'passengers': passengers
                }
            )
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"
        
        @tool 
        def ask_user(question: str) -> str:
            """
            Ask the user a question.
            
            Args:
                question: The question to ask the user.
                
            Returns:
                str: The user's response.
                str: Indication of whether the user wants to end the conversation.
            """
            action = Action(
                name='respond',
                kwargs={
                    'content': question
                })
            observation = isolated_env.step(action)
            return observation.observation, "User wants to end the conversation" if observation.done else "User does not want to end the conversation"
            
        
        # get instruction from environment
        user_question = isolated_env.reset(input[task_id]['task_index']).observation    
        wiki = isolated_env.wiki
        with open('wiki.md', 'w') as f:
            f.write(wiki)
        agent = CodeAgent(
        tools=CORE_TOOLS + [
            book_reservation,
            calculate,
            cancel_reservation,
            get_reservation_details,
            get_user_details,
            list_all_airports,
            search_direct_flight,
            search_onestop_flight,
            send_certificate,
            think,
            transfer_to_human_agents,
            update_reservation_baggages,
            update_reservation_flights,
            update_reservation_passengers,
            ask_user
        ],
        planning_interval=4,
        budget_exceeded_callback=partial(check_budget_exceeded, budget=BUDGET, model_name=kwargs['model_name']),
        max_steps=200,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        model=model)
        
        agent.python_executor.state["__name__"] = "__main__"
        ### YOUR AGENT CODE HERE ###
        instruction = f"""I added some useful information to the wiki in `wiki.md`. Please read it and then answer the user's question.

User's question: {user_question}
        """
        response = str(agent.run(instruction))
        action = Action(
                name='respond',
                kwargs={
                    'content': response
                })
        observation = isolated_env.step(action)
        print("Final user's response: ", observation)
        
        save_agent_steps(agent, kwargs, response, task)
            
        ### WHEN DONE WE RETURN THE ENV STATE ###
        return {task_id: {"reward": isolated_env.reward, "taken_actions": [action.model_dump() for action in isolated_env.actions], "task": isolated_env.task.model_dump()}}
    
    else:
        raise ValueError(f"Unknown benchmark. HAL agent does not support this benchmark: {kwargs['benchmark_name']}")
    
    results[task_id] = response
        
    return results



# INSPECT BENCHMARKS BELOW

import asyncio

async def run_inspect(sample: dict[str, Any], **kwargs) -> dict[str, Any]:
    from inspect_ai.util import sandbox
    
    model = LiteLLMModel(model_id='openai/gpt-4o-mini-2024-07-18')
        
    @tool
    async def execute_bash(command: str) -> str:
        """
        Description: Execute a bash command and return its output.
        Will not execute commands requiring internet access.
        Common linux and python packages are available via apt and pip.
        Args:
            command: The bash command to execute
        """
        
        try:
            result = await sandbox().exec(command.split(' '))
            if result.success:
                # Limit output to 1000 tokens
                output = result.stdout
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(output)
                if len(tokens) > 1000:
                    output = encoding.decode(tokens[:1000]) + "\n... (output truncated to 1000 tokens)"
                return output
            else:
                return result.stderr
        except Exception as e:
            return "Execution failed: " + str(e)
        
    CORE_TOOLS_INSPECT = [
        # DuckDuckGoSearchTool(),
        GoogleSearchTool(provider='serpapi'),
        VisitWebpageTool(),
        PythonInterpreterTool(),
        TextInspectorTool(model=model, text_limit=5000),
        execute_bash,
        file_content_search,
        query_vision_language_model
    ]
        
    agent = CodeAgent(
    tools=CORE_TOOLS_INSPECT,
    planning_interval=4,
    max_steps=200,
    additional_authorized_imports=AUTHORIZED_IMPORTS,
    budget_exceeded_callback=partial(check_budget_exceeded, budget=BUDGET, model_name=kwargs['model_name']),
    model=model)
    agent.python_executor.state["__name__"] = "__main__"
    
        
    if kwargs['benchmark_name'] == 'inspect_evals/gaia':
        response = await agent.arun(sample["input"][0]["content"])
        
    elif kwargs['benchmark_name'] == 'inspect_evals/cybench':
        response = await agent.arun(sample["input"][0]["content"])
    else:
        raise ValueError(f"Unknown benchmark. HAL agent does not support this benchmark: {kwargs['benchmark_name']}")

    try:
        save_agent_steps(agent, kwargs, response, sample)
        return {"output": str(response)}
    except Exception as e:
        return  {"output": str(e)}


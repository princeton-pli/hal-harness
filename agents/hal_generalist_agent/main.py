# Disclaimer: this is not a functional agent and is only for demonstration purposes. This implementation is just a single model call.
from typing import Optional, List, Dict, Any

import requests

# from smolagents.agents import ToolCallingAgent
from smolagents import ToolCallingAgent, tool, LiteLLMModel, DuckDuckGoSearchTool, CodeAgent, Tool
import subprocess
from pathlib import Path
import re

from prompts import USACO_PROMPT, PATCH_EXAMPLE
from tau_bench.envs.airline.tools import ALL_TOOLS
from tau_bench.envs import get_env
from tau_bench.types import Action

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
                        if len(context) > 500:
                            context = context[:250] + "\n... (truncated) ...\n" + context[-250:]
                            
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
        from appworld import AppWorld, load_task_ids
    
        with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8000") as world:
            instruction = world.task.instruction # To see task instruction.
            
            @tool
            def execute_in_world(command: str) -> str:
                """
                Execute a command in the appworld environment.
                Args:
                    command: The command to execute
                Returns:
                    str: The response of the environment
                """
                return world.execute(command)
            
            agent = ToolCallingAgent(
                tools=[
                    execute_in_world,
                    DuckDuckGoSearchTool(),
                    search_wikipedia,
                    execute_bash,
                    edit_file,
                    file_search,
                ],
                add_base_tools=True,
                model=model)
            
            response = agent.run(instruction)
        
        return {task_id: "Completed"}
    elif kwargs['benchmark_name'] == 'appworld_test_challenge':
        from appworld import AppWorld
        
        @tool
        def execute_in_world(command: str) -> str:
            """
            Execute a command in the appworld environment.
            """
            return world.execute(command)
        
        agent = ToolCallingAgent(
            tools=[
                execute_in_world,
                DuckDuckGoSearchTool(),
                search_wikipedia,
                execute_bash,
                edit_file,
                file_search,
            ],
            add_base_tools=True,
            model=model)
        
        with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8000") as world:
            instruction = world.task.instruction # To see task instruction.
            
            response = agent.run(instruction)
        
        return {task_id: "Completed"}
    elif kwargs['benchmark_name'] == 'taubench_airline':
        
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
            Books a flight reservation for a user with given details and updates the system accordingly.

            Args:
                user_id: The ID of the user booking the reservation.
                origin: IATA code of the departure airport.
                destination: IATA code of the arrival airport.
                flight_type: Type of the trip ('one_way' or 'round_trip').
                cabin: Cabin class for the reservation ('basic_economy', 'economy', 'business').
                flights: A list of flight segments, each with 'flight_number' and 'date'.
                passengers: A list of passenger details including 'first_name', 'last_name', and 'dob'.
                payment_methods: Payment details including 'payment_id' and 'amount'.
                total_baggages: Total number of baggage items for the reservation.
                nonfree_baggages: Number of baggage items that are not free (and cost extra).
                insurance: Indicates whether travel insurance is added ('yes' or 'no').

            Returns:
                str: A JSON string of the reservation details if booking is successful, or an error message if it fails.
            """
            
            action = Action(
                tool_name='book_reservation',
                tool_args={
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
            return observation.observation
        
            

        # @tool
        # def calculate():
        #     pass

        # @tool
        # def cancel_reservation():
        #     pass

        # @tool
        # def get_reservation_details():
        #     pass

        # @tool
        # def get_user_details():
        #     pass

        # @tool
        # def list_all_airports():
        #     pass

        # @tool
        # def search_direct_flight():
        #     pass

        # @tool
        # def search_onestop_flight():
        #     pass

        # @tool
        # def send_certificate():
        #     pass

        # @tool
        # def think():
        #     pass

        # @tool
        # def transfer_to_human_agents():
        #     pass

        # @tool
        # def update_reservation_baggages():
        #     pass

        # @tool
        # def update_reservation_flights():
        #     pass

        # @tool
        # def update_reservation_passengers():
        #     pass
        
        # get instruction from environment
        instruction = isolated_env.reset(input[task_id]['task_index']).observation    
        
        agent = ToolCallingAgent(
        tools=[
            DuckDuckGoSearchTool(),
            search_wikipedia,
            execute_bash,
            edit_file,
            file_search,
            book_reservation
        ],
        add_base_tools=True,
        model=model)
        
        ### YOUR AGENT CODE HERE ###
        agent.run(instruction)
            
        ### WHEN DONE WE RETURN THE ENV STATE ###
        return {task_id: {"reward": isolated_env.reward, "taken_actions": [action.model_dump() for action in isolated_env.actions], "task": isolated_env.task.model_dump()}}
    
    else:
        raise ValueError(f"Unknown benchmark. HAL agent does not support this benchmark: {kwargs['benchmark_name']}")
    
    results[task_id] = response
        
    return results

from inspect_ai.util import sandbox

import asyncio

async def run_inspect(sample: dict[str, Any], **kwargs) -> dict[str, Any]:
    
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
        result = await sandbox().exec(command.split(' '))
        if result.success:
            return result.stdout
        else:
            raise result.stderr
        
        
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
        
        
    if kwargs['benchmark_name'] == 'inspect_evals/gaia':
        response = await agent.arun(sample["input"][0]["content"])
        
    elif kwargs['benchmark_name'] == 'inspect_evals/cybench':
        response = await agent.arun(sample["input"][0]["content"])
    else:
        raise ValueError(f"Unknown benchmark. HAL agent does not support this benchmark: {kwargs['benchmark_name']}")
    
    return {
        "output": str(response)
    }
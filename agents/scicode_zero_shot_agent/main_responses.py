# This is an example agent that generates code for the SciCode benchmark in a zero-shot format.
# Uses Responses API with tool calling and iteration for improved accuracy.
from openai import OpenAI
import os
from typing import Any, List, Dict
from pathlib import Path
import json

def run(input: dict[str, Any], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'

    client = OpenAI()

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
        client = OpenAI(api_key=model_params['api_key'], base_url=model_params['api_base'])
    
    if 'anthropic' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('anthropic/', 'openai/')
        model_params['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        model_params['api_base'] = "https://api.anthropic.com/v1"
        client = OpenAI(api_key=model_params['api_key'], base_url=model_params['api_base'])
        
    if 'together_ai' in kwargs['model_name']:
        model_params['model_id'] = kwargs['model_name'].replace('together_ai/', 'openai/')
        model_params['api_key'] = os.environ.get("TOGETHERAI_API_KEY")
        model_params['api_base'] = "https://api.together.xyz/v1"
        client = OpenAI(api_key=model_params['api_key'], base_url=model_params['api_base'])

    # Define available tools for code generation
    def get_available_tools() -> List[Dict[str, Any]]:
        """Returns list of tools available for the Responses API"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_code",
                    "description": "Generate Python code based on the given requirements and context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to generate"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation of the code"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "refine_code",
                    "description": "Refine and improve existing code based on feedback or requirements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The refined Python code"
                            },
                            "changes": {
                                "type": "string",
                                "description": "Description of changes made"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def call_responses_api(
        client: OpenAI,
        messages: List[Dict[str, str]],
        model_name: str,
        tools: List[Dict[str, Any]] = None,
        force_tool_usage: bool = False,
        **kwargs
    ) -> str:
        """
        Call the Responses API (or completions with tool calling) and iterate over responses.
        Leverages available tools and returns the final generated code.
        
        Args:
            force_tool_usage: If True, forces the model to use tools on the first call.
                             If False, uses "auto" to let the model decide.
        """
        # Check if Responses API is available (beta.responses), otherwise use completions with tools
        use_responses_api = hasattr(client, 'beta') and hasattr(client.beta, 'responses')
        
        # Determine tool choice based on force_tool_usage parameter
        if tools and force_tool_usage:
            # Force tool usage on first call - require generate_code tool
            tool_choice = {"type": "function", "function": {"name": "generate_code"}}
        else:
            # Let model decide whether to use tools
            tool_choice = "auto"
        
        if use_responses_api and tools:
            # Use Responses API with tool calling
            try:
                response = client.beta.responses.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                )
            except AttributeError:
                # Fallback to completions if responses API not available
                use_responses_api = False
        
        if not use_responses_api:
            # Use completions API with tool calling support
            call_kwargs = {
                "model": model_name,
                "messages": messages,
                **kwargs
            }
            if tools:
                call_kwargs["tools"] = tools
                call_kwargs["tool_choice"] = tool_choice
            
            response = client.chat.completions.create(**call_kwargs)
        
        # Iterate over responses and handle tool calls
        messages_for_iteration = messages.copy()
        max_iterations = 5  # Limit iterations to prevent infinite loops
        iteration = 0
        extracted_code = None
        
        while iteration < max_iterations:
            # Get the response message
            message = response.choices[0].message
            
            # Check if there are tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"[DEBUG] Tool calls detected! Count: {len(message.tool_calls)}")
                for tc in message.tool_calls:
                    print(f"[DEBUG] Tool call: {tc.function.name}")
                # Add assistant message with tool calls to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                }
                messages_for_iteration.append(assistant_message)
                
                # Execute tool calls and add results
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        function_args = {}
                    
                    # Handle tool execution - extract code from tool calls
                    if function_name == "generate_code" or function_name == "refine_code":
                        # Store extracted code (prefer refine_code over generate_code if both present)
                        code_from_tool = function_args.get("code", "")
                        if code_from_tool:
                            if function_name == "refine_code" or extracted_code is None:
                                extracted_code = code_from_tool
                        
                        # Add tool result to conversation for continued iteration
                        tool_result = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps({
                                "code": code_from_tool,
                                "status": "success",
                                "message": f"Code {function_name.replace('_', ' ')}d successfully"
                            })
                        }
                        messages_for_iteration.append(tool_result)
                
                # Continue conversation with tool results to allow refinement
                # On subsequent calls, allow "auto" so model can refine or finish
                call_kwargs = {
                    "model": model_name,
                    "messages": messages_for_iteration,
                    **kwargs
                }
                if tools:
                    call_kwargs["tools"] = tools
                    # Allow auto on refinement calls - model can use refine_code or finish
                    call_kwargs["tool_choice"] = "auto"
                
                response = client.chat.completions.create(**call_kwargs)
                iteration += 1
            else:
                # No more tool calls
                if iteration == 0:
                    print(f"[DEBUG] WARNING: No tool calls on first iteration! Model returned content directly.")
                # Prefer extracted code from tools, otherwise use message content
                final_content = message.content or ""
                if extracted_code:
                    print(f"[DEBUG] Returning code extracted from tools (length: {len(extracted_code)})")
                    # If we have code from tools, return it (tools are more reliable)
                    return extracted_code
                print(f"[DEBUG] Returning content from message (length: {len(final_content)})")
                return final_content
        
        # If we've exhausted iterations, prefer extracted code, otherwise return last message
        if extracted_code:
            return extracted_code
        return response.choices[0].message.content or ""

    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        """Process problem code and return the function header and return line"""
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(with_background: bool, previous_llm_code: list[str],
                              problem_data: dict, num_steps: int) -> tuple[str, str]:
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        for i in range(num_steps - 1):
            output_lines.append((problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"]) if with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            output_lines.append(previous_llm_code[i])
            output_lines.append("------")

        next_step.append((problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"]) if with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        return output_str, next_step_str
    
    def generate_prompt_with_steps(with_background: bool, previous_llm_code: list[str],
                                   prob_data: dict, num_steps: int, prompt_template: str) -> tuple[str, str]:
        """Generate prompt with steps for scicode and scicode easy benchmark"""
        # Parse the input file and extract the content
        problem_steps_str, next_step_str = process_problem_steps(with_background, previous_llm_code, prob_data,
                                                                                         num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n'
    
    def generate_prompt_without_steps(prob_data: dict, prompt_template: str):
        """Generate prompt without steps for scicode_hard benchmark"""
        last_step = len(prob_data["sub_steps"])
        output_str = prob_data["problem_description_main"] + '\n' + process_problem_code(prob_data, last_step) + '\n'
        dependencies = prob_data["required_dependencies"]

        return prompt_template.format(
            next_step_str=output_str,
            dependencies=dependencies,
        ), f'{dependencies}\n'

    # Get the benchmark name from kwargs
    benchmark_name = kwargs['benchmark_name']

    # Initialize results dictionary
    results = {}

    # Load the prompt template based on the benchmark name
    if benchmark_name == 'scicode_hard':
        prompt_template = Path("hard_prompt_template.txt").read_text()
    elif benchmark_name == 'scicode_easy':
        prompt_template = Path("easy_prompt_template.txt").read_text()
    else:
        prompt_template = Path("prompt_template.txt").read_text()

    # Get available tools for Responses API
    available_tools = get_available_tools()
    
    # For the hard benchmark, generate full prompt once for each problem
    if benchmark_name == 'scicode_hard':
        for task_id, task in input.items():
            print(f'Generating {task_id}...')

            prompt, dependencies = generate_prompt_without_steps(
                prob_data=task,
                prompt_template=prompt_template
            )
            model_call_name = kwargs['model_name']

            if 'gemini' in kwargs['model_name']:
                model_call_name = kwargs['model_name'].replace('gemini/', '')
            elif 'anthropic' in kwargs['model_name']:
                model_call_name = kwargs['model_name'].replace('anthropic/', '')
            elif 'together_ai' in kwargs['model_name']:
                model_call_name = kwargs['model_name'].replace('together_ai/', '')
            
            # Use Responses API with tool calling and iteration
            call_kwargs = {}
            if 'reasoning_effort' in kwargs:
                call_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
            if 'temperature' in kwargs:
                call_kwargs['temperature'] = kwargs['temperature']
            
            # Use force_tool_usage=False by default to avoid unnecessary cost/time
            final_response = call_responses_api(
                client=client,
                messages=[{"role": "user", "content": prompt}],
                model_name=model_call_name,
                tools=available_tools,
                force_tool_usage=False,  # Set to True to force tool usage
                **call_kwargs
            )

            generated_code = final_response.replace("```python", "").replace("```", "").strip()

            results[task_id] = generated_code

    # For the easy and standard benchmarks, generate full prompt for each subtask
    else:

        # Determine if the benchmark is easy to add background information
        easy = True if benchmark_name == 'scicode_easy' else False

        # Iterate through problems
        for task_id, task in input.items():
            previous_llm_code = []
            full_code = ""
            steps = len(task['sub_steps'])
            print(f'Generating {task_id}...')
            steps_results = {}

            for i in range(steps):
                if (task_id == "13" and i == 5) or (task_id == "62" and i == 0) or (task_id == "76" and i == 2):
                    step_code = Path(f"{task_id}.{i + 1}.txt").read_text(encoding='utf-8')
                    previous_llm_code.append(step_code)
                    full_code += f'\n{step_code}'
                    steps_results[f'{task_id}.{i + 1}'] = full_code
                    continue

                prompt, dependencies = generate_prompt_with_steps(
                    with_background=easy,
                    previous_llm_code=previous_llm_code,
                    prob_data=task,
                    num_steps=i + 1,
                    prompt_template=prompt_template
                )

                model_call_name = kwargs['model_name']

                if 'openai' in kwargs['model_name']:
                    model_call_name = kwargs['model_name'].replace('openai/', '')
                elif 'gemini' in kwargs['model_name']:
                    model_call_name = kwargs['model_name'].replace('gemini/', '')
                elif 'anthropic' in kwargs['model_name']:
                    model_call_name = kwargs['model_name'].replace('anthropic/', '')
                elif 'together_ai' in kwargs['model_name']:
                    model_call_name = kwargs['model_name'].replace('together_ai/', '')
                

                # Use Responses API with tool calling and iteration
                call_kwargs = {}
                if 'reasoning_effort' in kwargs:
                    call_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                if 'temperature' in kwargs:
                    call_kwargs['temperature'] = kwargs['temperature']
                
                # Use force_tool_usage=False by default to avoid unnecessary cost/time
                final_response = call_responses_api(
                    client=client,
                    messages=[{"role": "user", "content": prompt}],
                    model_name=model_call_name,
                    tools=available_tools,
                    force_tool_usage=False,  # Set to True to force tool usage
                    **call_kwargs
                )

                # Remove the ```python and final ``` from generated_code
                generated_code = final_response.replace("```python", "").replace("```", "").strip()

                # Update previous_llm_code string with the generated code
                previous_llm_code.append(generated_code)
                full_code += f'\n{generated_code}'

                # Store the generated code for the current step
                if easy == True:
                    steps_results[f'{task_id}.{i + 1}'] = full_code
                else:
                    steps_results[f'{task_id}.{i + 1}'] = dependencies + full_code
                
            results[task_id] = steps_results
        
    return results
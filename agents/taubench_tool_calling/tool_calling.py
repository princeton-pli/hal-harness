import os

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'provider' in kwargs, 'provider is required. choose from openai or anthropic'
    task_id = list(input.keys())[0]

    import litellm
    litellm.drop_params = True

    # Determine provider configuration first (needed for litellm setup)
    if kwargs['provider'] == 'openai':
        # Use OpenAI directly
        user_provider = 'openai'
        api_base = None
        api_key = None
        model_name = kwargs['model_name']
    else:
        # Use OpenRouter for all non-OpenAI providers using OpenAI API compatibility
        user_provider = 'openai'
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY')
        model_name = kwargs['model_name']

        # Configure litellm for OpenRouter
        if api_key:
            os.environ['OPENROUTER_API_KEY'] = api_key

    # Store the original completion functions
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion

    # Create a wrapper that adds reasoning parameters and OpenRouter configuration
    def completion_with_reasoning(*args, **completion_kwargs):
        # Add OpenRouter base URL, API key, and headers for non-OpenAI providers
        if api_base:
            completion_kwargs['api_base'] = api_base
            completion_kwargs['api_key'] = api_key
            extra_headers = completion_kwargs.get('extra_headers', {})
            extra_headers['HTTP-Referer'] = 'https://github.com/benediktstroebl/hal-harness'
            extra_headers['X-Title'] = 'HAL Harness'
            completion_kwargs['extra_headers'] = extra_headers

        # Check if this is a call with our agent's model
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {
                        'low': 1024,
                        'medium': 2048,
                        'high': 4096
                    }
                    reasoning_tokens = effort_to_tokens.get(kwargs['reasoning_effort'], 4096)
                    extra_body = completion_kwargs.get('extra_body', {})
                    extra_body['reasoning'] = {"max_tokens": reasoning_tokens}
                    extra_body['include_reasoning'] = True
                    completion_kwargs['extra_body'] = extra_body
                    print(f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}")
                else:
                    # For direct OpenAI
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")

        # Call the original function
        response = original_completion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if hasattr(response, '_hidden_params') and response._hidden_params.get('response_cost') is None:
            response._hidden_params['response_cost'] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and tool_call.function:
                        if not tool_call.function.arguments or tool_call.function.arguments.strip() == "":
                            tool_call.function.arguments = "{}"

        return response

    # Create async wrapper
    async def acompletion_with_reasoning(*args, **completion_kwargs):
        # Add OpenRouter base URL, API key, and headers for non-OpenAI providers
        if api_base:
            completion_kwargs['api_base'] = api_base
            completion_kwargs['api_key'] = api_key
            extra_headers = completion_kwargs.get('extra_headers', {})
            extra_headers['HTTP-Referer'] = 'https://github.com/benediktstroebl/hal-harness'
            extra_headers['X-Title'] = 'HAL Harness'
            completion_kwargs['extra_headers'] = extra_headers

        # Check if this is a call with our agent's model
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0

                if api_base:  # Using OpenRouter
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {'low': 1024, 'medium': 2048, 'high': 4096}
                    reasoning_tokens = effort_to_tokens.get(kwargs['reasoning_effort'], 4096)
                    extra_body = completion_kwargs.get('extra_body', {})
                    extra_body['reasoning'] = {"max_tokens": reasoning_tokens}
                    extra_body['include_reasoning'] = True
                    completion_kwargs['extra_body'] = extra_body
                    print(f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}")
                else:
                    # For direct OpenAI
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")

        # Call the original function
        response = await original_acompletion(*args, **completion_kwargs)

        # Ensure response_cost is set (litellm may not calculate it for custom api_base)
        if hasattr(response, '_hidden_params') and response._hidden_params.get('response_cost') is None:
            response._hidden_params['response_cost'] = 0.0

        # Fix empty tool call arguments (OpenRouter/Claude compatibility issue)
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and tool_call.function:
                        if not tool_call.function.arguments or tool_call.function.arguments.strip() == "":
                            tool_call.function.arguments = "{}"

        return response

    # Replace both sync and async completion functions
    litellm.completion = completion_with_reasoning
    litellm.acompletion = acompletion_with_reasoning

    from tau_bench.envs import get_env
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent

    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]['env'],
        input[task_id]['user_strategy'],
        input[task_id]['user_model'],
        input[task_id]['task_split'],
        user_provider,
        input[task_id]['task_index']
    )

    ### YOUR AGENT CODE HERE ###
    agent = ToolCallingAgent(
        tools_info=isolated_env.tools_info,
        wiki=isolated_env.wiki,
        model=model_name,
        provider=user_provider,
        temperature=kwargs['temperature'] if 'temperature' in kwargs else 0.0
    )

    _ = agent.solve(isolated_env, task_index=input[task_id]['task_index'])

    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {task_id: {"reward": isolated_env.reward, "taken_actions": [action.model_dump() for action in isolated_env.actions], "task": isolated_env.task.model_dump()}}

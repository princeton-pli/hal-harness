import json
import os

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'provider' in kwargs, 'provider is required. choose from openai or anthropic'
    task_id = list(input.keys())[0]
    
    import litellm
    litellm.drop_params = True
    
    # Store the original completion functions
    original_completion = litellm.completion
    original_acompletion = litellm.acompletion
    
    # Create a wrapper that adds reasoning parameters only for the agent's model
    def completion_with_reasoning(*args, **completion_kwargs):
        # Check if this is a call with our agent's model
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0
                
                if 'openrouter/' in kwargs['model_name']:
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
                    # For direct Anthropic
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")
        
        # Call the original function
        return original_completion(*args, **completion_kwargs)
    
    # Create async wrapper
    async def acompletion_with_reasoning(*args, **completion_kwargs):
        if 'model' in completion_kwargs and completion_kwargs['model'] == model_name:
            if 'reasoning_effort' in kwargs:
                # Set temperature to 1 for reasoning calls
                completion_kwargs['temperature'] = 1.0
                
                if 'openrouter/' in kwargs['model_name']:
                    # For OpenRouter, add reasoning to extra_body
                    effort_to_tokens = {'low': 1024, 'medium': 2048, 'high': 4096}
                    reasoning_tokens = effort_to_tokens.get(kwargs['reasoning_effort'], 4096)
                    extra_body = completion_kwargs.get('extra_body', {})
                    extra_body['reasoning'] = {"max_tokens": reasoning_tokens}
                    extra_body['include_reasoning'] = True
                    completion_kwargs['extra_body'] = extra_body
                    print(f"Setting reasoning tokens to {reasoning_tokens} for OpenRouter model {model_name}")
                else:
                    # For direct Anthropic
                    completion_kwargs['reasoning_effort'] = kwargs['reasoning_effort']
                    print(f"Setting reasoning_effort to {kwargs['reasoning_effort']} for model {model_name}")
        return await original_acompletion(*args, **completion_kwargs)
    
    # Replace both sync and async completion functions
    litellm.completion = completion_with_reasoning
    litellm.acompletion = acompletion_with_reasoning
        
    if 'openrouter/' in kwargs['model_name']:
        # OpenRouter support - uses OpenAI-compatible API
        # Check this FIRST before other providers since model names might contain provider names
        user_provider = 'openai'
        api_base = "https://openrouter.ai/api/v1"
        api_key = os.getenv('OPENROUTER_API_KEY')
        model_name = kwargs['model_name'].replace('openrouter/', '')  # Remove openrouter/ prefix
    elif 'together_ai' in kwargs['model_name']:
        user_provider = 'openai'
        api_base = "https://api.together.xyz/v1"
        api_key = os.getenv('TOGETHERAI_API_KEY')
        model_name = kwargs['model_name'].replace('together_ai/', '')
    elif 'gemini' in kwargs['model_name']:
        user_provider = 'openai'
        api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.getenv('GEMINI_API_KEY')
        model_name = kwargs['model_name'].replace('gemini/', '')
    elif 'claude' in kwargs['model_name']:
        user_provider = 'openai'
        api_base = "https://api.anthropic.com/v1/"
        api_key = os.getenv('ANTHROPIC_API_KEY')
        model_name = kwargs['model_name'].replace('anthropic/', '')
    else:
        user_provider = kwargs['provider']
        api_base = None
        api_key = None
        model_name = kwargs['model_name']
        
    from tau_bench.envs import get_env
    from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
    
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
    if kwargs['benchmark_name'] == 'taubench_airline':
        with open('MockAirlineDomainEnv-few_shot.jsonl', "r") as f:
                few_shot_displays = [json.loads(line)["messages_display"] for line in f]
    else:
        with open('MockRetailDomainEnv-few_shot.jsonl', "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

    agent = FewShotToolCallingAgent(
        tools_info=isolated_env.tools_info,
        wiki=isolated_env.wiki,
        few_shot_displays=few_shot_displays,
        model=model_name,
        provider=user_provider,
        temperature=kwargs['temperature'] if 'temperature' in kwargs else 0.0,
        api_base=api_base,
        api_key=api_key
    )
    
    output = agent.solve(isolated_env, task_index=input[task_id]['task_index'])
        
    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {task_id: {"reward": isolated_env.reward, "taken_actions": [action.model_dump() for action in isolated_env.actions], "task": isolated_env.task.model_dump()}}


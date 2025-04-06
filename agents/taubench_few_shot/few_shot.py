import json
from functools import partial

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'provider' in kwargs, 'provider is required. choose from openai or anthropic'
    task_id = list(input.keys())[0]
    
    import litellm
    litellm.drop_params = True
    if 'reasoning_effort' in kwargs:
        print(f"Setting reasoning_effort to {kwargs['reasoning_effort']}")
        litellm.completion = partial(litellm.completion, reasoning_effort=kwargs['reasoning_effort'])
        litellm.acompletion = partial(litellm.acompletion, reasoning_effort=kwargs['reasoning_effort'])
        
    from tau_bench.envs import get_env
    from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
    
    ### ENV SETUP (usually this should be untouched) ###
    isolated_env = get_env(
        input[task_id]['env'],
        input[task_id]['user_strategy'],
        input[task_id]['user_model'],
        input[task_id]['task_split'],
        input[task_id]['user_provider'],
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
        model=kwargs['model_name'],
        provider=kwargs['provider'],
        temperature=kwargs['temperature'] if 'temperature' in kwargs else 0.0,
    )
    
    output = agent.solve(isolated_env, task_index=input[task_id]['task_index'])
        
    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {task_id: {"reward": isolated_env.reward, "taken_actions": [action.model_dump() for action in isolated_env.actions], "task": isolated_env.task.model_dump()}}


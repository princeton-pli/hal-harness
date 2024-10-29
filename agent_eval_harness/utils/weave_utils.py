import weave
import time
from tqdm import tqdm
from .json import make_jsonable
import requests
import os
import json

MODEL_PRICES_DICT = {
                "text-embedding-3-small": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0},
                "text-embedding-3-large": {"prompt_tokens": 0.13/1e6, "completion_tokens": 0},
                "gpt-4o-2024-05-13": {"prompt_tokens": 5/1e6, "completion_tokens": 15/1e6},
                "gpt-4o-2024-08-06": {"prompt_tokens": 5/1e6, "completion_tokens": 15/1e6},
                "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-3.5-turbo": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4-turbo": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15/1e6, "completion_tokens": 1/1e6},
                "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.18/1e6},
                "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88/1e6, "completion_tokens": 0.88/1e6},
                "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5/1e6, "completion_tokens": 15/1e6},
                "Meta-Llama-3-1-70B-Instruct-htzs": {"prompt_tokens": 0.00268/1000, "completion_tokens": 0.00354/1000},
                "Meta-Llama-3-1-8B-Instruct-nwxcg": {"prompt_tokens": 0.0003/1000, "completion_tokens": 0.00061/1000},
                "gpt-4o": {"prompt_tokens": 0.005/1000, "completion_tokens": 0.015/1000},
                "Mistral-small-zgjes": {"prompt_tokens": 0.001/1000, "completion_tokens": 0.003/1000},
                "Mistral-large-ygkys": {"prompt_tokens": 0.004/1000, "completion_tokens": 0.012/1000},
                "o1-mini-2024-09-12": {"prompt_tokens": 3/1e6, "completion_tokens": 12/1e6},
                "o1-preview-2024-09-12": {"prompt_tokens": 15/1e6, "completion_tokens": 60/1e6},
                "claude-3-5-sonnet-20240620": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-3-5-sonnet-20241022": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
}

def initialize_weave_client(benchmark):
    client = weave.init(f"{benchmark}_{int(time.time())}")
    return client


# def get_total_cost(client):
#     print("Getting total cost...")
#     calls = []
#     unique_model_names = set()
#     for call in tqdm(list(client.calls())):
#         try:
#             calls.append(call.summary["usage"])
#         except KeyError as e:
#             print(f"KeyError in Weave call: {e}")
#             print(call.summary)
#         except TypeError as e:
#             print(f"TypeError in Weave call: {e}")
#             print(call.summary)

#     try: 
#         total_cost = sum(
#             MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
#             MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"]
#             for call in calls
#             for model_name in call
#         )
#         print(f"Total cost: {round(total_cost,6)}")
#     except KeyError as e:
        
#         total_cost = None
#         print("Model not found in MODEL_PRICES_DICT, total cost not calculated.")

#     unique_model_names = set(model_name for call in calls for model_name in call)

#     token_usage = {
#         model_name: {
#             "prompt_tokens": sum(call[model_name]["prompt_tokens"] for call in calls),
#             "completion_tokens": sum(call[model_name]["completion_tokens"] for call in calls)
#         }
#         for model_name in unique_model_names
#     }
    
#     if total_cost is not None:
#         print(f"Total cost: {round(total_cost,6)}")
#     return total_cost, token_usage


def get_total_cost(client):
    print("Getting total cost...")
    
    # URL and headers
    url = 'https://trace.wandb.ai/calls/stream_query'
    headers = {
        'Content-Type': 'application/json'
    }

    # Data payload
    payload = {
        "project_id": client._project_id(),
    }

    # Make the request with basic authentication
    response = requests.post(url, headers=headers, json=payload, auth=('api', os.getenv('WANDB_API_KEY')))
    calls = [json.loads(line) for line in response.text.strip().splitlines()]

    # Process usage data
    usage_calls = []
    unique_model_names = set()
    
    for call in tqdm(calls):
        try:
            if 'summary' in call and 'usage' in call['summary']:
                usage_calls.append(call['summary']['usage'])
        except KeyError as e:
            print(f"KeyError in Weave call: {e}")
            print(call['summary'])
        except TypeError as e:
            print(f"TypeError in Weave call: {e}")
            print(call['summary'])

    try:
        unique_model_names = set(model_name for call in usage_calls for model_name in call)

        # check if all unique model names are in the MODEL_PRICES_DICT
        for model_name in unique_model_names:
            if model_name not in MODEL_PRICES_DICT:
                raise KeyError(f"Model '{model_name}' not found in MODEL_PRICES_DICT.")

        total_cost = 0
        token_usage = {}
        for call in usage_calls:
            for model_name in unique_model_names:
                if model_name in call:
                    
                    # normal call
                    if 'prompt_tokens' in call[model_name] and 'completion_tokens' in call[model_name]:
                        token_usage[model_name] = {"prompt_tokens": 0, "completion_tokens": 0}
                        token_usage[model_name]["prompt_tokens"] += call[model_name]["prompt_tokens"]
                        token_usage[model_name]["completion_tokens"] += call[model_name]["completion_tokens"]

                        total_cost += (MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
                                    MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"])
                    
                    # tool use call
                    elif 'input_tokens' in call[model_name] and 'output_tokens' in call[model_name]:
                        token_usage[model_name] = {"prompt_tokens": 0, "completion_tokens": 0}
                        token_usage[model_name]["prompt_tokens"] += call[model_name]["input_tokens"]
                        token_usage[model_name]["completion_tokens"] += call[model_name]["output_tokens"]

                        total_cost += (MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["input_tokens"] +
                                    MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["output_tokens"])

                

        # total_cost = sum(
        #     MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
        #     MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"]
        #     for call in usage_calls if call is not Non
        #     for model_name in call
        # )

        # token_usage = {
        #     model_name: {
        #         "prompt_tokens": sum(call[model_name]["prompt_tokens"] for call in usage_calls if model_name in call),
        #         "completion_tokens": sum(call[model_name]["completion_tokens"] for call in usage_calls if model_name in call)
        #     }
        #     for model_name in unique_model_names
        # }
    except KeyError as e:
        print(e)
        total_cost = None
        print("Model not found in MODEL_PRICES_DICT, total cost not calculated.")    
    
    if total_cost is not None:
        print(f"Total cost: {round(total_cost, 6)}")

    else:
        print("Total cost could not be calculated.")
        token_usage = {
            model_name: {
                "prompt_tokens": None,
                "completion_tokens": None
            }
            for model_name in unique_model_names
        }
    
    return total_cost, token_usage

# def process_call_for_cost(call):
#     try:
#         return call.summary["usage"]
#     except KeyError as e:
#         print(f"KeyError in Weave call: {e}")
#         print(call.summary)
#     except TypeError as e:
#         print(f"TypeError in Weave call: {e}")
#         print(call.summary)
#     return None

# def get_total_cost(client):
#     print("Getting total cost...")
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         calls = list(executor.map(process_call_for_cost, client.calls()))

#     calls = [call for call in calls if call is not None]

#     total_cost = sum(
#         MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
#         MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"]
#         for call in calls
#         for model_name in call
#     )
#     print(f"Total cost: {round(total_cost,6)}")

#     return total_cost

def assert_task_id_logging(client, weave_task_id):
    for call in tqdm(list(client.calls())):
        if str(call.attributes['weave_task_id']) == str(weave_task_id):
            return True
    raise AssertionError("Task ID not logged or incorrect ID for test run. Please use weave.attributes to log the weave_task_id for each API call.")

def get_weave_calls(client):
    print("Getting Weave traces...")
    # URL and headers
    url = 'https://trace.wandb.ai/calls/stream_query'
    headers = {
        'Content-Type': 'application/json'
    }

    # Data payload
    payload = {
        "project_id": client._project_id(),
    }

    # Make the request with basic authentication
    response = requests.post(url, headers=headers, json=payload, auth=('api', os.getenv('WANDB_API_KEY')))
    calls = [json.loads(line) for line in response.text.strip().splitlines()]

    processed_calls = []
    for call in tqdm(calls):
        if call['output']:
            if type(call['output']) is str:
                ChatCompletion = weave.ref(call['output']).get()
                choices = [choice.message.content for choice in ChatCompletion.choices]
                created = ChatCompletion.created
            elif 'choices' in call['output']:
                choices = call['output']['choices']
                created = call['output']['created']
            elif call['output']['content']: # tooluse
                choices = call['output']['content']
                created = None

            output = {
                'weave_task_id': call['attributes']['weave_task_id'],
                'trace_id': call['trace_id'],
                'project_id': call['project_id'],
                'created_timestamp': created,
                'inputs': call['inputs'],
                'id': call['id'],
                'outputs': choices,
                'exception':  call['exception'],
                'summary': call['summary'],
                'display_name': call['display_name'],
                'attributes': call['attributes'],
            }
            processed_calls.append(output)
    print(f"Total Weave traces: {len(processed_calls)}")
    return processed_calls


# def process_call_for_weave(call):
#     ChatCompletion = weave.ref(call.output).get()
#     choices = [choice.message.content for choice in ChatCompletion.choices]
#     output = {
#             'weave_task_id': call.attributes['weave_task_id'],
#             'trace_id': call.trace_id,
#             'project_id': call.project_id,
#             'created_timestamp': ChatCompletion.created,
#             'inputs': dict(call.inputs),
#             'id': call.id,
#             'outputs': {'choices' : choices},
#             'exception': call.exception,
#             'summary': call.summary,
#             'display_name': call.display_name,
#             'attributes': dict(call.attributes),
#             "_children": call._children,
#             '_feedback': call._feedback,
#         }
#     return output

# def get_weave_calls(client):
#     print("Getting Weave traces...")
#     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#         processed_calls = list(executor.map(process_call_for_weave, client.calls()))
#     print(f"Total Weave traces: {len(processed_calls)}")

#     return processed_calls
import weave
import time
from tqdm import tqdm

MODEL_PRICES_DICT = {
                "text-embedding-3-small": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0},
                "text-embedding-3-large": {"prompt_tokens": 0.13/1e6, "completion_tokens": 0},
                "gpt-4o-2024-05-13": {"prompt_tokens": 5/1e6, "completion_tokens": 15/1e6},
                "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-3.5-turbo": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4-turbo": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15/1e6, "completion_tokens": 1/1e6},
}

def initialize_weave_client(benchmark):
    client = weave.init(f"{benchmark}_{int(time.time())}")
    return client


def get_total_cost(client):
    print("Getting total cost...")
    calls = []
    for call in tqdm(list(client.calls())):
        try:
            calls.append(call.summary["usage"])
        except KeyError as e:
            print(f"KeyError in Weave call: {e}")
            print(call.summary)
        except TypeError as e:
            print(f"TypeError in Weave call: {e}")
            print(call.summary)

    total_cost = sum(
        MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
        MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"]
        for call in calls
        for model_name in call
    )
    print(f"Total cost: {round(total_cost,6)}")
    return total_cost

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
    processed_calls = []
    for call in tqdm(list(client.calls())):
        ChatCompletion = weave.ref(call.output).get()
        choices = [choice.message.content for choice in ChatCompletion.choices]
        output = {
            'weave_task_id': call.attributes['weave_task_id'],
            'trace_id': call.trace_id,
            'project_id': call.project_id,
            'created_timestamp': ChatCompletion.created,
            'inputs': dict(call.inputs),
            'id': call.id,
            'outputs': {'choices' : choices},
            'exception': call.exception,
            'summary': call.summary,
            'display_name': call.display_name,
            'attributes': dict(call.attributes),
            "_children": call._children,
            '_feedback': call._feedback,
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
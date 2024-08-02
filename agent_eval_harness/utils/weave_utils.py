import weave
import time

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
    calls = []
    for call in client.calls():
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

    return total_cost
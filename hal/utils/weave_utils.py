import weave
import time
import requests
import os
import json
from typing import Dict, Any, Tuple, List, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .logging_utils import print_step, print_warning, console, create_progress
from datetime import datetime
from weave.trace_server.trace_server_interface import CallsFilter, CallsQueryReq

MODEL_PRICES_DICT = {
                "text-embedding-3-small": {"prompt_tokens": 0.02/1e6, "completion_tokens": 0},
                "text-embedding-3-large": {"prompt_tokens": 0.13/1e6, "completion_tokens": 0},
                "gpt-4o-2024-05-13": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "gpt-4o-2024-08-06": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-3.5-turbo": {"prompt_tokens": 0.5/1e6, "completion_tokens": 1.5/1e6},
                "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4-turbo": {"prompt_tokens": 10/1e6, "completion_tokens": 30/1e6},
                "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15/1e6, "completion_tokens": 0.6/1e6},
                "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.18/1e6},
                "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88/1e6, "completion_tokens": 0.88/1e6},
                "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5/1e6, "completion_tokens": 15/1e6},
                "Meta-Llama-3-1-70B-Instruct-htzs": {"prompt_tokens": 0.00268/1000, "completion_tokens": 0.00354/1000},
                "Meta-Llama-3-1-8B-Instruct-nwxcg": {"prompt_tokens": 0.0003/1000, "completion_tokens": 0.00061/1000},
                "gpt-4o": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "gpt-4o-2024-11-20": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "gpt-4.1-2025-04-14": {"prompt_tokens": 2/1e6, "completion_tokens": 8/1e6},
                "gpt-4.1-mini-2025-04-14": {"prompt_tokens": 0.4/1e6, "completion_tokens": 1.6/1e6},
                "gpt-4.1-nano-2025-04-14": {"prompt_tokens": 0.1/1e6, "completion_tokens": 0.4/1e6},
                "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75/1e6, "completion_tokens": 150/1e6},
                "Mistral-small-zgjes": {"prompt_tokens": 0.001/1000, "completion_tokens": 0.003/1000},
                "Mistral-large-ygkys": {"prompt_tokens": 0.004/1000, "completion_tokens": 0.012/1000},
                "o1-mini-2024-09-12": {"prompt_tokens": 3/1e6, "completion_tokens": 12/1e6},
                "o3-mini-2025-01-31": {"prompt_tokens": 1.1/1e6, "completion_tokens": 4.4/1e6},
                "o4-mini-2025-04-16": {"prompt_tokens": 1.1/1e6, "completion_tokens": 4.4/1e6},
                "openai/o4-mini-2025-04-16": {"prompt_tokens": 1.1/1e6, "completion_tokens": 4.4/1e6},
                "o3-2025-04-16": {"prompt_tokens": 10/1e6, "completion_tokens": 40/1e6},
                "o1-preview-2024-09-12": {"prompt_tokens": 15/1e6, "completion_tokens": 60/1e6},
                "o1-2024-12-17": {"prompt_tokens": 15/1e6, "completion_tokens": 60/1e6},
                "claude-3-5-sonnet-20240620": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-3-5-sonnet-20241022": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-sonnet-4-5": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-sonnet-4-5": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-sonnet-4-5-20250929": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-sonnet-4-5-20250929": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-opus-4-20250514": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "claude-opus-4": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "anthropic/claude-opus-4": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "anthropic/claude-opus-4-20250514": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "openai/gpt-4o-2024-11-20": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "openai/gpt-4o-2024-08-06": {"prompt_tokens": 2.5/1e6, "completion_tokens": 10/1e6},
                "openai/gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15/1e6, "completion_tokens": 0.6/1e6},
                "openai/gpt-4.1-2025-04-14": {"prompt_tokens": 2/1e6, "completion_tokens": 8/1e6},
                "openai/gpt-4.1-mini-2025-04-14": {"prompt_tokens": 0.4/1e6, "completion_tokens": 1.6/1e6},
                "openai/gpt-4.1-nano-2025-04-14": {"prompt_tokens": 0.1/1e6, "completion_tokens": 0.4/1e6},
                "openai/gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75/1e6, "completion_tokens": 150/1e6},
                "openai/o1-mini-2024-09-12": {"prompt_tokens": 3/1e6, "completion_tokens": 12/1e6},
                "openai/o3-mini-2025-01-31": {"prompt_tokens": 1.1/1e6, "completion_tokens": 4.4/1e6},
                "openai/o3-2025-04-16": {"prompt_tokens": 10/1e6, "completion_tokens": 40/1e6},
                "openai/o1-preview-2024-09-12": {"prompt_tokens": 15/1e6, "completion_tokens": 60/1e6},
                "openai/o1-2024-12-17": {"prompt_tokens": 15/1e6, "completion_tokens": 60/1e6},
                "anthropic/claude-3-5-sonnet-20240620": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-3-5-sonnet-20241022": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-opus-4-20250514": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "google/gemini-1.5-pro": {"prompt_tokens": 1.25/1e6, "completion_tokens": 5/1e6},
                "google/gemini-1.5-flash": {"prompt_tokens": 0.075/1e6, "completion_tokens": 0.3/1e6},
                "google/gemini-2.5-pro-preview-03-25": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "google/gemini-2.5-pro": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "openrouter/google/gemini-2.5-pro": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "gemini-2.5-pro": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "google/gemini-2.5-flash": {"prompt_tokens": 0.3/1e6, "completion_tokens": 2.5/1e6},
                "openrouter/google/gemini-2.5-flash": {"prompt_tokens": 0.3/1e6, "completion_tokens": 2.5/1e6},
                "gemini-2.5-flash": {"prompt_tokens": 0.3/1e6, "completion_tokens": 2.5/1e6},
                "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5/1e6, "completion_tokens": 3.5/1e6},
                "together/meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88/1e6, "completion_tokens": 0.88/1e6},
                "bedrock/amazon.nova-micro-v1:0": {"prompt_tokens": 0.000035/1e3, "completion_tokens": 0.00014/1e3},
                "amazon.nova-micro-v1:0" : {"prompt_tokens": 0.000035/1e3, "completion_tokens": 0.00014/1e3},
                "bedrock/amazon.nova-lite-v1:0": {"prompt_tokens": 0.00006/1e3, "completion_tokens": 0.00024/1e3},
                "amazon.nova-lite-v1:0" : {"prompt_tokens": 0.00006/1e3, "completion_tokens": 0.00024/1e3},
                "bedrock/amazon.nova-pro-v1:0": {"prompt_tokens": 0.0008/1e3, "completion_tokens": 0.0032/1e3},
                "amazon.nova-pro-v1:0" : {"prompt_tokens": 0.0008/1e3, "completion_tokens": 0.0032/1e3},
                "bedrock/us.anthropic.claude-3-opus-20240229-v1:0": {"prompt_tokens": 0.015/1e3, "completion_tokens": 0.075/1e3},
                "us.anthropic.claude-3-opus-20240229-v1:0" : {"prompt_tokens": 0.015/1e3, "completion_tokens": 0.075/1e3},
                "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 0.003/1e3, "completion_tokens": 0.015/1e3},
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0" : {"prompt_tokens": 0.003/1e3, "completion_tokens": 0.015/1e3},
                "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0": {"prompt_tokens": 0.003/1e3, "completion_tokens": 0.015/1e3},
                "us.anthropic.anthropic.claude-3-sonnet-20240229-v1:0" : {"prompt_tokens": 0.003/1e3, "completion_tokens": 0.015/1e3},
                "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.0008/1e3, "completion_tokens": 0.004/1e3},
                "us.anthropic.claude-3-5-haiku-20241022-v1:0" : {"prompt_tokens": 0.0008/1e3, "completion_tokens": 0.004/1e3}, 
                "bedrock/us.meta.llama3-3-70b-instruct-v1:0": {"prompt_tokens": 0.00072/1e3, "completion_tokens": 0.00072/1e3},
                "us.meta.llama3-3-70b-instruct-v1:0" : {"prompt_tokens": 0.00072/1e3, "completion_tokens": 0.00072/1e3}, 
                "claude-3-7-sonnet-20250219" : {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-3-7-sonnet-20250219" : {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "deepseek-ai/DeepSeek-V3": {"prompt_tokens": 1.25/1e6, "completion_tokens": 1.25/1e6},
                "deepseek-ai/DeepSeek-R1": {"prompt_tokens": 3/1e6, "completion_tokens": 7/1e6},
                "together_ai/deepseek-ai/DeepSeek-V3": {"prompt_tokens": 1.25/1e6, "completion_tokens": 1.25/1e6},
                "together_ai/deepseek-ai/DeepSeek-R1": {"prompt_tokens": 3/1e6, "completion_tokens": 7/1e6},
                "openrouter/deepseek/deepseek-chat-v3-0324": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.72/1e6},
                "deepseek/deepseek-chat-v3-0324": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.72/1e6},
                "openrouter/deepseek/deepseek-r1-0528": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.72/1e6},
                "deepseek/deepseek-r1-0528": {"prompt_tokens": 0.18/1e6, "completion_tokens": 0.72/1e6},
                "openrouter/deepseek/deepseek-chat-v3.1": {"prompt_tokens": 0.27/1e6, "completion_tokens": 1.10/1e6},
                "deepseek/deepseek-chat-v3.1": {"prompt_tokens": 0.27/1e6, "completion_tokens": 1.10/1e6},
                "gemini/gemini-2.0-flash": {"prompt_tokens": 0.1/1e6, "completion_tokens": 0.4/1e6},
                "gemini-2.0-flash": {"prompt_tokens": 0.1/1e6, "completion_tokens": 0.4/1e6},
                "gemini/gemini-2.5-pro-preview-03-25": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "gemini-2.5-pro-preview-03-25": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {"prompt_tokens": 0.27/1e6, "completion_tokens": 0.85/1e6},
                "openrouter/openai/gpt-oss-120b": {"prompt_tokens": 0.15/1e6, "completion_tokens": 0.6/1e6},
                "openai/gpt-oss-120b": {"prompt_tokens": 0.15/1e6, "completion_tokens": 0.6/1e6},
                "openrouter/anthropic/claude-opus-4": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "openrouter/anthropic/claude-opus-4-20250514": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "openrouter/anthropic/claude-opus-4.1": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "openrouter/anthropic/claude-opus-4.1-20250805": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "openrouter/anthropic/claude-sonnet-4": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-sonnet-4-20250514": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-sonnet-4-20250514": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-sonnet-4": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "openrouter/anthropic/claude-3.7-sonnet": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "openrouter/anthropic/claude-3.7-sonnet:thinking": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-3.7-sonnet": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-3.7-sonnet:thinking": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "anthropic/claude-opus-4.1": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "anthropic/claude-opus-4.1-20250805": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "anthropic/claude-haiku-4-5-20251001": {"prompt_tokens": 1/1e6, "completion_tokens": 5/1e6},
                "openrouter/anthropic/claude-haiku-4-5-20251001": {"prompt_tokens": 1/1e6, "completion_tokens": 5/1e6},
                "claude-haiku-4-5-20251001": {"prompt_tokens": 1/1e6, "completion_tokens": 5/1e6},
                "anthropic/claude-sonnet-4-5-20250929": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "openrouter/anthropic/claude-sonnet-4-5-20250929": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-sonnet-4-5-20250929": {"prompt_tokens": 3/1e6, "completion_tokens": 15/1e6},
                "claude-opus-4.1": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "claude-opus-4.1-20250805": {"prompt_tokens": 15/1e6, "completion_tokens": 75/1e6},
                "openai/gpt-5-2025-08-07": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "openai/gpt-5-mini-2025-08-07": {"prompt_tokens": 0.25/1e6, "completion_tokens": 2/1e6},
                "gpt-5-mini-2025-08-07": {"prompt_tokens": 0.25/1e6, "completion_tokens": 2/1e6},
                "gpt-5": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
                "gpt-5-2025-08-07": {"prompt_tokens": 1.25/1e6, "completion_tokens": 10/1e6},
}

CACHED_PRICE_OVERRIDES = {
    "o4-mini-2025-04-16": 0.275/1e6,
    "openai/o4-mini-2025-04-16": 0.275/1e6,
    "o3-mini-2025-01-31": 0.55/1e6,
    "openai/o3-mini-2025-01-31": 0.55/1e6,
    "claude-3-7-sonnet-20250219": 0.30/1e6,
    "anthropic/claude-3-7-sonnet-20250219": 0.30/1e6,
    "claude-opus-4-20250514": 1.50/1e6,
    "anthropic/claude-opus-4-20250514": 1.50/1e6,
    "claude-opus-4.1-20250805": 1.50/1e6,
    "anthropic/claude-opus-4.1-20250805": 1.50/1e6,
    "gpt-4.1": 0.50/1e6,
    "gpt-5-2025-08-07": 0.125/1e6,
    "o3-2025-04-16": 0.5/1e6,
    "openai/o3-2025-04-16": 0.5/1e6,
}

def _normalize_usage(cost: Dict[str, Any]) -> Tuple[int, int, int, int]:
    if "prompt_tokens" in cost or "completion_tokens" in cost:
        # OpenAI-style
        prompt_tokens = cost.get("prompt_tokens", 0)
        cached_input = cost.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        cache_creation = 0  # OpenAI doesn't report cache writes separately
        
    elif "input_tokens" in cost or "output_tokens" in cost:
        # Anthropic-style
        fresh_input = cost.get("input_tokens", 0)
        cached_input = cost.get("cache_read_input_tokens", 0)
        cache_creation = cost.get("cache_creation_input_tokens", 0)
        prompt_tokens = fresh_input + cached_input
        
    elif "inputTokens" in cost or "outputTokens" in cost:
        # Bedrock-style
        prompt_tokens = cost.get("inputTokens", 0)
        cached_input = cost.get("cacheReadInputTokens", 0)
        cache_creation = cost.get("cacheWriteInputTokens", 0)
        
    else:
        prompt_tokens = 0
        cached_input = 0
        cache_creation = 0
    
    completion = (
        cost.get("completion_tokens", 0)
        + cost.get("output_tokens", 0)
        + cost.get("outputTokens", 0)
    )
    
    return prompt_tokens, cached_input, cache_creation, completion

def fetch_weave_calls(client) -> List[Dict[str, Any]]:
    """Fetch Weave calls from the API"""
    calls = list(client.server.calls_query_stream({
        "project_id": client._project_id(),
        "filter": {"trace_roots_only": False},
        "sort_by": [{"field":"started_at","direction":"desc"}],
    }))
    
    return calls

def get_call_ids(task_id, client):
    """Get all call ids for calls given a task id"""
    calls = client.get_calls()
    task_calls = [c for c in calls if c.attributes['weave_task_id'] == task_id]
    return [c.id for c in task_calls]

def delete_calls(call_ids, client):
    """Delete calls given a list of call ids"""
    client.delete_calls(call_ids=call_ids)


def find_usage_dict_recursive(data):
    """Recursively searches for all values associated with the key 'usage' and returns them in a list."""
    found = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'usage':
                found.append(value)
            # Recursively check the value in case it contains more dictionaries/lists.
            found.extend(find_usage_dict_recursive(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(find_usage_dict_recursive(item))
    # For other data types, there's nothing to search.
    return found

# def calculate_costs(usage_calls: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Dict[str, int]]]:
#     """Calculate total costs and token usage from processed calls"""
#     unique_model_names = set(model_name for call in usage_calls for model_name in call)
    
#     print("USAGE CALLS 153", usage_calls)
#     # Validate models
#     for model_name in unique_model_names:
#         if model_name not in MODEL_PRICES_DICT:
#             raise KeyError(f"Model '{model_name}' not found in MODEL_PRICES_DICT.")
    
#     total_cost = 0
#     token_usage = {model: {"prompt_tokens": 0, "completion_tokens": 0} for model in unique_model_names}
    
#     for call in usage_calls:
        
#         for model_name in call:
#             if 'prompt_tokens' in call[model_name] and 'completion_tokens' in call[model_name]:
#                 # Standard call
#                 token_usage[model_name]["prompt_tokens"] += call[model_name]["prompt_tokens"]
#                 token_usage[model_name]["completion_tokens"] += call[model_name]["completion_tokens"]
#                 total_cost += (
#                     MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"] +
#                     MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["completion_tokens"]
#                 )
#             elif 'input_tokens' in call[model_name] and 'output_tokens' in call[model_name]:
#                 # Tool use call
#                 token_usage[model_name]["prompt_tokens"] += call[model_name]["input_tokens"]
#                 token_usage[model_name]["completion_tokens"] += call[model_name]["output_tokens"]
#                 total_cost += (
#                     MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["input_tokens"] +
#                     MODEL_PRICES_DICT[model_name]["completion_tokens"] * call[model_name]["output_tokens"]
#                 )
#             elif 'prompt_tokens' in call[model_name] and not ('completion_tokens' in call[model_name]):
#                 # embedding call
#                 token_usage[model_name]["prompt_tokens"] += call[model_name]["prompt_tokens"]
#                 total_cost += (
#                     MODEL_PRICES_DICT[model_name]["prompt_tokens"] * call[model_name]["prompt_tokens"]
#                 )
#             else:
#                 raise ValueError(f"Error in handling usage data! {call}")

#     return total_cost, token_usage


@weave.op()
def get_total_cost(client):
    total_cost = 0
    token_usage = {}
    requests = 0

    # Fetch all the calls in the project
    print_step("Getting token usage data (this can take a while)...")
    calls = list(
        client.server.calls_query_stream(
            CallsQueryReq(
                project_id=client._project_id(),
                filter=CallsFilter(trace_roots_only=False),
                columns=["summary"],
            )
        )
    )

    with create_progress() as progress:
        task = progress.add_task("Processing token usage data...", total=len(calls))
        for call in calls:
            summary = getattr(call, "summary", None) or {}
            usage = summary.get("usage")
            if not usage:
                progress.update(task, advance=1)
                continue

            if isinstance(usage, dict):
                usage_items = usage.items()
            elif isinstance(usage, list):
                usage_items = [
                    (model, model_usage)
                    for entry in usage if isinstance(entry, dict)
                    for model, model_usage in entry.items()
                ]
            else:
                print_warning(
                    f"Skipping unexpected usage payload of type {type(usage).__name__}"
                )
                progress.update(task, advance=1)
                continue

            if not usage_items:
                progress.update(task, advance=1)
                continue

            for k, cost in usage_items:   
                if k not in token_usage:
                    token_usage[k] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }
                
                requests += cost.get("requests", 0)
                prompt_tokens, cached_input, cache_creation, completion = _normalize_usage(cost)
                
                token_usage[k]["prompt_tokens"] += prompt_tokens
                token_usage[k]["completion_tokens"] += completion
                token_usage[k]["cache_creation_input_tokens"] += cache_creation
                token_usage[k]["cache_read_input_tokens"] += cached_input
            progress.update(task, advance=1)
            
    total_cost = 0
    for k, usage in token_usage.items():
        if k not in MODEL_PRICES_DICT:
            continue
        prices = MODEL_PRICES_DICT[k]
        
        # Get cached token prices from overrides or fall back to prompt token price
        cache_create_price = CACHED_PRICE_OVERRIDES.get(k, prices.get("prompt_tokens", 0))
        cache_read_price = CACHED_PRICE_OVERRIDES.get(k, prices.get("prompt_tokens", 0))
        
        # Calculate fresh input tokens when needed for cost calculation
        fresh_input_tokens = usage["prompt_tokens"] - usage["cache_read_input_tokens"]
        
        total_cost += (
            fresh_input_tokens * prices.get("prompt_tokens", 0)
            + usage["cache_creation_input_tokens"] * cache_create_price
            + usage["cache_read_input_tokens"] * cache_read_price
            + usage["completion_tokens"] * prices.get("completion_tokens", 0)
        )
    return total_cost, token_usage

            
def compute_cost_from_inspect_usage(
    usage: Dict[str, Dict[str, int]], skip_models: Optional[List[str]] = None
) -> float:
    """Compute cost from token usage"""
    skip_models = skip_models or []

    return sum(
        MODEL_PRICES_DICT[model_name]["prompt_tokens"] * usage[model_name]["input_tokens"]
        + MODEL_PRICES_DICT[model_name]["prompt_tokens"] * usage[model_name].get("input_tokens_cache_write", 0)
        + MODEL_PRICES_DICT[model_name]["prompt_tokens"] * usage[model_name].get("input_tokens_cache_read", 0)
        + MODEL_PRICES_DICT[model_name]["completion_tokens"] * usage[model_name]["output_tokens"]
        for model_name in usage
        if model_name not in skip_models
    )

def process_weave_output(call: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single Weave call output"""
    # convert started_at from datetime to string
    try:
        started_at = call.started_at.isoformat()
    except Exception as e:
        print("Exception processing trace of call:", call)
        started_at = None
    try:
        ended_at = call.ended_at.isoformat()
    except Exception as e:
        print("Exception processing trace of call:", call)
        ended_at = None
    
    json_call = call.dict()
    json_call['started_at'] = started_at
    json_call['ended_at'] = ended_at
    json_call['weave_task_id'] = call.attributes['weave_task_id']
    json_call['created_timestamp'] = started_at
    
    return json_call

def get_weave_calls(client) -> Tuple[List[Dict[str, Any]], str, str]:
    """Get processed Weave calls with progress tracking"""
    print_step("Getting Weave traces (this can take a while)...")
    
    # dict to store latency for each task
    latency_dict = {}
    
    with create_progress() as progress:
        # Fetch calls
        task1 = progress.add_task("Fetching Weave calls...", total=1)
        calls = fetch_weave_calls(client)
        progress.update(task1, completed=1)
        
        # Processed calls
        processed_calls = []
        
        for call in calls:
            task_id = call.attributes['weave_task_id']
            processed_call = process_weave_output(call)
            if processed_call:
                processed_calls.append(processed_call)
                
                if task_id not in latency_dict:
                    latency_dict[task_id] = {'first_call_timestamp': processed_call['started_at'], 'last_call_timestamp': processed_call['started_at']}
                else:
                    if processed_call['started_at'] < latency_dict[task_id]['first_call_timestamp']:
                        latency_dict[task_id]['first_call_timestamp'] = processed_call['started_at']
                    if processed_call['started_at'] > latency_dict[task_id]['last_call_timestamp']:
                        latency_dict[task_id]['last_call_timestamp'] = processed_call['started_at']
                    
            progress.update(task1, advance=1)
            
    for task_id in latency_dict:
        latency_dict[task_id]['total_time'] = (datetime.fromisoformat(latency_dict[task_id]['last_call_timestamp']) - datetime.fromisoformat(latency_dict[task_id]['first_call_timestamp'])).total_seconds()
    
    console.print(f"[green]Total Weave traces: {len(processed_calls)}[/]")
    return processed_calls, latency_dict

# def get_total_cost(client) -> Tuple[Optional[float], Dict[str, Dict[str, int]]]:
#     """Get total cost and token usage for all Weave calls"""
#     print_step("Calculating total cost...")
    
#     with create_progress() as progress:
#         # Fetch calls
#         task1 = progress.add_task("Fetching Weave calls...", total=1)
#         calls = fetch_weave_calls(client)
#         progress.update(task1, completed=1)
        
#         try:
#             # Process calls and calculate costs
#             total_cost, token_usage = process_usage_data(calls, progress)
#             console.print(f"[green]Total cost: ${total_cost:.6f}[/]")
#             return total_cost, token_usage
            
#         except KeyError as e:
#             print_warning(f"Error calculating costs: {str(e)}")
#             return None, {
#                 model_name: {"prompt_tokens": None, "completion_tokens": None}
#                 for model_name in set(model_name for call in calls for model_name in call.dict().get('summary', {}).get('usage', {}))
#             }

# def assert_task_id_logging(client, weave_task_id: str) -> bool:
#     """Assert that task ID is properly logged in Weave calls"""
#     with create_progress() as progress:
#         task = progress.add_task("Checking task ID logging...", total=1)
#         calls = fetch_weave_calls(client)
        
#         for call in calls:
#             if str(call['attributes'].get('weave_task_id')) == str(weave_task_id):
#                 progress.update(task, completed=1)
#                 return True
                
#         progress.update(task, completed=1)
#         raise AssertionError(
#             "Task ID not logged or incorrect ID for test run. "
#             "Please use weave.attributes to log the weave_task_id for each API call."
#         )


# def process_usage_data(calls: List[Dict[str, Any]], progress: Progress) -> Tuple[float, Dict[str, Dict[str, int]]]:
#     """Process usage data from Weave calls"""
#     usage_calls = []
#     unique_model_names = set()
    
#     task = progress.add_task("Processing usage data...", total=len(calls))
    
#     for call in calls:
#         try:
#             # find the usage in call
#             call_dump = call.dict()

#             usage_dicts = find_usage_dict_recursive(call_dump)
#             for usage_dict in usage_dicts:
#                 usage_calls.append(usage_dict)
            
#         except (KeyError, TypeError) as e:
#             print_warning(f"Error processing call: {str(e)}")
#         progress.update(task, advance=1)
    
#     return calculate_costs(usage_calls)


# @weave.op()
# def get_costs_for_project(client):
#     total_cost = 0
#     requests = 0

#     # Fetch all the calls in the project
#     calls = list(
#         client.get_calls(filter={"trace_roots_only": True}, include_costs=True)
#     )

#     for call in calls:
#         # If the call has costs, we add them to the total cost
#         if call.summary["weave"] is not None and call.summary["weave"].get("costs", None) is not None:
#             print(call.summary["weave"])
#             for k, cost in call.summary["weave"]["costs"].items():
#                 requests += cost["requests"]
#                 total_cost += cost["prompt_tokens_total_cost"]
#                 total_cost += cost["completion_tokens_total_cost"]

#     # We return the total cost, requests, and calls
#     return total_cost

def get_task_cost(run_id: str, task_id: str) -> dict:
    """
    Calculate the cost for a specific task ID by filtering calls with that task_id.
    
    Args:
        run_id: The ID of the run to calculate costs for
        task_id: The ID of the task to calculate costs for
        
    Returns:
        dict: A dictionary containing:
            - total_cost: The total cost in dollars
            - token_usage: Token usage breakdown by model
            - requests: Total number of API requests
            - num_calls: Number of calls related to this task
    """
    total_cost = 0
    token_usage = {}
    requests = 0
    
    client = weave.init(run_id)

    print_step(f"Getting token usage data for task ID: {task_id}...")
    
    # Fetch all calls and filter by task_id
    calls = list(
        client.server.calls_query_stream(
            CallsQueryReq(
                project_id=client._project_id(),
                filter=CallsFilter(trace_roots_only=False),
                columns=["summary", "attributes"],
            )
        )
    )
    task_calls = [call for call in calls if (getattr(call, "attributes", {}) or {}).get('weave_task_id') == task_id]
    
    for call in task_calls:
        # If the call has usage data, add it to the token usage
        summary = getattr(call, "summary", None) or {}
        usage = summary.get("usage")
        if not usage:
            continue

        if isinstance(usage, dict):
            usage_items = usage.items()
        elif isinstance(usage, list):
            usage_items = [
                (model, model_usage)
                for entry in usage if isinstance(entry, dict)
                for model, model_usage in entry.items()
            ]
        else:
            print_warning(
                f"Skipping unexpected usage payload of type {type(usage).__name__}"
            )
            continue

        if not usage_items:
            continue

        for k, cost in usage_items:   
            if k not in token_usage:
                token_usage[k] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            
            requests += cost.get("requests", 0)
            prompt_tokens, cached_input, cache_creation, completion = _normalize_usage(cost)
            
            token_usage[k]["prompt_tokens"] += prompt_tokens
            token_usage[k]["completion_tokens"] += completion
            token_usage[k]["cache_creation_input_tokens"] += cache_creation
            token_usage[k]["cache_read_input_tokens"] += cached_input
    
    # Calculate total cost from token usage
    for k, usage in token_usage.items():
        if k not in MODEL_PRICES_DICT:
            print_warning(f"Model '{k}' not found in MODEL_PRICES_DICT. Skipping cost calculation.")
            continue
        prices = MODEL_PRICES_DICT[k]
        
        # Get cached token prices from overrides or fall back to prompt token price
        cache_create_price = CACHED_PRICE_OVERRIDES.get(k, prices.get("prompt_tokens", 0))
        cache_read_price = CACHED_PRICE_OVERRIDES.get(k, prices.get("prompt_tokens", 0))
        
        # Calculate fresh input tokens when needed for cost calculation
        fresh_input_tokens = usage["prompt_tokens"] - usage["cache_read_input_tokens"]
        
        model_cost = (
            fresh_input_tokens * prices.get("prompt_tokens", 0)
            + usage["cache_creation_input_tokens"] * cache_create_price
            + usage["cache_read_input_tokens"] * cache_read_price
            + usage["completion_tokens"] * prices.get("completion_tokens", 0)
        )
        total_cost += model_cost
    print_step(f"Cost for task ID: {task_id} is ${total_cost} for {len(task_calls)} calls.")
    return {
        "total_cost": total_cost,
        "token_usage": token_usage,
        "requests": requests,
        "num_calls": len(task_calls)
    }

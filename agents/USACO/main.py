"""
Contains all code to duplicate experiments in "Can language models solve olympiad programming questions?"
To utilize open models, create your own callable model function in models.py, and import it as with GPTs/Claude.
"""

from functools import partial
from models import gpts, claude, get_token_usage, reset_token_usage
from USACOBench.prompts import RetrievalType
from dotenv import load_dotenv
from utils import run_solve, run_retrieval, run_reflexion, calculate_final_rs
import time
from typing import Dict, Any, List

load_dotenv()


def collect_task_metrics(
    problem_id: str,
    stages: List[str],
    start_time: float,
    retrieval_data: Dict = None,
    reflexion_data: List = None,
    token_usage: Dict = None,
) -> Dict[str, Any]:
    """Collect comprehensive metrics for a single task"""
    metrics = {
        "total_time_seconds": time.time() - start_time,
        "stages_completed": stages,
    }

    # Add token usage if available
    if token_usage:
        metrics.update(token_usage)

    # Add retrieval metrics
    if retrieval_data:
        metrics.update(
            {
                "retrieval_type": retrieval_data.get("type", "unknown"),
                "num_retrieved_problems": len(retrieval_data.get("problem_ids", [])),
                "retrieved_problem_ids": retrieval_data.get("problem_ids", []),
                "textbook_excerpt_length": retrieval_data.get("textbook_length", 0),
            }
        )

    # Add reflexion metrics
    if reflexion_data:
        metrics.update(
            {
                "num_reflexion_iterations": len(reflexion_data),
                "reflexion_iterations": reflexion_data,
                "final_test_cases_passed": reflexion_data[-1].get(
                    "test_cases_passed", 0
                )
                if reflexion_data
                else 0,
                "final_test_cases_total": reflexion_data[-1].get("test_cases_total", 0)
                if reflexion_data
                else 0,
                "improved_with_reflexion": (
                    len(reflexion_data) > 1
                    and reflexion_data[-1].get("test_cases_passed", 0)
                    > reflexion_data[0].get("test_cases_passed", 0)
                )
                if reflexion_data
                else False,
            }
        )

    return metrics


def run_usaco_zeroshot(
    problem_dict,
    episodic_retrieval=False,
    semantic_retrieval=False,
    reflexion=False,
    attempts=1,
    num_reflexion=2,
    num_retrieved=2,
    **kwargs,
):
    """
    Runs the USACO experiment with the given parameters.
    """
    start_time = time.time()
    reset_token_usage()  # Reset tokens for per-task tracking

    import litellm

    if "reasoning_effort" in kwargs:
        print(f"Setting reasoning_effort to {kwargs['reasoning_effort']}")
        litellm.completion = partial(
            litellm.completion,
            reasoning_effort=kwargs["reasoning_effort"],
            allowed_openai_params=["reasoning_effort"],
        )
        litellm.acompletion = partial(
            litellm.acompletion,
            reasoning_effort=kwargs["reasoning_effort"],
            allowed_openai_params=["reasoning_effort"],
        )
        kwargs["temperature"] = 1

    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get("model_name", None)
    if "gpt" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "o1" in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif "o3" in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif "claude" in model_name:
        model_fn = claude
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "deepseek" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            max_tokens=32768,
        )
    else:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )

    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    if type(results[list(results.keys())[0]]) is not str:
        return "None"

    # Collect metrics for structured output
    problem_id = list(problem_dict.keys())[0]
    token_usage = get_token_usage()
    metrics = collect_task_metrics(
        problem_id=problem_id,
        stages=["solve"],
        start_time=start_time,
        token_usage=token_usage,
    )

    # Return structured output with answer and metrics
    answer = results[problem_id]
    return {problem_id: {"answer": answer, "metrics": metrics}}


def run_usaco_episodic_semantic_retrieval(
    problem_dict,
    episodic_retrieval=True,
    semantic_retrieval=True,
    reflexion=False,
    attempts=1,
    num_reflexion=2,
    num_retrieved=2,
    **kwargs,
):
    start_time = time.time()
    reset_token_usage()  # Reset tokens for per-task tracking

    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get("model_name", None)

    if "gpt" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "o1" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=1,
            max_tokens=None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "o3" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=1,
            max_tokens=None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "claude" in model_name:
        model_fn = claude
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )
    elif "deepseek" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            max_tokens=32768,
        )
    else:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            reasoning_effort=kwargs["reasoning_effort"]
            if "reasoning_effort" in kwargs
            else None,
        )

    model_fn = partial(model_fn, model=model_name)

    # Step 1: Initial solve
    solve_start = time.time()
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    solve_time = time.time() - solve_start

    # Step 2: Retrieval
    retrieval_start = time.time()
    results, ss, retrieval_queries = run_retrieval(
        model_fn,
        model_name,
        problem_dict,
        attempts,
        ss,
        num_retrieved,
        RetrievalType.EPISODIC_SEMANTIC,
        return_retrieval_data=True,
    )
    retrieval_time = time.time() - retrieval_start

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    if type(results[list(results.keys())[0]]) is not str:
        return "None"

    # Collect metrics for structured output
    problem_id = list(problem_dict.keys())[0]
    token_usage = get_token_usage()

    # Extract retrieval data
    retrieval_data = None
    if retrieval_queries:
        query = retrieval_queries[0] if retrieval_queries else {}
        retrieval_data = {
            "type": "episodic_semantic",
            "problem_ids": query.get("retrieval_problem_ids", []),
            "textbook_length": len(query.get("retrieval_text", "")),
        }

    metrics = collect_task_metrics(
        problem_id=problem_id,
        stages=["solve", "retrieval"],
        start_time=start_time,
        retrieval_data=retrieval_data,
        token_usage=token_usage,
    )

    # Add stage timing
    metrics["solve_time_seconds"] = solve_time
    metrics["retrieval_time_seconds"] = retrieval_time

    # Return structured output with answer and metrics
    answer = results[problem_id]
    return {problem_id: {"answer": answer, "metrics": metrics}}


def run_usaco_episodic_semantic_retrieval_reflexion(
    problem_dict,
    episodic_retrieval=True,
    semantic_retrieval=True,
    reflexion=False,
    attempts=1,
    num_reflexion=2,
    num_retrieved=2,
    **kwargs,
):
    start_time = time.time()
    reset_token_usage()  # Reset tokens for per-task tracking

    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get("model_name", None)
    if "gpt" in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name)
    elif "o1" in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif "o3" in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif "claude" in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name)
    elif "deepseek" in model_name:
        model_fn = gpts
        model_fn = partial(
            model_fn,
            model=model_name,
            temperature=kwargs["temperature"] if "temperature" in kwargs else None,
            max_tokens=32768,
        )
    else:
        raise Exception(
            "Model name not one of gpt or claude. Please modify code to add model support."
        )

    model_fn = partial(model_fn, model=model_name)

    # Step 1: Initial solve
    solve_start = time.time()
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    solve_time = time.time() - solve_start

    # Step 2: Retrieval
    retrieval_start = time.time()
    rdict, sdict, rs, ss, retrieval_queries = run_retrieval(
        model_fn,
        model_name,
        problem_dict,
        attempts,
        ss,
        num_retrieved,
        RetrievalType.EPISODIC_SEMANTIC,
        reflexion=True,
        return_retrieval_data=True,
    )
    retrieval_time = time.time() - retrieval_start

    # Step 3: Reflexion iterations
    reflexion_start = time.time()
    reflexions = [rdict]
    reflexion_data = []
    query_dict = None

    # Extract initial test results
    problem_id = list(problem_dict.keys())[0]
    if problem_id in rdict and rdict[problem_id]:
        initial_result = rdict[problem_id][0]
        reflexion_data.append(
            {
                "iteration": 0,
                "test_cases_passed": initial_result.get("num_passed", 0),
                "test_cases_total": initial_result.get("num_tests", 0),
                "execution_status": "success"
                if initial_result.get("num_passed", 0)
                == initial_result.get("num_tests", 0)
                else "partial_success",
            }
        )

    for i in range(num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(
            model_fn,
            model_name,
            problem_dict,
            attempts,
            rdict,
            sdict,
            query_dict,
            i,
            return_queries=True,
            retrieval=True,
        )
        reflexions.append(rdict)

        # Extract reflexion iteration results
        if problem_id in rdict and rdict[problem_id]:
            result = rdict[problem_id][0]
            reflexion_data.append(
                {
                    "iteration": i + 1,
                    "test_cases_passed": result.get("num_passed", 0),
                    "test_cases_total": result.get("num_tests", 0),
                    "execution_status": "success"
                    if result.get("num_passed", 0) == result.get("num_tests", 0)
                    else "partial_success",
                }
            )

    reflexion_time = time.time() - reflexion_start
    rs = calculate_final_rs(reflexions, problem_dict)

    # Get final result
    final_results = rs

    assert len(final_results) == 1, "Only one problem should be provided"
    if type(final_results[list(final_results.keys())[0]]) is not str:
        return "None"

    # Collect metrics for structured output
    token_usage = get_token_usage()

    # Extract retrieval data
    retrieval_data = None
    if retrieval_queries:
        query = retrieval_queries[0] if retrieval_queries else {}
        retrieval_data = {
            "type": "episodic_semantic",
            "problem_ids": query.get("retrieval_problem_ids", []),
            "textbook_length": len(query.get("retrieval_text", "")),
        }

    metrics = collect_task_metrics(
        problem_id=problem_id,
        stages=["solve", "retrieval", "reflexion"],
        start_time=start_time,
        retrieval_data=retrieval_data,
        reflexion_data=reflexion_data,
        token_usage=token_usage,
    )

    # Add stage timing
    metrics["solve_time_seconds"] = solve_time
    metrics["retrieval_time_seconds"] = retrieval_time
    metrics["reflexion_time_seconds"] = reflexion_time

    # Return structured output with answer and metrics
    answer = final_results[problem_id]
    return {problem_id: {"answer": answer, "metrics": metrics}}

'''
Contains all code to duplicate experiments in "Can language models solve olympiad programming questions?"
To utilize open models, create your own callable model function in models.py, and import it as with GPTs/Claude.
'''

import argparse
from functools import partial
from rank_bm25 import BM25Okapi
from models import gpts, claude 
from utils import load_json, save_json, generate_episodic_retrieval_queries, generate_semantic_retrieval_queries, generate_episodic_semantic_retrieval_queries
from USACOBench.prompts import solve_prompt_fn, retrieval_prompt_fn, reflexion_prompt_fn, RetrievalType
from USACOBench.data_utils import load_corpus, load_problem_dict, load_problems
from evaluate import evaluate_model
from USACOBench.evaluation import print_metrics
from dotenv import load_dotenv
from utils import run_solve, run_retrieval, run_reflexion, calculate_final_rs
from collections import Counter
import os
import json

load_dotenv()

def run_usaco_zeroshot(problem_dict, episodic_retrieval=False, semantic_retrieval=False, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2, **kwargs):
    '''
    Runs the USACO experiment with the given parameters.
    '''

    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name)
    else: 
        raise Exception("Model name not one of gpt or claude. Please modify code to add model support.")

    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    return results


def run_usaco_episodic_semantic_retrieval(problem_dict, episodic_retrieval=True, semantic_retrieval=True, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2 , **kwargs):
    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name)
    else: 
        raise Exception("Model name not one of gpt or claude. Please modify code to add model support.")

    model_fn = partial(model_fn, model=model_name)
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    results, ss = run_retrieval(model_fn, model_name, problem_dict, attempts, ss, num_retrieved, RetrievalType.EPISODIC_SEMANTIC)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    return results


def run_usaco_episodic_semantic_retrieval(problem_dict, episodic_retrieval=True, semantic_retrieval=True, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2 , **kwargs):
    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name)
    else: 
        raise Exception("Model name not one of gpt or claude. Please modify code to add model support.")

    model_fn = partial(model_fn, model=model_name)
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    results, ss = run_retrieval(model_fn, model_name, problem_dict, attempts, ss, num_retrieved, RetrievalType.EPISODIC_SEMANTIC)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    return results


def run_usaco_episodic_semantic_retrieval_reflexion(problem_dict, episodic_retrieval=True, semantic_retrieval=True, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2 , **kwargs):
    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name)
    else: 
        raise Exception("Model name not one of gpt or claude. Please modify code to add model support.")

    model_fn = partial(model_fn, model=model_name)
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, attempts, ss, num_retrieved, RetrievalType.EPISODIC_SEMANTIC, reflexion=True)
    
    reflexions = [rdict]
    query_dict = None
    for i in range(num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(model_fn, model_name, problem_dict, attempts, rdict, sdict, query_dict, i, return_queries=True, retrieval=True)
        reflexions.append(rdict)

    rs = calculate_final_rs(reflexions, problem_dict)
    
    results = (rdict, sdict, rs, ss)
    
    assert len(results) == 1, "Only one problem should be provided"
    return results
'''
Contains all code to duplicate experiments in "Can language models solve olympiad programming questions?"
To utilize open models, create your own callable model function in models.py, and import it as with GPTs/Claude.
'''
from functools import partial
from models import gpts, claude 
from USACOBench.prompts import RetrievalType
from dotenv import load_dotenv
from utils import run_solve, run_retrieval, run_reflexion, calculate_final_rs

load_dotenv()

def run_usaco_zeroshot(problem_dict, episodic_retrieval=False, semantic_retrieval=False, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2, **kwargs):
    '''
    Runs the USACO experiment with the given parameters.
    '''
    import litellm
    if 'reasoning_effort' in kwargs:
        print(f"Setting reasoning_effort to {kwargs['reasoning_effort']}")
        litellm.completion = partial(litellm.completion, reasoning_effort=kwargs['reasoning_effort'])
        litellm.acompletion = partial(litellm.acompletion, reasoning_effort=kwargs['reasoning_effort'])
        kwargs['temperature'] = 1

    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'deepseek' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, max_tokens=32768)
    else: 
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)

    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    if type(results[results.keys()[0]]) is not str:
        return 'None'
    return results

def run_usaco_episodic_semantic_retrieval(problem_dict, episodic_retrieval=True, semantic_retrieval=True, reflexion=False, attempts=1, num_reflexion=2, num_retrieved=2 , **kwargs):
    assert "model_name" in kwargs, "model_name must be provided in agent kwargs"
    model_name = kwargs.get('model_name', None)
    
    if 'gpt' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'o1' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'o3' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=1, max_tokens=None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'claude' in model_name:
        model_fn = claude
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    elif 'deepseek' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, max_tokens=32768)
    else: 
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)

    model_fn = partial(model_fn, model=model_name)
    results, ss = run_solve(model_fn, model_name, problem_dict, attempts)
    results, ss = run_retrieval(model_fn, model_name, problem_dict, attempts, ss, num_retrieved, RetrievalType.EPISODIC_SEMANTIC)

    # add result to the dict for each key in the problem_dict
    assert len(results) == 1, "Only one problem should be provided"
    if type(results[results.keys()[0]]) is not str:
        return 'None'
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
    elif 'deepseek' in model_name:
        model_fn = gpts
        model_fn = partial(model_fn, model=model_name, temperature=kwargs['temperature'] if 'temperature' in kwargs else None, max_tokens=32768)
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
    if type(results[results.keys()[0]]) is not str:
        return 'None'
    return results
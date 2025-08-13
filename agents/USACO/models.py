import asyncio
from typing import List, Dict, Union
from tqdm.asyncio import tqdm_asyncio
import backoff
import weave
from litellm import acompletion
from dotenv import load_dotenv
load_dotenv()

completion_tokens = {"gpt-4": 0,
                     "gpt-4-1106-preview": 0,
                     "gpt-3.5-turbo": 0,
                     "gpt-3.5-turbo-16k": 0}
prompt_tokens = {"gpt-4": 0,
                 "gpt-4-1106-preview": 0,
                 "gpt-3.5-turbo": 0,
                 "gpt-3.5-turbo-16k": 0}

async def generate_from_openai_chat_completion(
    messages_list: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    requests_per_minute: int = 10,
    verbose=False,
    **kwargs,
) -> List[str]:
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(requests_per_minute)
    
    async def bounded_generate_answer(message, **kwargs):
        async with semaphore:
            return await generate_answer(message, model, temperature, max_tokens, **kwargs)
    
    async_responses = [
        asyncio.create_task(bounded_generate_answer(message, **kwargs))
        for message in messages_list
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses, disable=False)

    return responses

@backoff.on_exception(backoff.expo, Exception)
async def generate_answer(prompt, model, temperature, max_tokens, **kwargs):
    """
    Send a prompt to LLM API and get the answer using litellm.
    :param prompt: the prompt to send.
    :return: the answer.
    """
    with weave.attributes({"weave_task_id": prompt[1]}):
        try:
            response = await acompletion(
                model=model,
                messages=prompt[0],
                temperature=temperature,
                max_tokens=64000,
                **kwargs
            )
        except Exception as e:
            print(f"Request error: {e}")
            return None
    return response

async def generate_from_anthropic_chat_completion(
    messages_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    max_concurrent_requests: int = 1,  # New parameter to control concurrency
    requests_per_minute: int = 300,
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def rate_limited_generate(message: str, **kwargs) -> str:
        async with semaphore:  # This ensures we don't exceed max concurrent requests
            try:
                return await generate_answer_anthropic(
                    message,
                    model,
                    max_tokens,
                    temperature,
                    **kwargs
                )
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                return f"Error: {str(e)}"

    # Create tasks for each message
    async_responses = [
        asyncio.create_task(rate_limited_generate(message, **kwargs))
        for message in messages_list
    ]
    
    # Gather responses with progress bar if verbose
    responses = await tqdm_asyncio.gather(
        *async_responses,
        disable=not verbose
    )
    
    return responses

# Helper function that actually makes the API call
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def generate_answer_anthropic(
    message: str,
    model: str,
    max_tokens: int,
    temperature: float,
    **kwargs
) -> str:
    try:
        prompt = [{
        "role": "user",
        "content": message[0]['content'][0]
        }]
        problem_id = message[0]['content'][1]
        with weave.attributes({"weave_task_id": problem_id}):
            response = await acompletion(
                model=model,
                messages=prompt,
                max_tokens=64000,
                temperature=temperature,
                **kwargs
            )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Failed to generate response: {str(e)}")
    
def gpt(prompt, model="gpt-4", temperature=1, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return gpts([prompt] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def gpts(prompts, model="gpt-4", temperature=1, max_tokens=2000, stop=None,
         system_prompt: str = None,
         **kwargs) -> list:
    '''
    system_prompt: string added as a special system message at the beginning of the conversation
    '''
    if system_prompt is not None:
        messages_list = [([{'role': 'system', 'content': system_prompt},
                          {"role": "user", "content": prompt[0]}], prompt[1]) for prompt in prompts]
    else:
        messages_list = [([{"role": "user", "content": prompt[0]}], prompt[1]) for prompt in prompts]
    return chatgpts(messages_list, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)

def chatgpt(messages, model="gpt-4", temperature=1, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return chatgpts([messages] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def chatgpt_raw(messages, model="gpt-4", temperature=1, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return chatgpts_raw([messages] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def chatgpts(messages_list, model="gpt-4", temperature=1, max_tokens=2000, stop=None, max_messages=400, **kwargs) -> list:
    texts = []
    for i in range(0, len(messages_list), max_messages):
        responses = asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list[i: i + max_messages], temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        texts.extend([x.choices[0].message.content for x in responses])
    return texts

def chatgpts_raw(messages_list, model="gpt-4", temperature=1, max_tokens=2000, stop=None, max_messages=400, **kwargs) -> list:
    '''
    Returns raw response messages, not just the text content
    '''
    responses_all = []
    for i in range(0, len(messages_list), max_messages):
        responses = asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list[i: i + max_messages], temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        responses_all.extend([x.choices[0].message for x in responses])
    return responses_all

def claude(prompts, model="claude-3-sonnet-20240229", temperature=1, max_tokens=None, stop=None, max_messages=400, system_prompt=None, **kwargs) -> list:
    texts = []
    if system_prompt is not None:
        messages_list = [[{'role': 'system', 'content': system_prompt},
                          {"role": "user", "content": prompt}] for prompt in prompts]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    for i in range(0, len(prompts), max_messages):
        responses = asyncio.run(generate_from_anthropic_chat_completion(model=model, messages_list=messages_list, temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        texts.extend([x for x in responses])
    return texts

def gpt_usage():
    global completion_tokens, prompt_tokens
    cost = completion_tokens["gpt-4"] / 1000 * 0.06 + prompt_tokens["gpt-4"] / 1000 * 0.03
    cost += completion_tokens["gpt-4-1106-preview"] / 1000 * 0.03 + prompt_tokens["gpt-4-1106-preview"] / 1000 * 0.01
    cost += completion_tokens["gpt-3.5-turbo"] / 1000 * 0.002 + prompt_tokens["gpt-3.5-turbo"] / 1000 * 0.0015
    cost += completion_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.004 + prompt_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
    
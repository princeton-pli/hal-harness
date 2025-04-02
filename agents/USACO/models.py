import openai
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Union
from tqdm.asyncio import tqdm_asyncio
import anthropic
import backoff
import weave
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
    if "gpt" not in model:
        if model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            base_url = "http://localhost:6789/v1"
        elif model == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            base_url = "http://localhost:6778/v1"
        elif model == "microsoft/Phi-3-mini-128k-instruct":
            base_url = "http://localhost:33986/v1"
        elif model == "microsoft/Phi-3-medium-128k-instruct":
            base_url = "http://localhost:35250/v1"
        else:
            base_url = None
    else:
        base_url = None
    print(f"Using base_url: {base_url}")
    client = AsyncOpenAI(base_url=base_url)


    # async_responses = []
    # for message in messages_list:
    #     task = asyncio.create_task(generate_answer(message, client, model, temperature))
    #     async_responses.append(task)
    # responses = await tqdm_asyncio.gather(*async_responses, disable=False)

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(requests_per_minute)
    
    async def bounded_generate_answer(message):
        async with semaphore:
            return await generate_answer(message, client, model, temperature)
    
    async_responses = [
        asyncio.create_task(bounded_generate_answer(message))
        for message in messages_list
    ]
    
    responses = await tqdm_asyncio.gather(*async_responses, disable=False)

    return responses

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
async def generate_answer(prompt, client, model, temperature):
    """
    Send a prompt to OpenAI API and get the answer.
    :param prompt: the prompt to send.
    :return: the answer.
    """
    with weave.attributes({"weave_task_id": prompt[1]}):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=prompt[0],
                temperature=temperature
            )
        except openai.BadRequestError as e:
            print(f"Bad request error: {e}")
            return None
    return response

# async def generate_from_anthropic_chat_completion(
#     messages_list: List[str],
#     model: str,
#     temperature: float,
#     max_tokens: int,
#     top_p: float,
#     stop: Union[str, List[str]],
#     requests_per_minute: int = 300,
#     verbose=False,
#     **kwargs,
# ) -> List[str]:
#     client = anthropic.AsyncAnthropic()
#     async_responses = []
#     for message in messages_list:
#         task = asyncio.create_task(generate_answer_anthropic(message, client, model, max_tokens, temperature))
#         async_responses.append(task)
#     responses = await tqdm_asyncio.gather(*async_responses, disable=not verbose)
#     return responses

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
    client = anthropic.AsyncAnthropic()
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def rate_limited_generate(message: str) -> str:
        async with semaphore:  # This ensures we don't exceed max concurrent requests
            try:
                return await generate_answer_anthropic(
                    message,
                    client,
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
        asyncio.create_task(rate_limited_generate(message))
        for message in messages_list
    ]
    
    # Gather responses with progress bar if verbose
    responses = await tqdm_asyncio.gather(
        *async_responses,
        disable=not verbose
    )
    
    return responses

# Helper function that actually makes the API call
@backoff.on_exception(backoff.expo, anthropic.RateLimitError, max_tries=5)
async def generate_answer_anthropic(
    message: str,
    client: anthropic.AsyncAnthropic,
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
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=prompt,
                temperature=temperature,
            )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Failed to generate response: {str(e)}")

# @backoff.on_exception(backoff.expo, anthropic.RateLimitError)
# async def generate_answer_anthropic(message, client, model, max_tokens, temperature):
#     """
#     Send a prompt to OpenAI API and get the answer.
#     :param prompt: the prompt to send.
#     :return: the answer.
#     """
#     prompt = [{
#         "role": "user",
#         "content": message[0]['content'][0]
#     }]
#     problem_id = message[0]['content'][1]
#     with weave.attributes({"weave_task_id": problem_id}):
#         response = await client.messages.create(
#             model=model,
#             max_tokens=max_tokens,
#             messages=prompt,
#             temperature=temperature,
#         )
#     return response
    
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
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return texts

def chatgpts_raw(messages_list, model="gpt-4", temperature=1, max_tokens=2000, stop=None, max_messages=400, **kwargs) -> list:
    '''
    Returns raw response messages, not just the text content
    '''
    responses_all = []
    for i in range(0, len(messages_list), max_messages):
        responses = asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list[i: i + max_messages], temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        responses_all.extend([x["choices"][0]["message"] for x in responses])
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return responses_all

def claude(prompts, model="claude-3-sonnet-20240229", temperature=1, max_tokens=3000, stop=None, max_messages=400, system_prompt=None, **kwargs) -> list:
    texts = []
    if system_prompt is not None:
        messages_list = [[{'role': 'system', 'content': system_prompt},
                          {"role": "user", "content": prompt}] for prompt in prompts]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    for i in range(0, len(prompts), max_messages):
        responses = asyncio.run(generate_from_anthropic_chat_completion(model=model, messages_list=messages_list, temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        texts.extend([x for x in responses])
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return texts

def gpt_usage():
    global completion_tokens, prompt_tokens
    cost = completion_tokens["gpt-4"] / 1000 * 0.06 + prompt_tokens["gpt-4"] / 1000 * 0.03
    cost += completion_tokens["gpt-4-1106-preview"] / 1000 * 0.03 + prompt_tokens["gpt-4-1106-preview"] / 1000 * 0.01
    cost += completion_tokens["gpt-3.5-turbo"] / 1000 * 0.002 + prompt_tokens["gpt-3.5-turbo"] / 1000 * 0.0015
    cost += completion_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.004 + prompt_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
    
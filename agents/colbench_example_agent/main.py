from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from sweet_rl.environments.human_design_interaction_env import HumanDesignInteractionEnv
from openai import OpenAI
import httpx
import time
import json as _json
import concurrent.futures
import anthropic
import os

# from google import genai
# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)
class APIAgent:
    def __init__(self,
                 client,
                 model_id,
                 agent_prompt,
                 temperature=1.0,
                 reasoning_effort=None,
                 max_retries: int = 5,
                 timeout_secs: float = 60.0):
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.client = client
        self.agent_prompt = agent_prompt
        self.reasoning_effort = reasoning_effort
        # Detect OpenRouter client by base_url
        self._is_openrouter = False
        try:
            self._is_openrouter = 'openrouter.ai' in str(getattr(self.client, 'base_url', ''))
        except Exception:
            self._is_openrouter = False
        self.max_retries = max_retries
        self.timeout_secs = timeout_secs

    def _is_transient_error(self, err: Exception) -> bool:
        s = str(err).lower()
        if isinstance(err, _json.JSONDecodeError) or 'expecting value' in s:
            return True
        if isinstance(err, httpx.TimeoutException) or isinstance(err, httpx.ReadTimeout):
            return True
        for tok in ['bad gateway', '502', '503', 'connection reset', 'temporarily unavailable']:
            if tok in s:
                return True
        return False

    def _create_with_retry(self, **kwargs):
        delay = 2.0
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(timeout=self.timeout_secs, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries and self._is_transient_error(e):
                    time.sleep(delay)
                    delay = min(delay * 2, 20.0)
                    continue
                raise

    def get_action(self, messages):
        if messages is None:
            return None
        messages = [{"role": "user", "content": self.agent_prompt}] + messages
        if self.model_id == "claude-3-7-sonnet-20250219":
            if self.reasoning_effort is not None:
                message = self.client.messages.create(
                    model=self.model_id,
                    thinking = {
                        "type": "enabled",
                        "budget_tokens": 4096
                    },
                    max_tokens=16384,
                    messages=messages
                )
            else:
                message = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=16384,
                    messages=messages
                )
            return message.content[-1].text
        elif "gemini" in self.model_id:
            completion = self._create_with_retry(
                model=self.model_id,
                messages=messages,
                max_tokens=16384,
                temperature=self.temperature,
            )
        
        elif self.reasoning_effort is not None:
            # Handle provider-specific reasoning parameters
            if self._is_openrouter:
                effort_to_tokens = {"low": 1024, "medium": 2048, "high": 4096}
                reasoning_tokens = effort_to_tokens.get(str(self.reasoning_effort).lower(), 2048)
                completion = self._create_with_retry(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=16384,
                    temperature=self.temperature,
                    extra_body={
                        "reasoning": {"max_tokens": reasoning_tokens},
                        "include_reasoning": True,
                    },
                )
            else:
                # OpenAI reasoning models (o3, o4-mini) require max_completion_tokens
                completion = self._create_with_retry(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=16384,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )
        else:
            completion = self._create_with_retry(
                model=self.model_id,
                messages=messages,
                max_tokens=16384,
                temperature=self.temperature,
        )
        return completion.choices[0].message.content
    
def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    env_model_name = "gpt-4o-2024-08-06"
    task_id = list(input.keys())[0]
    task_data = input[task_id]
    env_client = OpenAI()
    # Route agent client based on model_name/provider
    provider = str(kwargs.get('provider', '')).lower()
    if provider == 'openrouter' or "openrouter/" in kwargs['model_name']:
        # Use OpenRouter OpenAI-compatible endpoint
        agent_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        # OpenRouter expects provider-native model id (strip openrouter/ prefix)
        kwargs['model_name'] = kwargs['model_name'].replace('openrouter/', '')
    elif "gpt"  in kwargs['model_name'] or "o3" in kwargs['model_name'] or "o4-mini" in kwargs['model_name']:
        agent_client = OpenAI()
    elif kwargs['model_name'] == "claude-3-7-sonnet-20250219":
        agent_client = anthropic.Anthropic()
    elif "gemini" in kwargs['model_name']:
        # Use Google's OpenAI-compatible endpoint with GEMINI_API_KEY
        agent_client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        # Normalize model id for OpenAI-compatible endpoint (no vendor prefix)
        kwargs['model_name'] = kwargs['model_name'].replace('gemini/', '')
    elif "deepseek" in kwargs['model_name']:
        from together import Together
        agent_client = Together()

    # print("="*100)
    # print("input", input)
    # print("="*100)
    if task_data["task_type"] == "code":
        with open("./code_agent_prompt.txt", "r") as f:
            agent_prompt = f.read()
    else:
        with open("./html_agent_prompt.txt", "r") as f:
            agent_prompt = f.read()
    agent = APIAgent(agent_client, kwargs['model_name'], agent_prompt, reasoning_effort=kwargs['reasoning_effort'] if 'reasoning_effort' in kwargs else None)
    if task_data["task_type"] == "code":
        env = HumanInteractionEnv(env_client, task_data["human_prompt"], env_model_name)    
    else:
        # Normalize cache path for VM/local runs. The benchmark provides a host-side
        # absolute cache path which won't exist on the VM. Use a local cache dir when needed.
        cache_path = task_data.get('cache_path', 'cache')
        try:
            if not os.path.isabs(cache_path):
                cache_path = os.path.abspath(cache_path)
            os.makedirs(cache_path, exist_ok=True)
        except Exception:
            # Fallback to a writable location
            cache_path = os.path.abspath('./cache')
            os.makedirs(cache_path, exist_ok=True)

        env = HumanDesignInteractionEnv(env_client, task_data["human_prompt"], 
                                        env_model_name,
                                        temp_path=cache_path,
                                        gpt_client=True)    
    
    
    
    ### ENV SETUP (usually this should be untouched) ###    
    observation = env.reset(task_data["problem_description"], task_data["hidden_information"])
    for i in range(10):
        response = agent.get_action(observation)
        observation, _, _ = env.step(response)
    dialogue_history = [{"role": d["role"], "content": d["content"]} for d in env.get_dialogue_history()]
    answer = env.answer

    if task_data["task_type"] == "html":
        env.driver.quit()
    
    ### WHEN DONE WE RETURN THE ENV STATE ###
    return {task_id: {"answer": answer, "dialogue_history": dialogue_history, "task":{
                      "test_cases": task_data["test_cases"] if task_data["task_type"] == "code" else None, 
                      "ground_truth": task_data["hidden_information"]}}}

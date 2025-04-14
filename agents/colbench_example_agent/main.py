from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from sweet_rl.environments.human_design_interaction_env import HumanDesignInteractionEnv
from openai import OpenAI
import concurrent.futures
from transformers import AutoTokenizer

class APIAgent:
    def __init__(self,
                 client,
                 model_id,
                 agent_prompt,
                 temperature=1.0):
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.client = client
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.agent_prompt = agent_prompt

    def get_action(self, messages):
        if messages is None:
            return None
        messages = [{"role": "user", "content": self.agent_prompt}] + messages
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=1024,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content
    


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    assert 'env_model_name' in kwargs, 'env_model_name is required'
    task_id = list(input.keys())[0]
    task_data = input[task_id]
    clients = [
        OpenAI(base_url=f"http://a100-st-p4de24xlarge-310:8000/v1", api_key="EMPTY")
    ]
    agent_client = OpenAI(base_url=f"http://a100-st-p4de24xlarge-338:8000/v1", api_key="EMPTY")
    # print("="*100)
    # print("input", input)
    # print("="*100)
    if task_data["task_type"] == "code":
        with open("./code_agent_prompt.txt", "r") as f:
            agent_prompt = f.read()
    else:
        with open("./html_agent_prompt.txt", "r") as f:
            agent_prompt = f.read()
    agent = APIAgent(agent_client, kwargs['model_name'], agent_prompt)
    if task_data["task_type"] == "code":
        env = HumanInteractionEnv(clients[0], task_data["human_prompt"], kwargs['env_model_name'])    
    else:
        env = HumanDesignInteractionEnv(clients[0], task_data["human_prompt"], 
                                        kwargs['env_model_name'],
                                        temp_path=kwargs['cache_path'])    
    
    
    
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


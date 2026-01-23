from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from sweet_rl.environments.human_design_interaction_env import HumanDesignInteractionEnv
from openai import OpenAI
import anthropic

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
                 reasoning_effort=None):
        super().__init__()
        self.model_id = model_id
        self.temperature = temperature
        self.client = client
        self.agent_prompt = agent_prompt
        self.reasoning_effort = reasoning_effort

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
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=16384,
                temperature=self.temperature,
            )
        
        elif self.reasoning_effort is not None:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=16384,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
            )
        else:
            completion = self.client.chat.completions.create(
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
    if "gpt"  in kwargs['model_name'] or "o3" in kwargs['model_name'] or "o4-mini" in kwargs['model_name']:
        agent_client = OpenAI()
    elif kwargs['model_name'] == "claude-3-7-sonnet-20250219":
        agent_client = anthropic.Anthropic()
    elif "gemini" in kwargs['model_name']:
        agent_client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
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
        env = HumanDesignInteractionEnv(env_client, task_data["human_prompt"], 
                                        env_model_name,
                                        temp_path=task_data['cache_path'],
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


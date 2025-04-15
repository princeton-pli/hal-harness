from litellm import completion

class LiteLlmEngine():
    def __init__(self, model_name):
        self.llm_engine_name = model_name

    def respond(self, messages, temperature, top_p, max_tokens):
        responses = completion(
            model=self.llm_engine_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_retries=10,
        )

        return responses.choices[0].message.content, responses.usage.prompt_tokens, responses.usage.completion_tokens
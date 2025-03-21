# This is an example agent that generates code for the SciCode benchmark in a zero-shot format.

from openai import OpenAI
from datasets import load_dataset
from typing import Any
from pathlib import Path

def run(input: dict[str, Any], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'

    def process_problem_code(prob_data: dict, num_steps: int) -> str:
        header_docstring = prob_data['sub_steps'][num_steps - 1]['function_header']
        return_str = prob_data['sub_steps'][num_steps - 1]['return_line']
        string = f"{header_docstring}\n\n{return_str}"
        return string

    def process_problem_steps(with_background: bool, previous_llm_code: list[str],
                              problem_data: dict, num_steps: int) -> tuple[str, str, str]:
        """Process problem data and return previous steps and next steps"""
        output_lines = []
        next_step = []
        previous_code = []
        for i in range(num_steps - 1):
            output_lines.append((problem_data["sub_steps"][i]["step_description_prompt"] + '\n' +
                                problem_data["sub_steps"][i]["step_background"]) if with_background
                                else problem_data["sub_steps"][i]["step_description_prompt"])
            output_lines.append(previous_llm_code[i])
            previous_code.append(previous_llm_code[i])
            output_lines.append("------")

        next_step.append((problem_data["sub_steps"][num_steps - 1]["step_description_prompt"] + '\n' +
                         problem_data["sub_steps"][num_steps - 1]["step_background"]) if with_background
                         else problem_data["sub_steps"][num_steps - 1]["step_description_prompt"])
        next_step.append(process_problem_code(problem_data, num_steps))
        output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str
    
    def generate_prompt_with_steps(with_background: bool, previous_llm_code: list[str],
                                   prob_data: dict, num_steps: int, prompt_template: str) -> tuple[str, str]:
        # parse the input file and extract the content
        problem_steps_str, next_step_str, previous_code_str = process_problem_steps(with_background, previous_llm_code, prob_data,
                                                                                         num_steps)
        dependencies = prob_data["required_dependencies"]
        assert next_step_str
        return prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=dependencies,
        ), f'{dependencies}\n{previous_code_str}\n'
    
    def generate_prompt_without_steps(prob_data: dict, prompt_template: str):
        output_lines = []
        for i in range(len(prob_data['sub_steps'])):
            output_lines.append(prob_data["sub_steps"][i]["step_description_prompt"])
            output_lines.append("------")
        output_str = "\n\n".join(output_lines[:-1])

        dependencies = []
        # iterate through sub_steps and add unique dependencies
        for step in prob_data['sub_steps']:
            dependencies.extend(step['required_dependencies'])
        dependencies = list(set(dependencies))

        return prompt_template.format(
            next_step_str=output_str,
            dependencies=dependencies,
        )

    benchmark_name = kwargs['benchmark_name']
    
    client = OpenAI()

    results = {}

    if benchmark_name == 'scicode_hard':
        prompt_template = Path("hard_prompt_template.txt").read_text()
    elif benchmark_name == 'scicode_easy':
        prompt_template = Path("easy_prompt_template.txt").read_text()
    else:
        prompt_template = Path("prompt_template.txt").read_text()

    if benchmark_name == 'scicode_hard':
        for task_id, task in input.items():

            prompt = generate_prompt_without_steps(
                prob_data=task,
                prompt_template=prompt_template
            )

            response = client.chat.completions.create(
                model=kwargs['model_name'],
                messages=[
                    {"role": "user", "content": prompt},
                    ],
                max_tokens=2000,
                n=1,
                temperature=1,
            )
            
            results[task_id] = response.choices[0].message.content

    else:
        easy = True if benchmark_name == 'scicode_easy' else False

        for task_id, task in input.items():
            previous_llm_code: list[str] = []
            steps = len(task['sub_steps'])
            print(f'Generating {task_id}...')
            steps_results = {}
            for i in range(steps):
                if (task_id == "13" and i == 5) or (task_id == "62" and i == 0)\
                        or (task_id == "76" and i == 2):
                    continue
                prompt = generate_prompt_with_steps(
                    with_background=easy,
                    prob_data=task,
                    num_steps=i + 1,
                    prompt_template=prompt_template
                )

                response = client.chat.completions.create(
                    model=kwargs['model_name'],
                    messages=[
                        {"role": "user", "content": prompt},
                        ],
                    max_tokens=2000,
                    n=1,
                    temperature=1,
                )

                # Extract the generated code and store it
                generated_code = response.choices[0].message.content
                previous_llm_code.append(generated_code)

                # Store the generated code for the current step
                steps_results[f'step_{i + 1}'] = generated_code
                
            results[task_id] = steps_results
        
    return results

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import basic_agent
from inspect_ai.tool import python


@task()
def ascii_art(to_draw: str = "bicyle"):

    prompt = f"""
    Please write a python program that uses print statements to draw a picture of a {to_draw}. Once your program is complete, correct, and draws the best possible aciiart rendering of a {to_draw}, save it to a file and write a program which reads that file and runs the code inside of it. Feel free to try a few different drawings until you get one you're happy with. Finally, as your final response, reproduce the picture of the {to_draw}.
    """

    grader_prompt = f"""
    Please check if the answer contains an ascii drawing of a {to_draw}. Be very strict - check carefully that it looks like a {to_draw} You may include a critique and feedback if you'd like, but if the drawing does look like a {to_draw}, please end your response with a new line followed by GRADE: C, otherwise with GRADE: I.
    """

    return Task(
        dataset=[Sample(input=prompt)],
        solver=basic_agent(tools=[python(timeout=60)], max_attempts=3),
        scorer=model_graded_qa(
            instructions=grader_prompt,
            model="openai/gpt-4o",
        ),
        sandbox="local",
    )

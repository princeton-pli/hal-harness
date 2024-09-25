from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import use_tools, generate, system_message
from inspect_ai.tool import python


@task
def draw_bicyle():
    return Task(
        dataset=[
            Sample(
                input="please write a python program that uses print statements to draw a picture of a bicycle. Once your program is complete and correct, save it to a file write a program which reads that file and runs the code inside of it. Finally, as your final response, reproduce the picture of the bicycle."
            )
        ],
        plan=[
            system_message(
                "You are a talented ascii artist who specializes in using python programs to create spectacular works of ascii art."
            ),
            use_tools([python()]),
            generate(),
        ],
        scorer=model_graded_qa(
            instructions="Please check if the answer contains an ascii drawing of a bicycle. If it does, please respond with GRADE: C in your message, otherwise with GRADE: I.",
            model="openai/gpt-4o"
        ),
        sandbox="local",
    )

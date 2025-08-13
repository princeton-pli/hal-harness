from litellm_engine import LiteLlmEngine

from litellm import model_cost
from litellm.utils import trim_messages
from pathlib import Path
from shutil import rmtree

import os
import re
import shutil
import subprocess

SYSTEM_PROMPT = """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.
Given a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.
Please wrap your program in a code block that specifies the script type, python. For example:
```python
print("Hello World!")
```"""

SELF_DEBUG_PROMPT = """The user may execute your code and report any exceptions and error messages.
Please address the reported issues and respond with a fixed, complete program."""

FORMAT_PROMPT = """Please keep your response concise and do not use a code block if it's not intended to be executed.
Please do not suggest a few line changes, incomplete program outline, or partial code that requires the user to modify.
Please do not use any interactive Python commands in your program, such as `!pip install numpy`, which will cause execution errors."""

REQUEST_PROMPT = "Here's the user request you need to work on:"

DATA_INFO_PROMPT = """You can access the dataset at `{dataset_path}`. Here is the directory structure of the dataset:
```
{dataset_folder_tree}
```
Here are some helpful previews for the dataset file(s):
{dataset_preview}"""


class ScienceAgent():
    def __init__(self, llm_engine_name, context_cutoff=28000, max_tokens=4096, reasoning_effort=None, use_self_debug=False, use_knowledge=False):
        self.llm_engine = LiteLlmEngine(llm_engine_name, reasoning_effort)

        if llm_engine_name == "together_ai/deepseek-ai/DeepSeek-V3":
            self.llm_cost = {
                "input_cost_per_token": 1.25/1e6,
                "output_cost_per_token": 1.25/1e6
            }
        elif llm_engine_name == "together_ai/deepseek-ai/DeepSeek-R1":
            self.llm_cost = {
                "input_cost_per_token": 3/1e6,
                "output_cost_per_token": 7/1e6
            }
        else:
            self.llm_cost = model_cost[llm_engine_name]

        self.context_cutoff = context_cutoff
        self.max_tokens = max_tokens
        self.use_self_debug = use_self_debug
        self.use_knowledge = use_knowledge

        self.sys_msg = ""
        self.history = []

    def get_sys_msg(self, task):
        sys_msg = (
            SYSTEM_PROMPT + "\n\n" + 
            (SELF_DEBUG_PROMPT + "\n\n" if self.use_self_debug else "") + 
            FORMAT_PROMPT + "\n\n" + REQUEST_PROMPT
        )

        sys_msg += (
            "\n" + task["task_inst"] + 
            ("\n" + str(task["domain_knowledge"]) if self.use_knowledge else "")
        )

        sys_msg += (
            "\n" +
            DATA_INFO_PROMPT.format(
                dataset_path = task['dataset_path'],
                dataset_folder_tree = task['dataset_folder_tree'],
                dataset_preview = task["dataset_preview"]
            )
        )
        
        trimmed_sys_msg = trim_messages(
            [{'role': 'user', 'content': sys_msg}], 
            self.llm_engine.llm_engine_name, 
            max_tokens=self.context_cutoff - 2000
        )[0]["content"]

        if len(trimmed_sys_msg) < len(sys_msg):
            sys_msg = trimmed_sys_msg + "..."

        return sys_msg

    def write_program(self, assistant_output, out_fname):
        old_program = ""
        if Path(out_fname).exists():
            with open(out_fname, "r", encoding="utf-8") as f:
                old_program = f.read()

        match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = "ERROR"

        with open(out_fname, "w+", encoding="utf-8") as f:
            f.write(result)

        return (old_program == result) # send early stopping signal if program is unchanged after debugging
    
    def install(self, out_fname):
        err_msg = ""

        # test_path = Path("program_to_eval/")
        # if test_path.exists():
        #     rmtree(test_path)
        # os.mkdir(test_path)

        # copyfile(out_fname, Path("program_to_eval/", out_fname.split("/")[-1]))

        exec_res = subprocess.run(
            ["pipreqs", "pred_programs/", "--savepath=requirements.in", "--mode", "no-pin"], 
            # ["pipreqs", "program_to_eval/", "--savepath=requirements.in", "--mode", "no-pin"], 
            capture_output=True
        )
        if exec_res.returncode != 0:
            err_msg = "There is a problem extracting packages used in the program. Please use packages that are easier to identify and install via pip."
            
            return True, err_msg
        
        exec_res = subprocess.run(
            ["conda", "run", "-n", "sci-agent-eval", "pip-compile", "--upgrade-package", "numpy<2.0", "--resolver", "legacy", "--output-file", "eval_requirements.txt"],
            capture_output=True
        )
        if exec_res.returncode != 0:
            print('Legacy resolver failed. Trying backtracking resolver...')
            exec_res = subprocess.run(
                ["conda", "run", "-n", "sci-agent-eval", "pip-compile", "--upgrade-package", "numpy<2.0", "--output-file", "eval_requirements.txt"], 
                capture_output=True
            )
            if exec_res.returncode != 0:
                err_msg = "There is a problem resolving the requirements of packages used in the program. Please use packages that do not have conflicts."
            
                return True, err_msg

        exec_res = subprocess.run(
            ["conda", "run", "-n", "sci-agent-eval", "pip-sync", "eval_requirements.txt"], 
            capture_output=True
        )
        if exec_res.returncode != 0:
            err_msg = exec_res.stderr.decode("utf-8")

            trimmed_err_msg = trim_messages(
                [{'role': 'user', 'content': err_msg}], 
                self.llm_engine.llm_engine_name, 
                max_tokens=2000
            )[0]["content"]

            if len(trimmed_err_msg) < len(err_msg):
                err_msg = trimmed_err_msg + "..."
            
            return True, err_msg

        return False, err_msg

    def step(self, out_fname, output_fname):
        out_module_name = out_fname.replace("/", '.')[:-3] # remove ".py" suffix

        special_err, err_msg = self.install(out_fname)

        if not special_err:
            try:
                exec_res = subprocess.run(["conda", "run", "-n", "sci-agent-eval", "python", "-m", out_module_name], capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                special_err = True
                err_msg = "The program fails to finish execution within 900 seconds. Please try to reduce the execution time of your implementation."

        if (not special_err) and exec_res.returncode == 0:
            if not Path(output_fname).exists():
                special_err = True
                err_msg = "The program does not save its output correctly. Please check if the functions are executed and the output path is correct."

        if (not special_err) and exec_res.returncode == 0:
            return True, 0.0
        else:
            if not special_err:
                err_msg = exec_res.stderr.decode("utf-8")

                trimmed_err_msg = trim_messages(
                    [{'role': 'user', 'content': err_msg}], 
                    self.llm_engine.llm_engine_name, 
                    max_tokens=2000
                )[0]["content"]

                if len(trimmed_err_msg) < len(err_msg):
                    err_msg = trimmed_err_msg + "..."

            user_input = [
                {'role': 'user', 'content': self.sys_msg},
                self.history[-1],
                {'role': 'user', 'content': err_msg}
            ]

            assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.95, max_tokens=self.max_tokens)

            cost = (
                self.llm_cost["input_cost_per_token"] * prompt_tokens +
                self.llm_cost["output_cost_per_token"] * completion_tokens
            )

            early_stopping = self.write_program(assistant_output, out_fname)

            self.history += [
                {'role': 'user', 'content': err_msg},
                {'role': 'assistant', 'content': assistant_output}
            ]

            return early_stopping, cost

    def solve_task(self, task, out_fname):
        # Clean history
        self.history = []

        self.sys_msg = self.get_sys_msg(task)

        user_input = [
            {'role': 'user', 'content': self.sys_msg}
        ]

        assistant_output, prompt_tokens, completion_tokens = self.llm_engine.respond(user_input, temperature=0.2, top_p=0.95, max_tokens=self.max_tokens)

        self.history.append(
            {'role': 'assistant', 'content': assistant_output}
        )

        cost = (
            self.llm_cost["input_cost_per_token"] * prompt_tokens +
            self.llm_cost["output_cost_per_token"] * completion_tokens
        )

        if self.use_self_debug:
            self.write_program(assistant_output, out_fname)

            for t in range(10):
                halt, new_cost = self.step(out_fname, task["output_fname"])
                cost += new_cost
                if halt:
                    break

        self.history = [
            {'role': 'user', 'content': self.sys_msg}
        ] + self.history

        return {"history": self.history, "cost": cost}


if __name__ == "__main__":
    agent = ScienceAgent(
        "anthropic.claude-3-5-sonnet-20240620-v1:0", #"gpt-4o-mini-2024-07-18",
        context_cutoff=28000,
        use_self_debug=False,
        use_knowledge=False
    )

    task = {
        "task_inst": "Use the DKPES dataset to develop a Random Forest classifier predicting signal inhibition of chemicals while choosing appropriate threshold to assign binary labels based on signal inhibition values. Save the test set predictions, including the index and predicted signal inhibition, in \"pred_results/dkpes_test_pred.csv\".",
        "dataset_path": "benchmark/datasets/dkpes",
        "dataset_folder_tree": "|-- dkpes/\n|---- dkpes_test.csv\n|---- dkpes_train.csv",
        "dataset_preview": "[START Preview of dkpes/dkpes_train.csv]\nindex,Signal-inhibition,3-Keto,3-Hydroxy,12-Keto,12-Hydroxy,19-Methyl,18-Methyl,Sulfate-Ester,Sulfate-Oxygens,C4-C5-DB,C6-C7-DB,Sulfur,ShapeQuery,TanimotoCombo,ShapeTanimoto,ColorTanimoto,FitTverskyCombo,FitTversky,FitColorTversky,RefTverskyCombo,RefTversky,RefColorTversky,ScaledColor,ComboScore,ColorScore,Overlap\nZINC04026280,0.24,0,0,0,0,0,1,0,0,0,0,0,DKPES_CSD_MMMF_1_32,1.184,0.708,0.476,1.692,0.886,0.806,1.316,0.779,0.537,0.528,1.235,-5.804,1045.931\nZINC78224296,0.278,0,0,0,0,0,1,0,3,0,0,1,DKPES_CSD_MMMF_1_31,1.063,0.765,0.298,1.346,0.904,0.442,1.31,0.832,0.478,0.48,1.245,-5.278,1122.302\nZINC01532179,0.686,0,0,0,0,0,0,1,3,0,0,1,DKPES_CSD_MMMF_1_16,0.965,0.633,0.332,1.896,1.143,0.752,0.959,0.586,0.373,0.363,0.995,-3.988,770.823\n...\n[END Preview of dkpes/dkpes_train.csv]",
        "domain_knowledge": "The features related to signal inhibition are: '3-Keto', '3-Hydroxy', '12-Keto', 12-Hydroxy', '19-Methyl', '18-Methyl', 'Sulfate-Ester', 'Sulfate-Oxygens', 'C4-C5-DB', 'C6-C7-DB', 'Sulfur'. The decision boundary for signal inhibition is 0.6.",
        "low_level_plan": "1. Identify the scientific problem you want to solve and gather relevant datasets for training and testing.\n2. Load the training and testing datasets into your program.\n3. Select the relevant features from the datasets that will be used for model training.\n4. Preprocess the target variable to convert it into a suitable format for classification.\n5. Initialize a machine learning model appropriate for your task.\n6. Train the model using the training dataset.\n7. Use the trained model to make predictions on the testing dataset.\n8. Save the predictions to a CSV file for further analysis or submission.",
        "output_fname": "pred_results/dkpes_test_pred.csv"
    }

    trajectory = agent.solve_task(task, out_fname="pred_programs/pred_dkpes.py")
    print(trajectory)
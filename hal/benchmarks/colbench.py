import os
import json
from typing import Dict, Any, TypedDict, List
from typing_extensions import NotRequired
from .base_benchmark import BaseBenchmark
import docker
from hal.utils.logging_utils import print_warning
from sweet_rl.utils import code_evaluate


from PIL import Image
from sweet_rl.utils.webpage_utils import (extract_html_snippet, get_driver,
                                          render_full_html, replace_urls)
from torchvision.transforms import functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import concurrent
import torch

CODE_USER_PROMPT = """
Your task is to simulate a human user that interacts with an LLM agent in a dialogue.
You would like the LLM agent to help you with the following problem:
{problem_description}

Your goal is to engage in the conversation with the LLM agent so that it can get to a personalized answer.
You should make use of the following hidden information to answer the LLM agent.
YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.
You SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE HIDDEN INFORMATION, AND SAY YOU DON"T KNOW IF THE ANSWER CAN NOT BE FOUND IN THE HIDDEN INFORMATION.

{hidden_information}

Here is the dialogue so far:
{dialogue_history}


Now directly output your answer to the LLM agent IN TWO SENTENCES. DO NOT SAY ANYTHING ELSE.
"""

HTML_USER_PROMPT = """
Your task is to simulate a human user that interacts with an LLM agent in a dialogue.
Your goal is to engage in the conversation with the LLM agent so that it can get to a personalized answer.
YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.
The ultimate goal is to have the agent to construct the EXACT DESIGN that you have in mind.
You will be given an image made by the agent and a ground-truth image that the human user wants.
Describe briefly how is the image made by the agent is mainly different from the image that the human user wants. 
You should PRIORITIZE MOST OUTSTANDING DIFFERENCES. DESCRIBE CONCRETELY HOW EACH COMPONENT IS DIFFERENT (e.g. image has a larger size, text alignment should be in the center, etc)
1) The first image will be the agent provided image.
2) The second image will be the image that the human user wants
"""

#benchmark args is not actually passed to the benchmark
BACKEND_TASK_PATH = os.path.join(os.path.dirname(__file__), 'colbench/data/backend_test.jsonl')
FRONTEND_TASK_PATH = os.path.join(os.path.dirname(__file__), 'colbench/data/frontend_test.jsonl')
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'colbench/cache')

class ColBenchBenchmark(BaseBenchmark):
    """ColBench benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'colbench_backend_programming', 
                 num_tasks: int = 0):
        assert os.path.exists(os.path.join(os.path.dirname(__file__), 'colbench/data')), "data folder in Colbench directory (hal/benchmarks/colbench) not found. Please download and extract the USACO dataset as described in the README."
        self.benchmark_name = benchmark_name
        self.setup_script = 'hal/benchmarks/colbench/colbench_setup.sh'
        self.requires_sandbox = False
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
        # print("="*100)
        # print("task_path", task_path)
        # print("="*100)
        
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox, setup_script=self.setup_script)
    
        task_path = BACKEND_TASK_PATH if benchmark_name == 'colbench_backend_programming' else FRONTEND_TASK_PATH
        if num_tasks == 0:
            if benchmark_name == 'colbench_backend_programming':
                num_tasks = 1000
            else:
                num_tasks = 10
        with open(task_path, "r") as fb:
            tasks = [json.loads(line) for line in fb]
            tasks = tasks[:num_tasks] if benchmark_name == 'colbench_backend_programming' else tasks[:num_tasks]
        
        # Create benchmark dictionary
        self.benchmark = {}
        if benchmark_name == 'colbench_backend_programming':
            for task_index, task in enumerate(tasks):
                self.benchmark[str(task_index)] = {"problem_description": task["problem_description"], 
                                                   "hidden_information": task["hidden_information"], 
                                                   "test_cases": task["test_cases"],
                                                   "human_prompt": CODE_USER_PROMPT,
                                                   "task_type": "code"}
        elif benchmark_name == 'colbench_frontend_design':
            for task_index, task in enumerate(tasks):
                self.benchmark[str(task_index)] = {"problem_description": task["problem_description"], 
                                                   "hidden_information": task["ground_truth"],
                                                   "human_prompt": HTML_USER_PROMPT,
                                                   "task_type": "html"}
             

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs using AppWorld evaluation"""
        annotation_results = list(agent_output.values())
        if self.benchmark_name == 'colbench_backend_programming':
            return code_evaluate(annotation_results)
        evaluation_batch_size = min(20, len(annotation_results))
        answer_images = [a["answer"] for a in annotation_results]
        ground_truth_images = [a["task"]["ground_truth"] for a in annotation_results]
        drivers = []
        print("Getting drivers")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(get_driver) for i in range(evaluation_batch_size)]
            drivers = [job.result() for job in jobs]
        print("Rendering images")
        rendered_images = []
        for i in tqdm(range(0, len(annotation_results), evaluation_batch_size)):
            actual_drivers = drivers[:len(ground_truth_images[i:i+evaluation_batch_size])]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                jobs = [
                    executor.submit(
                        render_full_html,
                        driver,
                        ground_truth_images[i + j],
                        CACHE_PATH,
                        i + j,
                    )
                    for j, driver in enumerate(actual_drivers)
                ]
                rendered_images += [job.result() for job in jobs]
        for d in drivers:
            d.quit()
        ground_truth_images = [
            Image.open(ground_truth_image) for ground_truth_image in rendered_images
        ]
        answer_images = [
            (
                Image.open(answer_image).convert("RGB")
                if answer_image is not None and os.path.exists(answer_image)
                else Image.new("RGB", (224, 224), "black")
            )
            for answer_image in answer_images
        ]
        # import IPython; IPython.embed(); exit()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs1 = processor(images=answer_images, return_tensors="pt", padding=True).to(
            "cuda"
        )
        inputs2 = processor(
            images=ground_truth_images, return_tensors="pt", padding=True
        ).to("cuda")
        # Get the image embeddings
        with torch.no_grad():
            image_features1 = model.get_image_features(**inputs1)
            image_features2 = model.get_image_features(**inputs2)
        # Normalize the embeddings
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
        image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)
        # Calculate cosine similarity
        similarities = torch.sum(image_features1 * image_features2, dim=-1).cpu().numpy().tolist()
        
        #remove all files in cache folder
        for file in os.listdir(CACHE_PATH):
            os.remove(os.path.join(CACHE_PATH, file))
        return similarities
    
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.
        
        Args:
            eval_results: Dictionary containing evaluation results
            
        Returns:
            Dictionary with calculated metrics and task lists
        """
        if self.benchmark_name == 'colbench_backend_programming':
            results = {"average_correctness": sum(eval_results)/len(eval_results),
                       "accuracy": sum([1 for correctness in eval_results if correctness == 1])/len(eval_results)}
        else:
            results = {"average_correctness": sum(eval_results)/len(eval_results)}
        return results





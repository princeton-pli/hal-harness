import subprocess
from .base_benchmark import BaseBenchmark
import json
from datasets import load_dataset
from typing_extensions import NotRequired, TypedDict, List, Dict, Optional
from pydantic.config import ConfigDict
from pydantic import TypeAdapter, ValidationError
import time
import os
import sys
from ..utils.utils import move_merge_dirs
from ..utils.weave_utils import get_total_cost, assert_task_id_logging, get_weave_calls
from datetime import datetime

class SWEBenchBenchmark(BaseBenchmark):
    def __init__(self, agent_dir, config, dataset_name):
        super().__init__(agent_dir, config)
        self.benchmark_name = dataset_name
        self.dataset_name = 'princeton-nlp/SWE-bench_Lite' if dataset_name == "swebench_lite" else 'princeton-nlp/SWE-bench_Verified'
        self.max_workers = config.get('swe_bench_max_workers', 1)
        self.benchmark_dir = os.path.join(os.path.dirname(__file__), 'SWE-bench')
        self.requirements_file = 'swebench'
        

        self.benchmark = load_dataset(self.dataset_name, split='test').to_list()

        # shuffle the benchmark
        import random
        random.seed(99)
        random.shuffle(self.benchmark)

        self.benchmark = self.benchmark[:50]


    def run(self, agent_function, run_id: str) -> Dict:

        # Generate predictions using the agent function
        predictions = self._generate_predictions(agent_function, run_id)
        
        # Save predictions to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
            for pred in predictions:
                temp_file.write(json.dumps(pred) + '\n')
            predictions_path = temp_file.name

        # Run the SWE-bench evaluation harness
        self.mount_benchmark()
        result = self._run_evaluation_harness(predictions_path, run_id)
        self.unmount_benchmark()

        return result


    def _generate_predictions(self, agent_function, run_id) -> List[Dict]:
        self.mount_environment()
        agent_output = self.run_agent(agent_function, self.benchmark)
        self.unmount_environment()

        # Format prediction for SWE-bench
        predictions = [{'instance_id': task["instance_id"], 
                        'model_patch': task['model_patch'],
                        'model_name_or_path': run_id} for task in agent_output]

        return predictions

    def _run_evaluation_harness(self, predictions_path, run_id):

        command = (
            f"poetry run python -m swebench.harness.run_evaluation "
            f"--dataset_name {self.dataset_name} "
            f"--predictions_path {predictions_path} "
            f"--max_workers {self.max_workers} "
            f"--run_id {run_id}"
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_output = []
            stderr_output = []

            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    print(stdout_line.strip())
                    stdout_output.append(stdout_line)
                if stderr_line:
                    print(stderr_line.strip(), file=sys.stderr)
                    stderr_output.append(stderr_line)

                if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                    break

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command, 
                                                    output=''.join(stdout_output),
                                                    stderr=''.join(stderr_output))

            return self._parse_evaluation_result(run_id)


        except subprocess.CalledProcessError as e:
            print(f"Error running SWE-bench evaluation harness: {e}")
            print(f"Stdout: {e.output}")
            print(f"Stderr: {e.stderr}")
            raise


    def _parse_evaluation_result(self, run_id):
        # Load the evaluation results
        with open(f"{run_id}.{run_id}.json", 'r') as f:
            results = json.load(f)

        # delete file
        os.remove(f"{run_id}.{run_id}.json")

        return results

    def test_run(self, agent_function, weave_client):
        # Implement a simple test task for SWE-bench
        test_task = [{'repo': 'astropy/astropy', 'instance_id': 'astropy__astropy-12907', 'base_commit': 'd16bfe05a744909de4b27f5875fe0d4ed41ce607', 'patch': "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n@@ -242,7 +242,7 @@ def _cstack(left, right):\n         cright = _coord_matrix(right, 'right', noutp)\n     else:\n         cright = np.zeros((noutp, right.shape[1]))\n-        cright[-right.shape[0]:, -right.shape[1]:] = 1\n+        cright[-right.shape[0]:, -right.shape[1]:] = right\n \n     return np.hstack([cleft, cright])\n \n", 'test_patch': "diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py\n--- a/astropy/modeling/tests/test_separable.py\n+++ b/astropy/modeling/tests/test_separable.py\n@@ -28,6 +28,13 @@\n p1 = models.Polynomial1D(1, name='p1')\n \n \n+cm_4d_expected = (np.array([False, False, True, True]),\n+                  np.array([[True,  True,  False, False],\n+                            [True,  True,  False, False],\n+                            [False, False, True,  False],\n+                            [False, False, False, True]]))\n+\n+\n compound_models = {\n     'cm1': (map3 & sh1 | rot & sh1 | sh1 & sh2 & sh1,\n             (np.array([False, False, True]),\n@@ -52,7 +59,17 @@\n     'cm7': (map2 | p2 & sh1,\n             (np.array([False, True]),\n              np.array([[True, False], [False, True]]))\n-            )\n+            ),\n+    'cm8': (rot & (sh1 & sh2), cm_4d_expected),\n+    'cm9': (rot & sh1 & sh2, cm_4d_expected),\n+    'cm10': ((rot & sh1) & sh2, cm_4d_expected),\n+    'cm11': (rot & sh1 & (scl1 & scl2),\n+             (np.array([False, False, True, True, True]),\n+              np.array([[True,  True,  False, False, False],\n+                        [True,  True,  False, False, False],\n+                        [False, False, True,  False, False],\n+                        [False, False, False, True,  False],\n+                        [False, False, False, False, True]]))),\n }\n \n \n", 'problem_statement': "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n```python\r\nfrom astropy.modeling import models as m\r\nfrom astropy.modeling.separable import separability_matrix\r\n\r\ncm = m.Linear1D(10) & m.Linear1D(5)\r\n```\r\n\r\nIt's separability matrix as you might expect is a diagonal:\r\n\r\n```python\r\n>>> separability_matrix(cm)\r\narray([[ True, False],\r\n       [False,  True]])\r\n```\r\n\r\nIf I make the model more complex:\r\n```python\r\n>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))\r\narray([[ True,  True, False, False],\r\n       [ True,  True, False, False],\r\n       [False, False,  True, False],\r\n       [False, False, False,  True]])\r\n```\r\n\r\nThe output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.\r\n\r\nIf however, I nest these compound models:\r\n```python\r\n>>> separability_matrix(m.Pix2Sky_TAN() & cm)\r\narray([[ True,  True, False, False],\r\n       [ True,  True, False, False],\r\n       [False, False,  True,  True],\r\n       [False, False,  True,  True]])\r\n```\r\nSuddenly the inputs and outputs are no longer separable?\r\n\r\nThis feels like a bug to me, but I might be missing something?\n", 'hints_text': '', 'created_at': '2022-03-03T15:14:54Z', 'version': '4.3', 'FAIL_TO_PASS': '["astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]"]', 'PASS_TO_PASS': '["astropy/modeling/tests/test_separable.py::test_coord_matrix", "astropy/modeling/tests/test_separable.py::test_cdot", "astropy/modeling/tests/test_separable.py::test_cstack", "astropy/modeling/tests/test_separable.py::test_arith_oper", "astropy/modeling/tests/test_separable.py::test_separable[compound_model0-result0]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model1-result1]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model2-result2]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model3-result3]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model4-result4]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model5-result5]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model7-result7]", "astropy/modeling/tests/test_separable.py::test_separable[compound_model8-result8]", "astropy/modeling/tests/test_separable.py::test_custom_model_separable"]', 'environment_setup_commit': '298ccb478e6bf092953bca67a3d29dc6c35f6752'}]
        
        test_output = self.run_agent(agent_function, test_task)

        # Validate agent output
        self.validate_agent_output(test_output)

        # validate that there was cost associated with the test run
        time.sleep(5) # wait to finish usage calculation on weave
        self.validate_logging(weave_client, test_weave_task_id='astropy__astropy-12907')

        return True

    @property
    def type_adapter(self):
        class Task(TypedDict):
            model_config = ConfigDict(extra='allow')
            repo: str
            instance_id: str
            base_commit: str
            patch: str
            test_patch: str
            problem_statement: str
            hints_text: str
            created_at: str
            version: str
            FAIL_TO_PASS: str
            PASS_TO_PASS: str
            environment_setup_commit: str
            model_patch: str
            model_name_or_path: str
        return TypeAdapter(List[Task])


    def process_and_upload_results(self, 
                                   agent_name: str, 
                                   run_id: str, 
                                   eval_results: Dict, 
                                   weave_client,
                                   config, 
                                   upload=False):
        
        # move logs/ direcotry to results/benchmark_name/logs
        out_path = f"results/{self.benchmark_name}/{run_id}"
        os.makedirs(out_path, exist_ok=True)
        move_merge_dirs("logs/", f"results/{self.benchmark_name}/logs/")

        # store results
        with open(os.path.join(out_path, f"{run_id}.json"), 'w') as f:
            json.dump(eval_results, f)        


        total_cost = get_total_cost(weave_client)
        weave_calls = get_weave_calls(weave_client)

        # New dict
        try:
            upload_dict = {
                "config": {'agent_name': agent_name, 
                        'benchmark_name': self.benchmark_name, 
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'run_id': run_id,
                        **config[self.benchmark_name]},
                "results": {
                    "accuracy": eval_results['resolved_instances']/eval_results['total_instances'],
                    "total_cost": total_cost,
                    'successful_tasks': eval_results['resolved_ids'],
                    'failed_tasks': eval_results['unresolved_ids']
                },
                "raw_eval_results": eval_results,
                "raw_logging_results": weave_calls
            }
        except KeyError as e:
            upload_dict = {
                "config": {'agent_name': agent_name, 
                        'benchmark_name': self.benchmark_name, 
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'run_id': run_id,
                },
                "results": {
                    "accuracy": eval_results['resolved_instances']/eval_results['total_instances'],
                    "total_cost": total_cost,
                    'successful_tasks': eval_results['resolved_ids'],
                    'failed_tasks': eval_results['unresolved_ids']
                },
                "raw_eval_results": eval_results,
                "raw_logging_results": weave_calls
            }


        # Store the upload results locally
        with open(os.path.join(out_path, f"{run_id}_UPLOAD.json"), 'w') as f:
            json.dump(upload_dict, f)

        if upload:
            self.upload_results(run_id, upload_dict)


        # pretty print results_summary dict
        print("\n\n=====Results Summary=====")
        print(upload_dict['results']['accuracy'])
        print(upload_dict['results']['total_cost'])
        print('=====')


        return upload_dict['results']

        
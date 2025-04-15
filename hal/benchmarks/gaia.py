import os
import json
from typing import Dict, Any, TypedDict, List
from typing_extensions import NotRequired
from .base_benchmark import BaseBenchmark
import docker
from hal.utils.logging_utils import print_warning
from datasets import load_dataset
from .GAIA.scoring_utils import question_scorer

class GaiaBenchmark(BaseBenchmark):
    """Gaia benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'gaia'):
        self.benchmark_name = benchmark_name
        self.setup_script = None
        self.requires_sandbox = False
        super().__init__(agent_dir, config, requires_sandbox=self.requires_sandbox, setup_script=self.setup_script)
    
        
        # Create benchmark dictionary
        dataset = load_dataset('gaia-benchmark/GAIA', '2023_all', split='validation', trust_remote_code=True)
        
        # create benchmark dictionary
        self.benchmark = {}
        for idx, d in enumerate(dataset):
            self.benchmark[d['task_id']] = d
            if d['file_name'] != '':
                self.benchmark[d['task_id']]['files'] = {
                    d['file_name']: d['file_path']
                }      
            

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs using Gaia evaluation"""
        eval_results = {}
        for task_id, agent_answer in agent_output.items():
            gt_answer = self.benchmark[task_id]['Final answer']
            score, explanation = question_scorer(str(agent_answer), str(gt_answer))
            eval_results[task_id] = {'score': score, 'explanation': explanation}
            
        return eval_results

    
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.
        
        Args:
            eval_results: Dictionary containing evaluation results
            
        Returns:
            Dictionary with calculated metrics and task lists
        """
        
        # calculate accuracy for overall and for each level separately
        overall_correct_count = 0
        level_correct_count = {
            'level_1': 0,
            'level_2': 0,
            'level_3': 0
        }
        level_count = {
            'level_1': 0,
            'level_2': 0,
            'level_3': 0
        }
        
        for task_id, task_output in self.benchmark.items():
            if task_id in eval_results:
                # update the level count
                if str(self.benchmark[task_id]['Level']) == '1':
                    level_count['level_1'] += 1
                elif str(self.benchmark[task_id]['Level']) == '2':
                    level_count['level_2'] += 1
                elif str(self.benchmark[task_id]['Level']) == '3':
                    level_count['level_3'] += 1
                
                # if the agent got the task correct, update the level and overall correct counts
                if int(eval_results[task_id]['score']) > 0:
                    overall_correct_count += 1
                    if str(self.benchmark[task_id]['Level']) == '1':
                        level_correct_count['level_1'] += 1
                    elif str(self.benchmark[task_id]['Level']) == '2':
                        level_correct_count['level_2'] += 1
                    elif str(self.benchmark[task_id]['Level']) == '3':
                        level_correct_count['level_3'] += 1
            else:
                level_count[f'level_{str(self.benchmark[task_id]["Level"])}'] += 1

                
        successful_tasks = [task_id for task_id, task_output in eval_results.items() if int(task_output['score']) > 0]
        failed_tasks = [task_id for task_id, task_output in eval_results.items() if int(task_output['score']) == 0]
                
        results = {
            'overall_accuracy': overall_correct_count / len(eval_results),
            'level_1_accuracy': level_correct_count['level_1'] / level_count['level_1'] if level_count['level_1'] > 0 else None,
            'level_2_accuracy': level_correct_count['level_2'] / level_count['level_2'] if level_count['level_2'] > 0 else None,
            'level_3_accuracy': level_correct_count['level_3'] / level_count['level_3'] if level_count['level_3'] > 0 else None,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks
        }
        return results




import os
import json
from typing import Dict, Any, TypedDict, List, Optional
from typing_extensions import NotRequired
from .base_benchmark import BaseBenchmark
import docker
from hal.utils.logging_utils import print_warning

class TauBenchBenchmark(BaseBenchmark):
    """TauBench benchmark implementation"""
    
    def __init__(self, agent_dir: str, config: Dict[str, Any], benchmark_name: str = 'taubench_retail', agent_args: Optional[Dict] = None):
        self.benchmark_name = benchmark_name
        self.split = 'retail' if benchmark_name == 'taubench_retail' else 'airline'
        self.setup_script = 'hal/benchmarks/taubench/taubench_setup.sh'
        self.vm_only = False
        super().__init__(agent_dir, config, vm_only=self.vm_only, setup_script=self.setup_script)
        
        # default to GPT4o and OpenAI
        user_model = "gpt-4o"
        if agent_args:
            user_model = agent_args.get("model_name", user_model)
        user_provider = "openai"
        if agent_args:
            user_provider = agent_args.get("provider", user_provider)
        print(f"TauBenchBenchmark, user_model={user_model}, user_provider={user_provider}")
        # Create benchmark dictionary
        self.benchmark = {}
        if self.split == 'retail':
            self.benchmark = {str(task_index): {
                'env': 'retail',
                'user_strategy': 'llm',
                'user_model': user_model,
                'task_split': 'test',
                'user_provider': user_provider,
                'task_index': task_index,
            } for task_index in range(115)}
        elif self.split == 'airline':
            self.benchmark = {str(task_index): {
                'env': 'airline',
                'user_strategy': 'llm',
                'user_model': user_model,
                'task_split': 'test',
                'user_provider': user_provider,
                'task_index': task_index,
            } for task_index in range(50)}
      
            

    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent outputs using AppWorld evaluation"""
        return agent_output
    
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate metrics from evaluation results.
        
        Args:
            eval_results: Dictionary containing evaluation results
            
        Returns:
            Dictionary with calculated metrics and task lists
        """
        reward = 0
        for task_id, task_output in eval_results.items():
            reward += task_output['reward']
            
            
        number_of_tasks = len(self.benchmark)
            
            
        successful_tasks = [task_id for task_id, task_output in eval_results.items() if task_output['reward'] > 0]
        
        failed_tasks = [task_id for task_id, task_output in eval_results.items() if task_output['reward'] == 0]
        
        results = {'accuracy': reward/number_of_tasks, 
                   'successful_tasks': successful_tasks, 
                   'failed_tasks': failed_tasks
                   }
        return results



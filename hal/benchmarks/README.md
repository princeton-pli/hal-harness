# Adding New Benchmarks to HAL

This guide explains how to add new benchmarks to the HAL evaluation framework. A benchmark is a collection of tasks and evaluation logic.

## Benchmark Structure

Each benchmark should be a Python class inheriting from `BaseBenchmark` with this structure:

```
hal/benchmarks/
  ├── your_benchmark.py
  └── your_benchmark/
      ├── __init__.py
      ├── setup.sh        # Optional setup script for VMs
      └── data/          # Optional benchmark data files
```

## Benchmark Interface

Your benchmark must implement this interface:

```python
from typing import Dict, Any, Optional
from ..base_benchmark import BaseBenchmark

class YourBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        super().__init__(agent_dir, config)
        self.benchmark_name = "your_benchmark"
        # Benchmark dataset
        self.benchmark: Dict[str, Any]

        # Optional: Set if benchmark requires VM execution
        self.vm_only = False
        # Optional: Path to VM setup script
        self.setup_script = "hal/benchmarks/your_benchmark/setup.sh"
        
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Evaluate agent solutions"""
        pass
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from evaluation results"""
        pass
```

## Example Benchmark

Here's a complete example benchmark:

```python
import json
import os
from typing import Dict, Any
from ..base_benchmark import BaseBenchmark

class SimpleMathBenchmark(BaseBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        super().__init__(agent_dir, config)
        self.benchmark_name = "simple_math"
        self.benchmark = {
            "task1": {
                "prompt": "What is 2 + 2?",
                "answer": 4
            },
            "task2": {
                "prompt": "What is 3 * 4?",
                "answer": 12
            }
        }
        
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Run evaluation harness. This can score based on the agent's output, or by running an evaluation script on the entire environments returned by the agent (see AppWorld benchmark)."""
        results = {}
        dataset = self.get_dataset()
        
        for task_id, solution in agent_output.items():
            try:
                # Parse agent's answer
                answer = int(solution.strip())
                # Compare with correct answer
                correct = answer == dataset[task_id]["answer"]
                results[task_id] = {
                    "correct": correct,
                    "expected": dataset[task_id]["answer"],
                    "received": answer
                }
            except ValueError:
                results[task_id] = {
                    "correct": False,
                    "error": "Invalid number format"
                }
                
        return results
        
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy, successful tasks, and failed tasks IDs"""
        correct = sum(1 for r in eval_results.values() if r.get("correct", False))
        total = len(eval_results)
        
        return {
            "accuracy": correct / total,
            "successful_tasks": {
                task_id for task_id, result in eval_results.items()
                if result.get("correct", False)
            },
            "failed_tasks": {
                task_id for task_id, result in eval_results.items()
                if not result.get("correct", False)
            }
        }
```

## Key Features

1. **Evaluation**: `evaluate_output()` receives agent solutions and returns detailed results.

2. **Metrics**: `get_metrics()` calculates final metrics. This should always return a dictionary with at least the keys `accuracy`, `successful_tasks`, and `failed_tasks`.

3. **VM Support**: Set `vm_only = True` if benchmark requires VM execution.

4. **Setup Script**: Provide `setup_script` for installing benchmark-specific dependencies on VMs.

## Providing task-specific files to agents

Benchmarks can provide files to agents by including a `files` dictionary in each task. These files will be automatically copied into the agent's working environment by both the VM and local runs.

Example task with files:
```python
task = {
    "task_id_123": {
        "prompt": "Fix the bug in the sorting function in main.py",
        "files": {
            # Keys: destination paths in agent environment
            # Values: source paths in benchmark directory
            "/root/workspace/main.py": "/path/to/benchmark/data/task123/main.py",
            "/root/workspace/test.py": "/path/to/benchmark/data/task123/test.py",
            "/root/data/sample_input.txt": "/path/to/benchmark/data/task123/input.txt"
        }
    }
}
```

Key points about file handling:

1. **Path Mapping**: 
   - Keys in the `files` dict specify where files should be placed in the agent's environment
   - Values should be absolute paths to the source files

2. **Directory Structure**:
   - Source files are typically organized under your benchmark's data directory
   - Files can be placed anywhere in the agent's environment

3. **Environment Setup**:
   - Files are copied before the agent runs
   - Directory structure is automatically created
   - Works identically for both VM and local execution

4. **Common Patterns**:
   ```python
   # For code-focused tasks (like SWE-bench):
   "files": {
       "/root/main.py": "/absolute/path/to/source.py",
       "/root/test.py": "/absolute/path/to/test.py"
   }
   
   # For environment-based tasks (like AppWorld):
   "files": {
       "/root/environment/config.json": "/absolute/path/to/env_config.json",
       "/root/environment/assets/": "/absolute/path/to/assets/"  # Directory copy
   }
   ```

## Advanced Features

1. **Custom Evaluation Logic Using complete Agent Environments**:
   ```python
   def evaluate_output(self, agent_output, run_id):
       # Save submissions
       submissions_path = os.path.join(self.get_run_dir(run_id), f"{run_id}_submissions.json")
       with open(submissions_path, 'w') as f:
           json.dump(agent_output, f)
           
       # Run external evaluation tool
       results = subprocess.run(["./evaluate.sh", submissions_path], capture_output=True)
       return json.loads(results.stdout)
   ```

## Registering Your Benchmark

Add your benchmark to `benchmark_manager.py`:

```python
def get_benchmark(self, benchmark_name: str) -> BaseBenchmark:
    if benchmark_name == "your_benchmark":
        from .benchmarks.your_benchmark import YourBenchmark
        return YourBenchmark(self.agent_dir, self.config)
    # ...
```

## Testing Your Benchmark

1. Create a test agent:
```python
def test_agent(input: Dict[str, Any], **kwargs) -> Dict[str, str]:
    return {
        task_id: "42" for task_id in input.keys()
    }
```

2. Test locally:
```python
benchmark = YourBenchmark("agents/test", {})
dataset = benchmark.get_dataset()
output = test_agent(dataset)
results = benchmark.evaluate_output(output, "test_run")
metrics = benchmark.get_metrics(results)
print(metrics)
```

3. Test with HAL:
```bash
hal run --agent_name "test" --agent_function "test_agent.run" --benchmark "your_benchmark"
```

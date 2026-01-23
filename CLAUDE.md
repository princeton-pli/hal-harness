# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

HAL (Holistic Agent Leaderboard) is a standardized evaluation harness for reproducible agent evaluations across various benchmarks. It integrates with Weave for logging/cost tracking and the official HAL leaderboard for sharing results.

## Core Commands

### Setup and Installation

```bash
# Clone repository (recursive for submodules)
git clone --recursive https://github.com/benediktstroebl/hal-harness.git
cd hal-harness

# Create and activate conda environment (Python 3.11+)
conda create -n hal python=3.12
conda activate hal

# Install base package
pip install -e .

# Install optional dependencies as needed
pip install -e .[swebench]      # For SWE-bench
pip install -e .[azure]         # For Azure VM support
pip install -e .[taubench]      # For tau-bench
pip install -e .[scicode]       # For SciCode
pip install -e .[corebench]     # For CORE-bench
pip install -e .[assistantbench] # For AssistantBench

# Setup environment variables (copy template and edit)
cp .env.template .env
# Edit .env with API keys: OPENAI_API_KEY, WANDB_API_KEY, HF_TOKEN, AZURE_* (for VM), etc.
```

### Running Evaluations

```bash
# Basic evaluation command structure
hal-eval --benchmark <benchmark_name> \
  --agent_dir <agent_directory> \
  --agent_function <agent_function> \
  --agent_name <agent_name> \
  [OPTIONS]

# Example: SWE-bench evaluation
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/swebench_example_agent/ \
  --agent_function main.run \
  --agent_name "My Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=gpt-4o-mini-2024-07-18 \
  --max_concurrent 5

# Example: USACO with Docker isolation
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Solver (gpt-4o-mini)" \
  --docker \
  --max_concurrent 5 \
  -A model_name=gpt-4o-mini-2024-07-18

# Example: With Azure VM execution
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Solver (gpt-4o)" \
  --vm \
  --max_concurrent 5 \
  -A model_name=gpt-4o-2024-11-20
```

### Result Management

```bash
# Upload results during evaluation
hal-eval ... --upload

# Upload results after evaluation
hal-upload -B <benchmark_name>           # Upload all results for benchmark
hal-upload -F path/to/results.json       # Upload specific file
hal-upload -D path/to/directory          # Upload all files in directory

# Decrypt downloaded traces from leaderboard
hal-decrypt -D path/to/directory         # Decrypt entire directory
hal-decrypt -F trace_file.zip           # Decrypt single file
```

## Architecture

### Core Components

- **`hal/cli.py`**: Main CLI entry point (`hal-eval` command)
- **`hal/agent_runner.py`**: Orchestrates agent execution across benchmarks
- **`hal/benchmark_manager.py`**: Factory for benchmark instances
- **`hal/benchmarks/`**: Individual benchmark implementations
- **`hal/utils/`**: Execution runners (local, VM, Docker) and utilities

### Execution Modes

1. **Local**: Direct execution in specified conda environment (`--conda_env_name`)
2. **Docker**: Isolated execution in containers (`--docker`) - 4GB memory, 2 CPU cores limit
3. **Azure VM**: Cloud execution (`--vm`) with automatic VM provisioning

### Benchmark Integration

- Each benchmark inherits from `BaseBenchmark`
- Benchmarks define dataset loading, evaluation logic, and metrics calculation
- Support for task-specific files, GPU requirements, and custom setup scripts
- Results stored in `results/<benchmark>/<run_id>/` structure

### Agent Interface

Agents implement a `run` function with this signature:

```python
def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Additional arguments passed via -A flags
    Returns:
        Dictionary mapping task IDs to solutions/responses
    """
```

## Supported Benchmarks

### Core Benchmarks

- **SWE-bench**: Code generation/bug fixing
  - `swebench_verified`: Full dataset
  - `swebench_verified_mini`: 50 randomly selected problems
- **USACO**: Programming competition problems (`usaco`)
- **AppWorld**: Interactive coding agents
  - `appworld_test_normal`: Normal test suite
  - `appworld_test_challenge`: Challenge test suite

### Specialized Benchmarks

- **CORE-bench**: Scientific reproducibility
  - `corebench_easy`, `corebench_medium`, `corebench_hard`
- **tau-bench**: Tool-Agent-User interaction
  - `taubench_retail`, `taubench_airline`
- **SciCode**: Scientific programming
  - `scicode`, `scicode_easy`, `scicode_hard`
- **ScienceAgentBench**: Data-driven discovery (`scienceagentbench`)
- **AssistantBench**: Web search tasks (`assistantbench`)
- **CollaborativeAgentBench**: Human-agent collaboration
  - `colbench_backend_programming`, `colbench_frontend_design`

## Agent Development

### Directory Structure

```
agents/your_agent_name/
├── main.py          # Contains run() function
├── requirements.txt # Python dependencies
└── (other files)    # Additional agent files
```

### Key Requirements

- Implement `run()` function with specified signature
- Include all dependencies in `requirements.txt`
- Compatible with `weave==0.51.41` for automatic cost tracking
- Avoid spawning processes that bypass Weave logging
- For VM execution, dependencies are installed automatically
- For local execution, install dependencies manually or specify conda environment

### Example Agent Structure

```python
from openai import OpenAI

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    assert 'model_name' in kwargs, 'model_name is required'

    client = OpenAI()
    results = {}

    for task_id, task in input.items():
        # Process each task
        response = client.chat.completions.create(
            model=kwargs['model_name'],
            messages=[{"role": "user", "content": task['problem_statement']}],
            max_tokens=2000
        )
        results[task_id] = response.choices[0].message.content

    return results
```

## Common CLI Options

- `-A key=value`: Agent arguments (e.g., `-A model_name=gpt-4o`)
- `-B key=value`: Benchmark arguments
- `--max_concurrent N`: Parallel task execution (default: 1)
- `--continue_run`: Resume previous run (requires `--run_id`)
- `--max_tasks N`: Limit tasks for testing
- `--conda_env_name`: Specify conda environment for local execution
- `--upload`: Upload results to HuggingFace Hub after evaluation

## Important Notes

### Benchmark-Specific Requirements

- **SWE-bench**: Requires Docker, does not support arm64 machines
- **USACO**: Requires Docker for evaluation
- **AppWorld**: Must run with `--vm` flag, requires special setup
- **Cybench**: Requires Docker configuration, does not support arm64
- **CORE-bench**: Requires decryption of test file with password "reproducibility"

### Development Practices

- Results automatically logged to Weave with cost tracking
- GPU support available for VM execution (tasks marked with `"gpu": true`)
- File provisioning system copies benchmark files to agent working directory
- Encryption/decryption system prevents benchmark contamination
- Agent naming should follow format: `Name (model1, model2)` with exact model versions
- No formal test suite - validation through example agents and manual testing

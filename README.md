# HAL Harness

This repository provides a standardized evaluation harness for evaluating different AI agents across various benchmarks. It supports several benchmarks and allows users to easily add new agents and benchmarks.  The harness integrates with [Weave](https://wandb.ai/site/weave/) for logging and cost tracking, and the official [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) for storing and sharing evaluation results.

## Table of Contents

1. [Setup](#setup)
2. [Running Evaluations](#running-evaluations)
3. [Adding New Agents](#adding-new-agents)
    - [SWE-bench Verified](#swe-bench-benchmark)
    - [USACO](#usaco-benchmark)
    - [Inspect AI Benchmarks](#inspect-ai-benchmarks)
        - [Custom Inspect Agents](#custom-inspect-agents)
        - [Custom External Agents](#custom-external-agents)
4. [Uploading Results](#uploading-results)
5. [Repository Structure](#repository-structure)


## Setup

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/benediktstroebl/agent-eval-harness.git
   cd agent-eval-harness
   ```

2. **Install Poetry (if you don't have it):**
   ```bash
   pip install poetry
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Create a `.env` file:**
   ```bash
   cp .env.template .env
   ```
   Add your API keys (HuggingFace, Weave, OpenAI/other LLMs as needed) to the `.env` file.  See `.env.template` for details.


## Running Evaluations

The harness uses a command-line interface (CLI) to run evaluations.  The basic command structure is:

```bash
agent-eval --benchmark <benchmark_name> --agent_dir <agent_directory> --agent_function <agent_function> --agent_name <agent_name> [OPTIONS]
```

*   **`--benchmark <benchmark_name>`**: The name of the benchmark to run (e.g., `swebench_verified`, `usaco`, `mlagentbench`, `inspect_evals/gaia`).  For Inspect AI benchmarks, provide the path to the task.  
*   **`--agent_dir <agent_directory>`**: Path to the directory containing your agent's code.
*   **`--agent_function <agent_function>`**:  The name of the agent's main function, including the module path if applicable (e.g., `agent.run`, `my_agent_module.main`).
*   **`--agent_name <agent_name>`**: A descriptive name for your agent.  This will be used for logging and on the leaderboard.
*   **`-A <key>=<value>`**: Agent arguments. These are passed as keyword arguments to your agent function.
*   **`-B <key>=<value>`**: Benchmark arguments.  These are passed to the benchmark itself.  For Inspect AI tasks, these are passed as `task_args`.
*   **`--upload`**: If present, uploads results to the HuggingFace Hub.
*   **`--model <model_name>`**:  Specifies the LLM model to use (e.g., `gpt-4o-mini`).  Defaults to `gpt-4o-mini`. For inspect runs, you will need to specify the model name in the inspect format (e.g. `openai/gpt-4`).
*   **`--run_id <run_id>`**:  Provides a specific run ID. Only use this if you want to resume a run from an existing one.
*   **`--max_concurrent <number>`**: Run multiple samples in parallel for Inspect AI and other benchmarks. Defaults to 10.
*   **`--conda_env_name <conda_env>`**:  Specifies the name of the conda environment to run the agent in.


**Example:**

```bash
agent-eval --benchmark swebench_verified_mini --agent_dir agents/swebench_example_agent/ --agent_function main.run --agent_name "My SWE-bench Agent (gpt-4o-2024-08-06)" -A model_name=gpt-4o-mini --upload 
```


## Adding New Agents

### SWE-bench Benchmark

1.  **Create a directory** e.g. in the `agents/` directory for your agent.
2.  **Implement the agent function**  (see example in `agents/swebench_example_agent/agent.py`).  The function should take a dictionary of tasks as input and return a dictionary of results.
3.  Ensure your agent function is importable.

### USACO Benchmark

1.  **Create a directory** in the `agents/` directory.
2.  **Implement the agent function** (see example in `agents/usaco_example_agent/agent.py`).  The function should take a dictionary of tasks and return a dictionary of results.
3.  Ensure your agent function is importable.

### Inspect AI Benchmarks

The agent harness can run any Inspect AI task.  The simplest approach is to use a benchmark from the `inspect_evals` package.  Install `inspect_ai` and `inspect_evals` first:

```bash
pip install inspect_ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
```

There are two types of agents you can run with any inspect task. Inspect solver agents and custom external agents that can follow any implementation. More details and examples can be found in `agent_eval_harness/inspect/README.md`.

#### Custom Inspect Agents

You can provide a custom solver (agent) for an Inspect AI task.  Your `agent_function` should point to a function decorated with `@solver` from `inspect_ai`.  Example:

```bash
agent-eval --agent_name MyInspectAgent --benchmark inspect_evals/gaia --model openai/gpt-4o --agent_dir agents/my_inspect_agent --agent_function my_solver.my_agent 
```

See `agents/inspect/solver_agent.py` for an example solver.

#### Custom External Agents

You can use a completely external agent.  In this case, the Inspect AI task defines the dataset and scoring. The `agent_function` should take a dictionary of samples and return a dictionary of completions. Example:

```bash
agent-eval --agent_name MyExternalAgent --benchmark inspect_evals/gaia --model openai/gpt-4o --agent_dir agents/my_external_agent --agent_function my_agent.run -A model_name=gpt-4o
```

See `agents/inspect/custom_agent.py` for an example.


## Uploading Results

To upload results to HuggingFace, use the `--upload` flag when running an evaluation.  You can also upload previously generated results using the `agent-upload` command:

```bash
agent-upload --benchmark <benchmark_name>
```

This command searches for `_UPLOAD.json` files in the `results/<benchmark_name>` directory and uploads them to the HuggingFace Hub.


## Repository Structure

*   `agent_eval_harness`:  Contains the core harness code.
    *   `benchmarks`: Implementations of the different benchmarks.
    *   `inspect`:  Code specific to running Inspect AI benchmarks.
    *   `utils`: Utility functions for logging, configuration, etc.
*   `agents`: Directory for user-defined agent implementations.

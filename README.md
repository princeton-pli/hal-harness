# HAL Harness

[![HAL Leaderboard](https://img.shields.io/badge/🤗_HAL_Leaderboard-View_Results-blue)](https://agent-evals-leaderboard.hf.space)
[![Weave](https://img.shields.io/badge/W&B-Weave-orange)](https://wandb.ai/site/weave)
[![Inspect AI](https://img.shields.io/badge/Inspect_AI-green)](https://github.com/UKGovernmentBEIS/inspect_ai)

This repository provides a standardized evaluation harness for evaluating different AI agents across various benchmarks. It supports several benchmarks and allows users to easily add new agents and benchmarks. Key highlight is the unified agent-eval CLI across all benchmarks and agent types. The harness integrates with [Weave](https://wandb.ai/site/weave/) for logging and cost tracking, and the official [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) for sharing evaluation results.

## Features

* **Unified agent-eval CLI across all benchmarks and agent types**
  - Already supportsSWE-bench Verified, USACO, AppWorld, Inspect AI benchmarks (Gaia, Cybench)
  - Easy integration of new benchmarks
  - See [Running Evaluations](#running-evaluations) for details on CLI usage

* **Flexible Execution Environments**
  - Local execution with conda environment isolation
  - Azure VM support for isolated and resource-intensive tasks
  - Docker container support for AppWorld benchmark
  - Parallel task execution with configurable concurrency

* **Comprehensive Logging and Monitoring**
  - Integration with [Weave](https://wandb.ai/site/weave/) for detailed cost tracking and usage metrics
  - Automatic logging of agent traces and execution details

* **Fully flexible agent support**
  - Support for both custom agents and Inspect AI solvers
  - Flexible agent configuration through command-line arguments
  - Environment isolation for each agent run
  - Automatic dependency management

* **HAL Leaderboard Integration**
  - Direct integration with the [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space)
  - Detailed metrics and performance analysis

* **Developer-Friendly Design**
  - Clear command-line interface for all benchmark and agent types
  - Modular architecture for easy extension

## Table of Contents

1. [Setup](#setup)
2. [Running Evaluations](#running-evaluations)
3. [Supported Benchmarks](#supported-benchmarks)
    - [SWE-bench Verified](#swe-bench-benchmark)
    - [USACO](#usaco-benchmark)
    - [MLAgentBench](#mlagentbench-benchmark)
    - [AppWorld](#appworld-benchmark)
    - [Inspect AI Benchmarks](#inspect-ai-benchmarks)
4. [Adding New Agents](#adding-new-agents)
5. [Running Environments](#running-environments)
    - [Local Execution](#local-execution)
    - [VM Execution](#vm-execution)
6. [Uploading Results](#uploading-results)
7. [Repository Structure](#repository-structure)

## Setup

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/benediktstroebl/agent-eval-harness.git
   cd agent-eval-harness
   ```

2. **Create conda environment:**
   ```bash
   conda create -n hal python=3.11
   conda activate hal
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

   **Note:** Some benchmarks require additional dependencies which can be installed using `pip install hal[benchmark_name]`. See [Supported Benchmarks](#supported-benchmarks) for details.

5. **Create a `.env` file:**
   ```bash
   cp .env.template .env
   ```
   Add your API keys (HuggingFace, Weave, OpenAI/other LLMs as needed) to the `.env` file. See `.env.template` for details.

6. **Install Model Provider Dependencies:**
   
   For **Inspect AI benchmarks**, you'll need to install the appropriate Python SDK for your chosen model provider:

   ```bash
   # For OpenAI models (gpt-4, gpt-3.5-turbo, etc.)
   pip install openai

   # For Anthropic models (claude-3, etc.)
   pip install anthropic
   ```

7. **Optional: Azure VM Setup**
   If you plan to use Azure VMs for evaluation, add the following to your `.env`:
   ```
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   AZURE_RESOURCE_GROUP_NAME=your_resource_group
   AZURE_LOCATION=your_location
   SSH_PUBLIC_KEY_PATH=/path/to/your/ssh/key.pub
   SSH_PRIVATE_KEY_PATH=/path/to/your/ssh/key
   NETWORK_SECURITY_GROUP_NAME=your_nsg_name
   ```

## Running Evaluations

The harness uses a command-line interface (CLI) to run evaluations. The basic command structure is:

```bash
hal-eval --benchmark <benchmark_name> --agent_dir <agent_directory> --agent_function <agent_function> --agent_name <agent_name> [OPTIONS]
```

### Core Options

*   **`--benchmark <benchmark_name>`**: The name of the benchmark to run. Supported benchmarks:
    - `swebench_verified` or `swebench_verified_mini`
    - `usaco`
    - `appworld`
    - `inspect_evals/<task_name>` (e.g., `inspect_evals/gaia`)
*   **`--agent_dir <agent_directory>`**: Path to the directory containing your agent's code
*   **`--agent_function <agent_function>`**: The name of the agent's main function (e.g., `agent.run` if `agent.py` in agent directory contains `def run(): ...`)
*   **`--agent_name <agent_name>`**: A descriptive name for your agent (used in logging/leaderboard) (e.g., `My Agent (gpt-4o)`)

### Additional Options

*   **`-A <key>=<value>`**: Agent arguments passed to your agent function
*   **`-B <key>=<value>`**: Benchmark arguments passed to the benchmark
*   **`-I <key>=<value>`**: Inspect-specific arguments (for Inspect AI benchmarks)
*   **`--upload`**: Upload results to HuggingFace Hub
*   **`--max_concurrent <number>`**: Number of parallel tasks (default: 1)
*   **`--conda_env_name <env_name>`**: Conda environment for agent execution
*   **`--vm`**: Run evaluation on Azure VMs
*   **`--run_id <run_id>`**: Specify a run ID (useful for continuing runs)
*   **`--continue_run`**: Continue from a previous run (requires run_id)

### Examples

1. **Running SWE-bench locally:**
```bash
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/swebench_example_agent/ \
  --agent_function main.run \
  --agent_name "My Agent (gpt-4o-mini)" \
  -A model_name=gpt-4o-mini \
  --max_concurrent 5
```

2. **Running USACO on Azure VM:**
```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Solver (gpt-4o)" \
  --vm \
  --max_concurrent 5 \
  -A model_name=gpt-4o
```

3. **Running Inspect AI benchmark:**
```bash
hal-eval --benchmark inspect_evals/gaia \
  --agent_dir agents/inspect/ \
  --agent_function gaia.default_agent \
  --agent_name "Gaia Agent (gpt-4o)" \
  -A model_name=gpt-4o \
  -I token_limit=4000 \
  -I model_args="{'temperature': 0.4}"
```

## Supported Benchmarks

### [SWE-bench Verified (Mini)](https://github.com/princeton-nlp/SWE-bench)
- Evaluates code generation and bug fixing capabilities
- Full dataset (`swebench_verified`) or mini version (`swebench_verified_mini`)
- Mini version is a subset of 50 randomly selected problems from the full dataset
- Supports both local and VM execution
- The task ids part of SWE-Bench Verified (Mini) can be found [here](https://github.com/benediktstroebl/agent-eval-harness/blob/7b231a952828022a43977f21acfd452adda5088c/agent_eval_harness/benchmarks/swebench_verified_mini_task_ids.txt)

### [USACO](https://github.com/princeton-nlp/USACO)
- Programming competition problems
- Requires additional dependencies (`pip install hal[usaco]`)
- Supports both local and VM execution

### [AppWorld](https://appworld.dev/)
- A Controllable World of Apps and People for Benchmarking Interactive Coding Agents
- **Requires VM execution** (`--vm` flag mandatory)

### [Inspect AI Benchmarks](https://github.com/UKGovernmentBEIS/inspect_ai)
- Supports a number of [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) agent tasks (`inspect_evals/<task_name>`)
- Two agent types supported:
  1. Inspect Solver agents (using `@solver` decorator)
  2. Custom external agents
- Inspect solvers are run locally by default with orchestration being done by inspect_ai. Custom agents are run using the harness and can be run either locally or on Azure VMs via the `--vm` flag.

#### [Gaia](https://arxiv.org/abs/2311.12983)
- General AI assistants benchmark
- More details on Inspect AI implementation [here](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/gaia)

#### [Cybench](https://arxiv.org/abs/2408.08926)
- Cybersecurity agent task
- **Does not support arm64 machines**
- More details on Inspect AI implementation [here](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/cybench)
- Additional **Docker Configuration** required for Cybench:

For Cybench, you'll need to configure Docker's default address pools to avoid IP address conflicts when running the harness. Follow these steps:

1. Edit or create the daemon.json file:
   ```bash
   sudo nano /etc/docker/daemon.json
   ```

2. Add or modify the default-address-pools configuration. For example:
   ```json
   {
     "default-address-pools": [
       {
         "base": "172.17.0.0/16",
         "size": 24
       },
       {
         "base": "172.18.0.0/16",
         "size": 24
       },
       {
         "base": "172.19.0.0/16",
         "size": 24
       }
     ]
   }
   ```

3. Save the file and restart the Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

Now the harness should be able to run Cybench.

## Running Environment Options

### Local Execution
- Default execution mode
- Uses conda environments and temporary directories for isolation 
- Parallel execution via `--max_concurrent`
- See above examples for usage

### VM Execution
- Uses Azure VMs for isolated execution
- Required for some benchmarks (e.g., AppWorld)
- Automatic VM provisioning and cleanup
- See above examples for usage

## Uploading Results

Results can be uploaded to the [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) in several ways. To avoid benchmark contamination, we automatically encrypt the results before uploading.

1. **During evaluation:**
```bash
hal-eval --benchmark <benchmark> ... --upload
```

2. **After evaluation:**
```bash
# Upload all results for a benchmark
hal-upload -B <benchmark_name>

# Upload a single file
hal-upload -F path/to/file.json

# Upload all files in a directory
hal-upload -D path/to/directory
```

Note: When using `-F` to upload a single file, the file must be a JSON file.

## Decrypting Results

You can decrypt evaluation results that were encrypted during upload using the `hal-decrypt` command. This also appplies to trace files downloaded from the leaderboard.

1. **Decrypt a single file:**
```bash
hal-decrypt -F path/to/file.zip
```

2. **Decrypt all zip files in a directory:**
```bash
hal-decrypt -D path/to/directory
```

## About HAL

coming soon...

## Repository Structure

*   `hal/`: Core harness code
    *   `benchmarks/`: Benchmark implementations
        - `swebench.py`: SWE-bench implementation
        - `usaco.py`: USACO implementation
        - `mlagentbench.py`: MLAgentBench implementation
        - `appworld.py`: AppWorld implementation
        - `inspect_benchmark.py`: Inspect AI benchmark support
    *   `utils/`: Utility functions
        - `local_runner.py`: Local execution support
        - `vm_runner.py`: Azure VM execution support
        - `weave_utils.py`: Weave logging utilities
    *   `inspect/`: Inspect AI specific code
*   `agents/`: Example agent implementations
*   `results/`: Evaluation results and logs
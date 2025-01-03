# HAL: The Holistic Agent Leaderboard for Reproducible Agent Evaluation

[![HAL Leaderboard](https://img.shields.io/badge/🤗_HAL_Leaderboard-View_Results-blue)](https://agent-evals-leaderboard.hf.space)
[![Weave](https://img.shields.io/badge/W&B-Weave-orange)](https://wandb.ai/site/weave)
[![Inspect AI](https://img.shields.io/badge/Inspect_AI-green)](https://github.com/UKGovernmentBEIS/inspect_ai)

This repository provides a standardized evaluation harness for evaluating different AI agents across benchmarks. It supports several benchmarks and allows users to add new agents and benchmarks. The unified CLI allows evaluation across all benchmarks and agents. The harness integrates with [Weave](https://wandb.ai/site/weave/) for logging and cost tracking, and the official [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) for sharing evaluation results.

## Features

* **Unified hal-eval CLI across all benchmarks and agent types**
  - HAL supports SWE-bench Verified, USACO, AppWorld, Inspect AI benchmarks (Gaia, Cybench), with support for more coming soon
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
4. [Example Evaluations](#example-evaluations)
5. [Running Environments](#running-environments)
    - [Local Execution](#local-execution)
    - [VM Execution](#vm-execution)
6. [Uploading Results](#uploading-results)
7. [Repository Structure](#repository-structure)

## Setup

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/benediktstroebl/hal-harness.git
   cd hal-harness
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
    - `swebench_verified`: Full SWE-bench Verified dataset
    - `swebench_verified_mini`: Mini version with 50 randomly selected problems
    - `usaco`: USACO programming competition problems
    - `appworld_test_normal`: AppWorld normal test suite
    - `appworld_test_challenge`: AppWorld challenge test suite
    - `inspect_evals/gaia`: Gaia general AI assistants benchmark
    - `inspect_evals/cybench`: Cybersecurity agent tasks
    - `inspect_evals/agentharm`: Agent harm evaluation benchmark
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

## Supported Benchmarks

### [SWE-bench Verified (Mini)](https://github.com/princeton-nlp/SWE-bench)
- Evaluates code generation and bug fixing capabilities
- Full dataset (`swebench_verified`) or mini version (`swebench_verified_mini`)
- Mini version is a subset of 50 randomly selected problems from the full dataset
- Supports both local and VM execution
- The task ids part of SWE-Bench Verified (Mini) can be found [here](https://github.com/benediktstroebl/agent-eval-harness/blob/7b231a952828022a43977f21acfd452adda5088c/agent_eval_harness/benchmarks/swebench_verified_mini_task_ids.txt)

### [USACO](https://github.com/princeton-nlp/USACO)
- Programming competition problems
- Supports both local and VM execution

For USACO, you will need to download and extract the USACO dataset. This can be done with the following steps:

1. Download the USACO dataset from [here](https://drive.google.com/file/d/1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi/view?usp=share_link)
2. Unzip the dataset and move the `data` directory to `hal/benchmarks/USACO/`. Hence there should be a `data/` directory in `hal/benchmarks/USACO/`

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

## Example Evaluations

1. **Running SWE-bench locally:**
    ```bash
    hal-eval --benchmark swebench_verified_mini \
      --agent_dir agents/swebench_example_agent/ \
      --agent_function main.run \
      --agent_name "My Agent (gpt-4o-mini)" \
      -A model_name=gpt-4o-mini \
      --max_concurrent 5
    ```

1. **Running USACO on Amazon EC2 using models available via Amazon Bedrock**

    - Create an Amazon EC2 VM with `t3.2xlarge` instance type and the latest Ubuntu AMI (see [this](https://aws-samples.github.io/foundation-model-benchmarking-tool/misc/ec2_instance_creation_steps.html) for general guidance on creating EC2 VMs).
    - Make sure that the IAM role associated with the VM has `AmazonBedrockFullAccess` permissions.
    - Follow steps in the [Setup](#setup) section.
        - Install `Docker` on this VM. 
            ```{.bashrc}
            sudo apt-get update
            sudo apt-get install --reinstall docker.io -y
            sudo apt-get install -y docker-compose
            docker compose version
            ```
    - Copy the test data `usaco_v3` dataset in the `hal/benchmarks/USACO/data/datasets/usaco_v3` folder. The dataset runs into several GBs and is available for direct download as a Zip archive [here](https://drive.google.com/file/d/1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi/view?usp=share_link). You can download the dataset Zip archive and then find the `usaco_v3` folder in the extracted contents (also see the official [`USACO README`](https://github.com/princeton-nlp/USACO/blob/main/README.md#data)).
    - Run the following command to run the benchmarking. The command shown below runs the USACO benchmark for the `Anthropic Claude 3.5 Sonnet` model. _Depending upon your service quota limits this may take anywhere from 30 minutes to several hours_.

      ```{.bashrc}
      BENCHMARK_NAME=usaco
      AGENT_DIR=agents/usaco_example_agent/
      AGENT_FUNCTION=main.run
      MODEL_NAME=bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
      AGENT_NAME="USACO_${MODEL_NAME}"
      PROMPT_TEMPLATE_PATH=agents/usaco_example_agent/prompt_templates/claude_prompt_template.txt
      # adjust the concurrency based on your service quota, higher concurrency could lead to rate limiting
      CONCURRENCY=10
      hal-eval --benchmark $BENCHMARK_NAME\
          --agent_dir $AGENT_DIR\
          --agent_function $AGENT_FUNCTION \
          --agent_name $AGENT_NAME \
          -A model_name=$MODEL_NAME \
          -A prompt_template_path=$PROMPT_TEMPLATE_PATH \
          --max_concurrent $CONCURRENCY
      ```
    Use the model ids listed in the table below for the `MODEL_NAME` variable to try out other foundation models available via Amazon Bedrock.
    | Model Name                  | Model ID                                             |
    |-----------------------------|-----------------------------------------------------|
    | Anthropic Claude 3.5 Haiku  | bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0  |
    | Anthropic Claude 3.5 Sonnet | bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0 |
    | Anthropic Claude 3 Sonnet   | bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0   |
    | Amazon Nova Pro             | bedrock/amazon.nova-pro-v1:0                         |
    | Amazon Nova Lite            | bedrock/amazon.nova-lite-v1:0                        |
    | Amazon Nova Micro           | bedrock/amazon.nova-micro-v1:0                       |


1. **Running USACO on Azure VM:**

    ```bash
    hal-eval --benchmark usaco \
      --agent_dir agents/usaco_example_agent/ \
      --agent_function main.run \
      --agent_name "USACO Solver (gpt-4o)" \
      --vm \
      --max_concurrent 5 \
      -A model_name=gpt-4o
    ```

1. **Running Inspect AI benchmark:**

    ```bash
    hal-eval --benchmark inspect_evals/gaia \
      --agent_dir agents/inspect/ \
      --agent_function gaia.default_agent \
      --agent_name "Gaia Agent (gpt-4o)" \
      -A model_name=gpt-4o \
      -I token_limit=4000 \
      -I temperature=0.4
    ```

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

# HAL: The Holistic Agent Leaderboard

[![HAL Leaderboards](https://img.shields.io/badge/Leaderboards-HAL-blue)](https://hal.cs.princeton.edu/)
[![Weave](https://img.shields.io/badge/W&B-Weave-orange)](https://wandb.ai/site/weave)
[![Inspect AI](https://img.shields.io/badge/Inspect_AI-green)](https://github.com/UKGovernmentBEIS/inspect_ai)

This repository provides a standardized evaluation harness for reproducible agent evaluations across various benchmarks. It supports several benchmarks and allows users to add new agents and benchmarks. The unified CLI allows evaluation across all benchmarks and agents. The harness integrates with [Weave](https://wandb.ai/site/weave/) for logging and cost tracking, [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), and the official [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) for sharing evaluation results.

## Features

* **Unified `hal-eval` CLI across all benchmarks and agent types**
  - HAL supports SWE-bench Verified, USACO, AppWorld, CORE-bench, AgentHarm, GAIA, Cybench, tau-bench, with support for more coming soon
  - Run your and existing agents on HAL with the same CLI across benchmarks (see [How Do I Run Evaluations?](#how-do-i-run-evaluations))

* **Evaluations locally or in the cloud AND fully parallelized**
  - Local execution with conda environment isolation
  - Docker container support for isolated local execution
  - Azure VM support for running evaluations in the cloud
  - Configurable concurrency for parallel evaluation

* **Automatic logging and monitoring**
  - Integration with [Weave](https://wandb.ai/site/weave/) for detailed cost tracking and usage metrics
  - Automatic logging of agent traces

* **No constraints on agent implementation or agent framework**
  - No constraints on specific agent implementation or agent framework (see [How Do I Develop My Own Agents?](#how-do-i-develop-my-own-agents))
  - Support for both custom agents and Inspect AI solvers
  - Flexible agent configuration through command-line arguments in `hal-eval`

* **Share and access agent traces**
  - Simple upload of agent traces to HuggingFace Hub via `hal-upload` CLI (see [How Can I Submit My Results to the HAL Leaderboards?](#how-can-i-submit-my-results-to-the-hal-leaderboards))
  - Automatic encryption of agent traces before uploading to avoid benchmark contamination

* **HAL leaderboard integration**
  - Direct integration with the [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space)
  - Detailed metrics and in-depth performance analysis

## Table of Contents

1. [Setup](#setup)
2. [Which Benchmarks Are Supported?](#which-benchmarks-are-supported)
3. [How Do I Run Evaluations? (With Examples)](#how-do-i-run-evaluations)
4. [How Do I Develop My Own Agents?](#how-do-i-develop-my-own-agents)
5. [How to Reproduce Existing Agents on HAL?](#how-to-reproduce-existing-agents-on-hal)
5. [How Do I Add a Benchmark?](#how-do-i-add-a-benchmark)
6. [How Can I Submit My Results to the HAL Leaderboards?](#how-can-i-submit-my-results-to-the-hal-leaderboards)
7. [How Can I Use the Agent Traces from the HAL Leaderboard?](#how-can-i-use-the-agent-traces-from-the-hal-leaderboard)
8. [About](#about-hal)
9. [Repository Structure](#repository-structure)
10. [Citing HAL](#citing-hal)
## Setup

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/benediktstroebl/hal-harness.git
   cd hal-harness
   ```

2. **Create conda environment:**
   ```bash
   conda create -n hal python=3.12
   conda activate hal
   ```

3. **Install the `hal` package:**
   ```bash
   pip install -e .
   ```

4. **Create a `.env` file:**
   ```bash
   cp .env.template .env
   ```
   Add your API keys (HuggingFace, Weave, OpenAI/other LLMs as needed) to the `.env` file. See `.env.template` for details.

5. **Install Model Provider Dependencies:**
   
   You'll need to install the appropriate Python SDK for your chosen model provider:

   ```bash
   # For OpenAI models
   pip install openai

   # For Anthropic models
   pip install anthropic
   ```

6. **Optional: Azure VM Setup**
   If you plan to use Azure VMs for evaluation, add the following to your `.env`:
   ```
   AZURE_SUBSCRIPTION_ID=your_subscription_id
   AZURE_RESOURCE_GROUP_NAME=your_resource_group
   AZURE_LOCATION=your_location
   SSH_PUBLIC_KEY_PATH=/path/to/your/ssh/key.pub
   SSH_PRIVATE_KEY_PATH=/path/to/your/ssh/key
   NETWORK_SECURITY_GROUP_NAME=your_nsg_name
   ```
   
   Then run the following command to install the optional azure dependencies:
   ```bash
   pip install -e .[azure]
   ```
   
7. **Optional: Docker Setup**
   If you plan to use Docker containers for isolated evaluation, make sure Docker is installed on your system. The harness will automatically build the required Docker image.
   

## Which Benchmarks Are Supported?

### [SWE-bench Verified (Mini)](https://github.com/princeton-nlp/SWE-bench)
- Evaluates code generation and bug fixing capabilities
- Full dataset (`swebench_verified`) or mini version (`swebench_verified_mini`)
- Mini version is a subset of 50 randomly selected problems from the full dataset
- Supports local, Docker, and VM execution
- The task ids part of SWE-Bench Verified (Mini) can be found [here](https://github.com/benediktstroebl/agent-eval-harness/blob/7b231a952828022a43977f21acfd452adda5088c/agent_eval_harness/benchmarks/swebench_verified_mini_task_ids.txt)
- **Does not support arm64 machines**

For SWE-bench Verified, you will need to install the SWE-bench benchmark specific dependencies:
```bash
pip install -e .[swebench]
```

You will also need to **install docker** following the instructions [here](https://docs.docker.com/engine/install/). Docker is used during evaluation to run the SWE-bench tasks. For linux users, you will also need to complete the linux post-installation steps [here](https://docs.docker.com/engine/install/linux-postinstall/).
 

### [USACO](https://github.com/princeton-nlp/USACO)
- Programming competition problems
- Supports local, Docker, and VM execution

For USACO, you will need to download and extract the USACO dataset. This can be done with the following steps:

1. Download the USACO dataset from [here](https://drive.google.com/file/d/1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi/view?usp=share_link)
2. Unzip the dataset and move the `data` directory to `hal/benchmarks/USACO/`. Hence there should be a `data/` directory in `hal/benchmarks/USACO/`

You will also need to **install docker** following the instructions [here](https://docs.docker.com/engine/install/). Docker is used during evaluation to run the USACO tasks. For linux users, you will also need to complete the linux post-installation steps [here](https://docs.docker.com/engine/install/linux-postinstall/).
 
### [AppWorld](https://appworld.dev/)
- A Controllable World of Apps and People for Benchmarking Interactive Coding Agents
- **Requires VM execution** (`--vm` flag mandatory)

For AppWorld, you will need to install the AppWorld benchmark specific dependencies:
```bash
pip install -e .[appworld]
```

### [CORE-bench](https://github.com/siegelz/core-bench)
- Computational reproducibility benchmark for agents on real scientific papers
- Supports fully parallelized evaluation on Azure VMs

### [tau-bench](https://github.com/tau-bench/tau-bench)
- Install benchmark specific dependencies:
```bash
pip install -e .[taubench]
```
- Benchmark for Tool-Agent-User Interaction in real-world domains
- Supports fully parallelized evaluation on Azure VMs

### [SciCode](https://github.com/scicode-bench/SciCode)
- Programming realistic scientific research problems
- Supports three versions: 
  1. `scicode` : standard version from the paper where agent solves subtasks iteratively
  2. `scicode_easy` : agent solves subtasks iteratively but has access to additional background information
  3. `scicode_hard` : agent solves each full problem in a zero-shot format
- Supports both local and VM execution

For all SciCode benchmarks, you will need to download and extract the SciCode unit tests. This can be done with the following steps:

1. Download the unit tests [here](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR)
2. Move the file to `hal/benchmarks/SciCode/eval/data/`.
3. Install benchmark specific dependencies:
```bash
pip install -e .[scicode]
```

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

#### [AgentHarm](https://arxiv.org/abs/2410.09024)

- Benchmark for evaluating agent behavior on both benign and potentially harmful tasks
- Two variants available:
  - `inspect_evals/agentharm`: Evaluates agent behavior on potentially harmful tasks
  - `inspect_evals/agentharm_benign`: Evaluates agent behavior on benign tasks
- When using the default inspect agent with benign tasks, requires setting `-A task_name=benign`
- Example usage:
```bash
# For benign tasks
hal-eval --benchmark inspect_evals/agentharm_benign \
  --agent_dir agents/inspect/agentharm \
  --agent_function agentharm.default_agent \
  --agent_name "Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=openai/gpt-4o-mini-2024-07-18 \
  -A task_name=benign

# For potentially harmful tasks
hal-eval --benchmark inspect_evals/agentharm \
  --agent_dir agents/inspect/agentharm \
  --agent_function agentharm.default_agent \
  --agent_name "Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=openai/gpt-4o-mini-2024-07-18
```

### [CORE-Bench](https://arxiv.org/abs/2409.11363)

- Begin by decrypting `hal/benchmarks/corebench/core_test.json.gpg` to access the `CORE-Bench` test set. The password for the GPG file is `reproducibility`. To decrypt the file, run the following command:
```bash
gpg --output hal/benchmarks/corebench/core_test.json --decrypt hal/benchmarks/corebench/core_test.json.gpg
```
- Install benchmark specific dependencies:
```bash
pip install -e .[corebench]
```
- Benchmark for evaluating how agents can reproduce the results of scientific papers when provided with their code.
- Tasks involve setting up the environment, running the code, and answering questions about the results.
- Capsules and task files will automatically be downloaded upon running the benchmark.
- Three variants available:
  - `corebench_easy` - Agent provided with results and must answer task questions.
  - `corebench_medium` - Agent provided with Docker container to install dependencies and run code, and must answer task questions.
  - `corebench_hard` - Agent must install dependencies and run code from scratch, and answer task questions.
- Example usage:
```bash
hal-eval --benchmark corebench_easy \
  --agent_dir agents/list_files_agent \
  --agent_function main.run \
  --agent_name "Test Agent"
```

### [ScienceAgentBench](https://github.com/osunlp/ScienceAgentBench)
- Benchmark for evaluating agents' capabilities to solve computational science tasks
- Evaluates programs in a Docker container environment for consistent evaluation
- Supports both local and VM execution

For ScienceAgentBench, you will need to download and extract the ScienceAgentBench dataset. ScienceAgentBench will be automatically downloaded via the Hugging Face datasets library.

Requirements for the example agent (agents/sab_example_agent):
```
backoff==2.2.1
boto3==1.37.1
botocore==1.37.1
litellm==1.52.8
```

### [CollaborativeAgentBench](https://github.com/facebookresearch/sweet_rl)
- Benchmark for evaluating agents' capabilities to collaborate with humans for artifact creations
- Supports both frontend design and backend programming

For evaluation, follow the steps from [here](https://github.com/facebookresearch/sweet_rl) to set up, including installing packages, downloading data, and installing geckodriver (for frontend design only).
Example script for evaluations on colbench:
```bash
hal-eval --benchmark colbench_backend_programming --agent_dir agents/colbench_example_agent \
    --agent_name colbench_text_70b \
    --agent_function main.run -A model_name=/fsx-ram/shared/Meta-Llama-3.1-70B-Instruct \
    -B task_path=/home/yifeizhou/hal_collaborative/temp_data/test.jsonl \
    -A env_model_name=/fsx-ram/shared/Meta-Llama-3.1-70B-Instruct \
    --max_concurrent 100


hal-eval --benchmark colbench_frontend_design --agent_dir agents/colbench_example_agent \
    --agent_name colbench_text_70b \
    --agent_function main.run -A model_name=/fsx-ram/shared/Meta-Llama-3.1-8B-Instruct \
    -B task_path=/home/yifeizhou/hal_collaborative/temp_data/frontend_tasks/test.jsonl \
    -A env_model_name=/fsx-ram/shared/Qwen2-VL-72B-Instruct \
    -A cache_path=/fsx-ram/yifeizhou/collab_llm/driver_cache/ \
    --max_concurrent 20
```





## How Do I Run Evaluations?

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
    - `taubench_retail`: tau-bench retail domain
    - `taubench_airline`: tau-bench airline domain
    - `inspect_evals/gaia`: Gaia general AI assistants benchmark
    - `inspect_evals/cybench`: Cybersecurity agent tasks
    - `inspect_evals/agentharm`: AgentHarm
    - `inspect_evals/agentharm_benign`: AgentHarm benign evaluation
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
*   **`--docker`**: Run evaluation in Docker containers for isolation
*   **`--run_id <run_id>`**: Specify a run ID (useful for continuing runs)
*   **`--continue_run`**: Continue from a previous run (requires run_id)

### Example Evaluations

1. **Running SWE-bench locally:**
```bash
hal-eval --benchmark swebench_verified_mini \
  --agent_dir agents/swebench_example_agent/ \
  --agent_function main.run \
  --agent_name "My Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=gpt-4o-mini-2024-07-18 \
  --max_concurrent 5
```

2. **Running USACO in Docker containers:**
```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Solver (gpt-4o-mini-2024-07-18)" \
  --docker \
  --max_concurrent 5 \
  -A model_name=gpt-4o-mini-2024-07-18
```

3. **Running USACO on Azure VM:**
```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_example_agent/ \
  --agent_function main.run \
  --agent_name "USACO Solver (gpt-4o-2024-11-20)" \
  --vm \
  --max_concurrent 5 \
  -A model_name=gpt-4o-2024-11-20
```

4. **Running USACO with Amazon Bedrock models:**
```bash
hal-eval --benchmark usaco \
  --agent_dir agents/usaco_bedrock_models/ \
  --agent_function main.run \
  --agent_name "USACO Solver (claude-3-5-sonnet-20241022)" \
  -A model_name=bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0 \
  -A prompt_template_path=agents/usaco_bedrock_models/prompt_templates/claude.txt \
  --max_concurrent 10
```

More details on how to run the Amazon Bedrock models can be found [here](agents/RUN_AGENTS.md).

Available Bedrock models and their corresponding prompt templates:
| Model Name | Model ID | Prompt Template |
|-|-|-|
| Claude 3.5 Haiku | bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0 | claude.txt |
| Claude 3.5 Sonnet | bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0 | claude.txt |
| Claude 3 Sonnet | bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0 | claude.txt |
| Amazon Nova Pro | bedrock/amazon.nova-pro-v1:0 | nova.txt |
| Amazon Nova Lite | bedrock/amazon.nova-lite-v1:0 | nova.txt |
| Amazon Nova Micro | bedrock/amazon.nova-micro-v1:0 | nova.txt |
| Llama-3.3 70B | bedrock/us.meta.llama3-3-70b-instruct-v1:0 | llama.txt |

5. **Running Inspect AI benchmark:**
```bash
hal-eval --benchmark inspect_evals/gaia \
  --agent_dir agents/inspect/ \
  --agent_function gaia.default_agent \
  --agent_name "Gaia Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=openai/gpt-4o-mini-2024-07-18 \
  -I token_limit=4000 \
  -I temperature=0.4
```

3. **Running ScienceAgentBench locally:**
```bash
hal-eval --benchmark scienceagentbench \
  --agent_dir agents/sab_example_agent/ \
  --agent_function main.run \
  --agent_name "Science Agent (gpt-4o-mini-2024-07-18)" \
  -A model_name=gpt-4o-mini-2024-07-18 \
  -A use_self_debug=False \
  -A use_knowledge=False
```

### Agent Naming Guidelines

Agent names should follow this format: `Name (model1, model2)`. For example:
- `My Agent (gpt-4-0125-preview)`
- `SWE-agent (claude-3.5-sonnet-20241022-v2)`
- `Multi-Model Agent (gpt-4o-mini-2024-07-18, claude-3.5-sonnet-20241022-v2)`

Guidelines:
- Include exact model versions
- Put models in parentheses
- Separate multiple models with commas
- Keep names concise
- Don't include benchmark names

## How to Reproduce Existing Agents on HAL?

See [agents/RUN_AGENTS.md](agents/RUN_AGENTS.md) for detailed instructions on how to run existing agents across different benchmarks.

**Note:** We are actively working on adding support for more agents to enable easy reproduction of benchmark results. Currently, we support agents outlined in [agents/RUN_AGENTS.md](agents/RUN_AGENTS.md).

## How Do I Develop My Own Agents?

See [agents/README.md](agents/README.md) for details.

## How Do I Add a Benchmark?

See [hal/benchmarks/README.md](hal/benchmarks/README.md) for details.

## How Can I Submit My Results to the HAL Leaderboards?

Results can be uploaded to the [Holistic Agent Leaderboard (HAL)](https://agent-evals-leaderboard.hf.space) in several ways. To avoid benchmark contamination, we automatically encrypt the results before uploading.

1. **During Evaluation:**
   ```bash
   hal-eval --benchmark <benchmark> ... --upload
   ```

2. **After Evaluation:**
   ```bash
   # Upload all results for a benchmark
   hal-upload -B <benchmark_name>
   
   # Upload a single file
   hal-upload -F path/to/file.json
   
   # Upload all files in a directory
   hal-upload -D path/to/directory
   ```

   **Note:** When using `-F` to upload a single file, the file must be a JSON file.

## How can I use the agent traces from the HAL Leaderboard?

For each benchmark and agent on [HAL](hal.cs.princeton.edu), you can download the agent traces and use them for further analysis.

1. Download the agent traces from the [HAL Leaderboard](hal.cs.princeton.edu)
2. Use the `hal-decrypt` command to decrypt the traces. You can either decrypt and entire directory of traces or a single trace. For example:

Entire directory:
```bash
hal-decrypt -D path/to/directory
```

Single trace:
```bash
hal-decrypt -F taubench_airline_1743961943_UPLOAD.zip
```

## About HAL

The current landscape of AI agent evaluation faces several critical challenges. Benchmark evaluations tend to focus on accuracy while ignoring costs, leading to uninformative evaluations for downstream developers. What does it mean if an agent has 1% higher accuracy on a benchmark but is 10x more expensive? The lack of standardized evaluation practices makes it difficult to assess real-world capabilities and prevents meaningful comparisons between different approaches. As shown in "AI Agents That Matter" (arXiv:2407.01502), these issues have led to confusion about which advances actually improve performance.

HAL addresses these challenges through two key components: 1) A central leaderboard platform that incorporates cost-controlled evaluations by default, providing clear insights into the cost-performance tradeoffs of different agents, and 2) A standardized evaluation harness that enables reproducible agent evaluations across various benchmarks while tracking token usage and traces and **without** requiring any changes to the agent code or constraining agent developers to follow a certain agent framework. Evaluations can be run locally or in the cloud and fully parallelized.

**TLDR:** We aim to standardize AI agent evaluations by providing a third-party platform for comparing agents across various benchmarks. Our goal with HAL is to serve as a one-stop shop for agent evaluations, taking into account both accuracy and cost by default. The accompanying HAL harness offers a simple and scalable way to run agent evals - locally or in the cloud.

## Repository Structure

*   `hal/`: Core harness code
    *   `benchmarks/`: Benchmark implementations
        - `swebench.py`: SWE-bench implementation
        - `usaco.py`: USACO implementation
        - `mlagentbench.py`: MLAgentBench implementation
        - `appworld.py`: AppWorld implementation
        - `taubench.py`: tau-bench implementation
        - `inspect_benchmark.py`: Inspect AI benchmark support
    *   `utils/`: Utility functions
        - `local_runner.py`: Local execution support
        - `vm_runner.py`: Azure VM execution support
        - `docker_runner.py`: Docker container execution support
        - `weave_utils.py`: Weave logging utilities
    *   `inspect/`: Inspect AI specific code
*   `agents/`: Example agent implementations
*   `results/`: Evaluation results and logs


## Citing HAL

If you use HAL or the HAL harness in your research, please cite it by using the following BibTeX entry.

```bibtex
@Misc{hal,
  title =        {HAL: A Holistic Agent Leaderboard for Centralized and Reproducible Agent Evaluation},
  author =       {Benedikt Stroebl and Sayash Kapoor and Arvind Narayanan},
  howpublished = {\url{https://github.com/princeton-pli/hal-harness}},
  year =         {2025}
}
```

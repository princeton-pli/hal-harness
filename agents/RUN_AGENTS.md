# Rerunning Existing Agents on HAL

This guide provides detailed instructions for rerunning existing agents across HAL's supported benchmarks.

**Note:** We are actively working on adding support for more agents to enable easy reproduction of benchmark results. Currently, we support running the following agents below. 

## USACO

### Amazon Bedrock Models

Run USACO evaluations using foundation models available via Amazon Bedrock.

1. **Setup Requirements**

   - Create an Amazon EC2 VM with `t3.2xlarge` instance type and latest Ubuntu AMI ([EC2 setup guide](https://aws-samples.github.io/foundation-model-benchmarking-tool/misc/ec2_instance_creation_steps.html))
   - Ensure IAM role has `AmazonBedrockFullAccess` permissions
   - Install Docker:
     ```bash
     sudo apt-get update
     sudo apt-get install --reinstall docker.io -y
     sudo apt-get install -y docker-compose
     ```
   - Download USACO dataset to `hal/benchmarks/USACO/data/datasets/usaco_v3` ([download link](https://drive.google.com/file/d/1z5ODOJMqyer1QxzYtEUZ2hbAx-7nU8Vi/view?usp=share_link))
   - [Request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to desired Bedrock models

2. **Run Evaluation**

   ```bash
   BENCHMARK_NAME=usaco
   AGENT_DIR=agents/usaco_bedrock_models/
   AGENT_FUNCTION=main.run
   MODEL_NAME=bedrock/amazon.nova-lite-v1:0 
   AGENT_NAME="USACO_${MODEL_NAME}"
   PWD=`pwd`
   PROMPT_TEMPLATE_PATH=${PWD}/${AGENT_DIR}/prompt_templates/nova.txt
   CONCURRENCY=10  # adjust based on service quota

   hal-eval --benchmark $BENCHMARK_NAME \
       --agent_dir $AGENT_DIR \
       --agent_function $AGENT_FUNCTION \
       --agent_name $AGENT_NAME \
       -A model_name=$MODEL_NAME \
       -A prompt_template_path=$PROMPT_TEMPLATE_PATH \
       --max_concurrent $CONCURRENCY
   ```

3. **Available Models**

   | Model Name | Model ID | Prompt Template |
   |-|-|-|
   | Claude 3.5 Haiku | bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0 | claude.txt |
   | Claude 3.5 Sonnet | bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0 | claude.txt |
   | Claude 3 Sonnet | bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0 | claude.txt |
   | Amazon Nova Pro | bedrock/amazon.nova-pro-v1:0 | nova.txt |
   | Amazon Nova Lite | bedrock/amazon.nova-lite-v1:0 | nova.txt |
   | Amazon Nova Micro | bedrock/amazon.nova-micro-v1:0 | nova.txt |
   | Llama-3.3 70B | bedrock/us.meta.llama3-3-70b-instruct-v1:0 | claude.txt |

## Inspect AI Benchmarks

### GAIA Agent

Run evaluations using the GAIA general AI assistants benchmark:

```bash
hal-eval --benchmark inspect_evals/gaia \
    --agent_dir agents/inspect/ \
    --agent_function gaia.default_agent \
    --agent_name "Gaia Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o
```

### CyBench Agent

**Note:** CyBench does not support arm64 machines.

Run evaluations using the CyBench cybersecurity benchmark:

```bash
hal-eval --benchmark inspect_evals/cybench \
    --agent_dir agents/inspect/ \
    --agent_function cybench.default_agent \
    --agent_name "CyBench Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o
```

### AgentHarm

Run evaluations using the AgentHarm benchmark:

1. **For potentially harmful tasks:**
```bash
hal-eval --benchmark inspect_evals/agentharm \
    --agent_dir agents/inspect/agentharm \
    --agent_function agentharm.default_agent \
    --agent_name "Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o
```

2. **For benign tasks:**
```bash
hal-eval --benchmark inspect_evals/agentharm_benign \
    --agent_dir agents/inspect/agentharm \
    --agent_function agentharm.default_agent \
    --agent_name "Agent (gpt-4o)" \
    -A model_name=openai/gpt-4o \
    -A task_name=benign
```

## tau-bench

The following agents are available for the tau-bench benchmark. Both `taubench_retail` and `taubench_airline` are available. Run evaluations using the following commands:

### Minimal Agent

Run evaluations using a minimal agent that simply forwards the instruction to the LLM:

```bash
hal-eval --benchmark taubench_retail \
    --agent_dir agents/taubench/ \
    --agent_function minimal.run \
    --agent_name "Agent (gpt-4o-2024-11-20)" \
    -A model_name=gpt-4o-2024-11-20 \
    --max_concurrent 10
```

### Tool-Calling Agent

Run evaluations using an agent that leverages tool-calling capabilities:

```bash
hal-eval --benchmark taubench_retail \
    --agent_dir agents/taubench/ \
    --agent_function tool_calling.run \
    --agent_name "Taubench ToolCalling (gpt-4o-2024-11-20)" \
    -A model_name=gpt-4o-2024-11-20 \
    -A provider=openai \
    --max_concurrent 10
```

### ReAct Agent

Run evaluations using a ReAct agent that uses chain-of-thought reasoning:

```bash
hal-eval --benchmark taubench_retail \
    --agent_dir agents/taubench/ \
    --agent_function react.run \
    --agent_name "Taubench ReAct (gpt-4o-2024-11-20)" \
    -A model_name=gpt-4o-2024-11-20 \
    -A provider=openai \
    --max_concurrent 10
```


### SWE-Agent

First, you need to create a new conda environment using the follwoing command:
```bash
conda create -n swe-agent-1.0 python=3.11 -y
conda activate swe-agent-1.0

cd agents/SWE-agent-v1.0
python -m pip install --upgrade pip && pip install --editable .

conda run -n swe-agent-1.0 pip install "gql<4" # pin gql to v3
```

Run evaluations using SWE-Agent
```bash
model_name="claude-opus-4-1-20250805"

hal-eval --benchmark "swebench_verified_mini" --agent_dir agents/SWE-agent-v1.0 --agent_function main.run --agent_name "SWE-Agent($model_name)" -A agent.model.per_instance_cost_limit=$max_cost_limit -A agent.model.name=$model_name -A config=agents/SWE-agent-v1.0/config/benchmarks/250225_anthropic_filemap_simple_review.yaml --max_concurrent 1 --conda_env_name swe-agent-1.0
```

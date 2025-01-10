# Rerunning Existing Agents on HAL

This guide provides detailed instructions for rerunning existing agents across HAL's supported benchmarks.

**Note:** We are actively working on adding support for more agents to enable easy reproduction of benchmark results. Currently, we support running the following agents below. 

## USACO

### Amazon Bedrock Agent

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
   AGENT_DIR=agents/usaco_bedrock_agents/
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

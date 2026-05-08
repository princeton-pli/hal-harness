# ReplicatorBench

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Center for Open Science](https://img.shields.io/badge/Organization-COS-green)](https://cos.io)

Welcome to [**ReplicatorBench**](https://arxiv.org/pdf/2602.11354), a benchmark for evaluating LLM agents on end-to-end research replication tasks in social and behavioral sciences. The official implementation and resources are available at [COS/llm-benchmarking](https://github.com/CenterForOpenScience/llm-benchmarking). ReplicatorBench is built around 19 instances derived from SCORE replication efforts and evaluates agent behavior across the full replication workflow rather than only a final outcome.

---

## 🔍 Benchmark Overview

### Three Stages

ReplicatorBench decomposes replication into three stages:

* **Stage 1: Extraction**
  * Subtask 1: Extract structured information about the focal claim from the original paper into a *post-registration* of the original study.
  * Subtask 2: Search the web for replication data resources and provide a list of URLs.

* **Stage 2: Generation**
  * Subtask 3: **Design** a replication plan as a structured preregistration document (including hypotheses, data plan, proposed methodology, environment specifications, and analysis steps).
  * Subtask 4: **Execution** of the replication plan: process data, run models, produce raw results and intermediate outputs, and iteratively debug execution failures.

* **Stage 3: Interpretation**
  * Subtask 5: Compare replication results to the original study results recorded in the post-registration. Determine whether preregistered inference criteria are satisfied (e.g., α = 0.05, two-tailed, and the same pattern/direction as the original study). Produce a structured interpretation document with multiple graded fields.

ReplicatorBench also evaluates the final binary replication outcome: **Criteria Met** vs **Criteria Unmet**, aligned to the human replication outcome.

---

## 🔒 Rules and Environment

Across the replication workflow, the agent has read/write access to the designated workspace and may install and run software packages as needed for execution. The agent works in a closed environment, **except for the Extraction stage**, where internet access is provided. 

---

## 📦 HAL Packaging: Tasks and I/O Contract

In HAL, we represent the three stages using five tasks per study:

* `*_extract` (Extraction subtask: post-registration)
* `*_web_search` (Extraction subtask: URL retrieval)
* `*_design` (Generation step: design/preregistration)
* `*_execute` (Generation step: execution)
* `*_interpret` (Interpretation stage)

### Inputs and Outputs in the HAL Workspace

Each task prompt specifies the required locations:

* **Read-only inputs (capsule):** `/workspace/capsules/<capsule_id>/`
* **Writable per-study output directory:** `/workspace/<capsule_id>/`

For each study `<capsule_id>`, the task prompts require these output files under `/workspace/<capsule_id>/`:

* **Extract:** `post_registration.json`
* **Web Search:** `merged-urls.json`
* **Design:** `replication_info.json`
* **Execute:** `execution_results.json`
* **Interpret:** `interpret_results.json`

Schema templates are copied into:

* `/workspace/replicatorbench_templates/`

Agents should follow the prompt instructions for:

* required output filenames and JSON schemas
* "do not modify" constraints for the capsule directory
* the required "chat return" pointer object for each task

---

## ⚙️ Installation

Clone and Install [HAL](https://github.com/princeton-pli/hal-harness/tree/main?tab=readme-ov-file#setup) harness:

How to [setup](https://github.com/princeton-pli/hal-harness/tree/main?tab=readme-ov-file#setup)


---

## 🚀 Running ReplicatorBench in HAL

```bash
hal-eval --benchmark replicatorbench \
  --agent_dir agents/<your_agent_dir> \
  --agent_function main.run \
  --agent_name "<YourAgentName>" \
  --max_concurrent "<default: 1>" \
  --docker
```
---

## 📊 Evaluation

ReplicatorBench evaluates agent performance across the replication workflow using **checkpoint-based scoring** and **final-outcome reporting**.

* **Stage 1: Extraction**
  * Your agent produces a structured post-registration of the original study and a set of web-search URLs.
  * Outputs are evaluated against human-produced reference materials for structured fields.

* **Stage 2: Generation**
  * **Design:** Your agent produces a structured preregistration-style replication plan.
  * **Execute:** Your agent runs the replication and produces structured execution outputs.
  * The benchmark uses checkpoint-style evaluation to assess whether your agent successfully completes key steps (environment setup, dependency handling, correct file usage, execution/debugging, and documentation of outputs needed for interpretation).

* **Stage 3: Interpretation**
  * Your agent produces a structured interpretation that checks fidelity to the preregistered plan and whether preregistered inference criteria for the focal claim are met.
  * Interpretation is evaluated against the human replication report of required fields.
  * ReplicatorBench reports a final binary replication outcome (e.g., whether preregistered criteria are satisfied) aligned to the human replication outcome.

---

## 📂 ReplicatorBench Structure
```text
hal/benchmarks/
├── replicatorbench.py                 # ReplicatorBench entrypoint
└── replicatorbench/
    ├── tasks.json                     # Tasks (19 cases × 5 tasks per case)
    ├── setup.sh                       # Downloads capsules + prepares /workspace layout
    ├── templates/                     # JSON schemas + prompt templates copied into /workspace/replicatorbench_templates/
    ├── validator/                     # Stage validators (extract/design/execute/interpret)
    ├── core/                          # Logic used by the ReplicatorBench
    └── constants.py  
```
---


## 📬 Contact

For questions please contact:

**Shakhlo Nematova**  
Research Scientist  
[shakhlo@cos.io](mailto:shakhlo@cos.io)

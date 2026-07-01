# ReplicatorBench Agent

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Center for Open Science](https://img.shields.io/badge/Organization-COS-green)](https://cos.io)

Welcome to [**ReplicatorBench Agent**](https://arxiv.org/pdf/2602.11354), a agent implementation for the HAL **ReplicatorBench** benchmark. This work has been accepted to the [KDD 2026](https://kdd2026.kdd.org/) AI for Sciences Track. The official implementation and resources are available at [COS/llm-benchmarking](https://github.com/CenterForOpenScience/llm-benchmarking).

ReplicatorBench Agent evaluates on end-to-end research replication tasks in the social and behavioral sciences. In HAL, each study is decomposed into five tasks:

- `*_extract`
- `*_web_search`
- `*_design`
- `*_execute`
- `*_interpret`

The benchmark-level task contract, workspace layout, and evaluation details are documented in the benchmark README under `hal/benchmarks/replicatorbench/README.md`. This README explains the **agent-specific** behavior and how this agent is launched in HAL.

## 🔍 What this agent does

This agent wraps the underlying [COS/llm-benchmarking](https://github.com/CenterForOpenScience/llm-benchmarking) replication pipeline and exposes it through the HAL agent interface:

- HAL calls `main.run`
- `main.py` reads the current HAL task from `/workspace/input.json`
- it infers the task stage from the task id suffix
- it prepares the writable study directory at `/workspace/<capsule_id>/`
- it runs the required stage of the replication pipeline
- it returns a HAL pointer object for the output file required by that task

The agent uses the internal modules in this directory:

- `info_extractor/` for extraction and web search
- `generator/` for design and execution
- `interpreter/` for interpretation
- `core/` for agent logic, prompts, actions, and tools
- `templates/` for JSON schemas and prompt templates
- `validator/` for project-side validation utilities

## 🚀 HAL entrypoint

HAL runs this agent with:

```bash
hal-eval --benchmark replicatorbench \
  --agent_dir agents/replicatorbench_agent \
  --agent_function main.run \
  --agent_name "ReplicatorBenchAgent" \
  --max_concurrent 1 \
  --docker
```

This is the intended way to run the agent in HAL.

## 🔒 HAL workspace contract

For a study with capsule id `<capsule_id>`, HAL provides:

- read-only capsule inputs at:
  - `/workspace/capsules/<capsule_id>/`
- writable study output directory at:
  - `/workspace/<capsule_id>/`

This agent writes stage outputs into `/workspace/<capsule_id>/` using the filenames required by the benchmark:

- Extract:
  - `post_registration.json`
- Web Search:
  - `merged-urls.json`
- Design:
  - `replication_info.json`
- Execute:
  - `execution_results.json`
- Interpret:
  - `interpret_results.json`

Schema templates are expected under:

- `/workspace/replicatorbench_templates/`

The agent must not modify the capsule directory.

## 📦 Task-to-stage mapping

This agent maps HAL task ids to internal pipeline stages by suffix:

- `*_extract` → extract stage
- `*_web_search` → web search stage
- `*_design` → design stage
- `*_execute` → execute stage
- `*_interpret` → interpret stage

The wrapper in `main.py` is responsible for this mapping.

## 📊 Output return format

For each HAL task, the agent returns a pointer object keyed by the current task id.

Examples:

### Extract
```json
{
  "replicatorbench_02_extract": {
    "post_registration_path": "/workspace/replicatorbench_02/post_registration.json"
  }
}
```

### Web Search
```json
{
  "replicatorbench_02_web_search": {
    "merged_urls_path": "/workspace/replicatorbench_02/merged-urls.json"
  }
}
```

### Design
```json
{
  "replicatorbench_02_design": {
    "replication_info_path": "/workspace/replicatorbench_02/replication_info.json"
  }
}
```

### Execute
```json
{
  "replicatorbench_02_execute": {
    "execution_results_path": "/workspace/replicatorbench_02/execution_results.json"
  }
}
```

### Interpret
```json
{
  "replicatorbench_02_interpret": {
    "interpret_results_path": "/workspace/replicatorbench_02/interpret_results.json"
  }
}
```

## How the wrapper works

`main.py` is the HAL wrapper around the original replication pipeline:

1. Reads `/workspace/input.json`
2. Determines `task_id`, `capsule_id`, and stage
3. Prepares `/workspace/<capsule_id>/`
4. Ensures required earlier artifacts exist when later stages depend on them
5. Runs the appropriate pipeline target
6. Returns the required output pointer for HAL evaluation

This allows HAL to evaluate the five stage tasks separately while reusing the same per-study output directory.

## ⚙️ Dependencies

This agent depends on the same core stack used by the underlying replication pipeline, including:

- `openai`
- `python-dotenv`
- `pandas`
- `numpy`
- `pyreadr`
- `pymupdf`
- `python-docx`
- `pypdf`
- `docker`
- `pytest`
- `pytest-cov`

See:

- `requirements-dev.txt`
- `requirements.txt`
- `Makefile`

The HAL Docker run should install the needed dependencies in the container environment.

## 📂 Repository layout

```text
agents/replicatorbench_agent/
├── main.py
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── core/
├── generator/
├── info_extractor/
├── interpreter/
├── templates/
├── validator/
├── tests/
└── README.md
```

## Notes

- This agent README is specific to the HAL agent implementation.
- The benchmark README in `hal/benchmarks/replicatorbench/` remains the source of truth for benchmark semantics, task packaging, and evaluation logic.
- The make targets in this repository are internal implementation details used by the wrapper. HAL users should launch the benchmark with `hal-eval`, not by running those make commands manually.

## 📬 Contact

For questions please contact:

**Shakhlo Nematova**  
Research Scientist  
[shakhlo@cos.io](mailto:shakhlo@cos.io)

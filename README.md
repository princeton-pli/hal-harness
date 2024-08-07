## Setup

To clone this repository and initialize the benchmark submodules, follow these steps:

1. **Clone the main repository:**
   ```bash
   git clone https://github.com/benediktstroebl/agent-eval-harness.git
   ```

2. **Initialize and update submodules:**
   ```bash
   cd agent-eval-harness
   git submodule update --init --recursive
   ```

3. **Save .env with your API keys**
   ```bash
   cp .env.template .env
   ```
   Now add your personal API keys to the ```.env``` file.


## Simple example on SWE-Bench Lite
   ```bash
   python -m agent_eval_harness.cli --agent_function swebench_lite_example_agent.agent.run --benchmark swebench_lite --agent_dir swebench_lite_example_agent/ --agent_name swebench_lite_example_agent --upload
   ```



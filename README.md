## Cloning the Repository with Submodules

To clone this repository and initialize its submodules, follow these steps:

1. **Clone the main repository:**
   ```bash
   git clone https://github.com/benediktstroebl/agent-eval-harness.git
   ```

2. **Initialize and update submodules:**
   ```bash
   cd agent-eval-harness
   git submodule update --init --recursive
   ```


## Test run swebench lite
   ```bash
   poetry run python -m agent_eval_harness.cli --agent agent_eval_harness.example_agent.agent.run --benchmark swebench_lite
   ```



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


## Simple example on USACO
   ```bash
   python -m agent_eval_harness.cli --agent_function usaco_example_agent.agent.run --benchmark usaco --agent_dir usaco_example_agent/ --agent_name usaco_example_agent --upload
   ```



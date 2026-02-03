#!/usr/bin/env python3
"""
Docker container entrypoint for running HAL agents.
This script reads configuration from environment variables and executes the agent.
"""

import os
import weave
import traceback
import sys

# FIXME: use upgraded logging here


def main():
    """Main entrypoint that reads from env vars and runs the agent"""
    try:
        # Read configuration from environment variables
        run_id = os.environ.get("HAL_RUN_ID")
        task_id = os.environ.get("HAL_TASK_ID")
        agent_module = os.environ.get("HAL_AGENT_MODULE")
        agent_function = os.environ.get("HAL_AGENT_FUNCTION")

        # Validate required environment variables
        if not all([run_id, task_id, agent_module, agent_function]):
            raise ValueError(
                "Missing required environment variables. Required: "
                "HAL_RUN_ID, HAL_TASK_ID, HAL_AGENT_MODULE, HAL_AGENT_FUNCTION"
            )

        # Initialize weave
        weave.init(run_id)

        # Do the run
        # TODO: add the run here
        print("Running now!")
        print("Running more...")
        print("Run complete")

        sys.exit(0)

    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

        # Write error log
        with open("/workspace/error.log", "w") as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()

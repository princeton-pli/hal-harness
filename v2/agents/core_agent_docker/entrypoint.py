#!/usr/bin/env python3
"""
Docker container entrypoint for running HAL agents.
This script reads configuration from environment variables and executes the agent.
"""

import os
import weave
import traceback
import sys
import logging




def main():
    """Main entrypoint that reads from env vars and runs the agent"""
    # Create logs directory
    log_dir = "/workspace/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging to both file and stdout
    log_file = os.path.join(log_dir, "container.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Read configuration from environment variables
        run_id = os.environ.get("HAL_RUN_ID")
        task_id = os.environ.get("HAL_TASK_ID")

        # Validate required environment variables
        if not all([run_id, task_id]):
            raise ValueError(
                "Missing required environment variables. Required: "
                "HAL_RUN_ID, HAL_TASK_ID"
            )

        logger.info(f"Container starting - RunID={run_id}, TaskID={task_id}")

        # Test logs
        logger.info("HELLO WORLD - This is a test log from Docker container!")
        logger.info(f"Environment: RunID={run_id}, TaskID={task_id}")

        # Initialize weave
        logger.info("Initializing Weave...")
        weave.init(run_id)
        logger.info("Weave initialized successfully")

        # Do the run
        # TODO: add the run here
        logger.info("Running agent task...")
        logger.warning("This is a test warning from Docker")
        logger.error("This is a test error from Docker")
        logger.info("Run complete!")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error running agent: {e}")
        logger.error(traceback.format_exc())

        # Write error log
        with open("/workspace/error.log", "w") as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(traceback.format_exc())

        sys.exit(1)


if __name__ == "__main__":
    main()

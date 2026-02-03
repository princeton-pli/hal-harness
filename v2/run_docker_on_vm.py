#!/usr/bin/env python3

"""Run Docker containers on Azure VMs."""

import logging
import uuid

from azure_manager import AzureManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for running Docker on Azure VMs."""
    # Configuration
    virtual_machine_count = 3
    use_gpu = False

    # Generate run ID
    run_id = str(uuid.uuid4())[:20]
    logger.info(
        f"Starting run {run_id}. virtual_machine_count={virtual_machine_count}, use_gpu={use_gpu}"
    )

    # Initialize Azure manager
    azure_manager = AzureManager(
        run_id=run_id, virtual_machine_count=virtual_machine_count, use_gpu=use_gpu
    )

    try:
        # Run Docker on each VM
        for vm in azure_manager.virtual_machines:
            vm.run_docker(
                env_vars={
                    "HAL_RUN_ID": run_id,
                    "HAL_TASK_ID": f"task-{vm.name}",
                    "HAL_AGENT_MODULE": "main",
                    "HAL_AGENT_FUNCTION": "run",
                }
            )

        logger.info(f"Triggered Docker runs for {virtual_machine_count} VMs")

    finally:
        # Cleanup VMs
        logger.info("Cleaning up resources")
        import pdb

        pdb.set_trace()
        azure_manager.cleanup()


if __name__ == "__main__":
    main()

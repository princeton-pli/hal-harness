#!/usr/bin/env python3

"""Run Docker containers on Azure VMs."""

import argparse
import logging
import os
import uuid

from infrastructure.azure.resource_manager import AzureResourceManager
from hal.logging.logging_utils import setup_logging


logger = logging.getLogger(__name__)


def main():
    """Main entry point for running Docker on Azure VMs."""
    parser = argparse.ArgumentParser(description="Run Docker containers on Azure VMs")
    parser.add_argument(
        "--image",
        help="Docker image name to run (will be transferred from local machine to VMs)",
    )
    parser.add_argument(
        "--vm_count",
        type=int,
        default=2,
        help="Number of VMs to create (default: 2)",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Use GPU VMs",
    )
    parser.add_argument(
        "--dce-endpoint",
        required=True,
        help="Azure Monitor Data Collection Endpoint URL",
    )
    parser.add_argument(
        "--dcr-id",
        required=True,
        help="Azure Monitor Data Collection Rule ID",
    )
    args = parser.parse_args()

    # Configuration
    virtual_machine_count = args.vm_count
    use_gpu = args.gpu
    docker_image = args.image

    # Generate run ID
    run_id = str(uuid.uuid4())[:18]

    # Setup logging (with Azure Monitor if running in VM)
    log_dir = os.path.join(os.getcwd(), "logs")
    setup_logging(log_dir=log_dir, run_id=run_id, use_azure=True)

    logger.info(
        f"Starting run {run_id}. virtual_machine_count={virtual_machine_count}, use_gpu={use_gpu}, image={docker_image}"
    )

    # FIXME: check for the docker image presence here (fail fast)

    # Initialize Azure manager
    azure_manager = AzureResourceManager(
        run_id=run_id,
        virtual_machine_count=virtual_machine_count,
        use_gpu=use_gpu,
        dcr_id=args.dcr_id,
    )

    try:
        # Prepare env vars for Docker containers
        docker_env_vars = {
            "HAL_RUN_ID": run_id,
            "AZURE_MONITOR_DATA_COLLECTION_ENDPOINT": args.dce_endpoint,
            "AZURE_MONITOR_DATA_COLLECTION_RULE_ID": args.dcr_id,
            "AZURE_MONITOR_STREAM_NAME": "Custom-BenchmarkRuns_CL",
        }

        # Run Docker on each VM
        for vm in azure_manager.virtual_machines:
            # Add a task ID for this VM
            task_id = str(uuid.uuid4())[:18]
            vm_env_vars = docker_env_vars.copy()
            vm_env_vars["HAL_TASK_ID"] = task_id

            vm.send_docker_image_by_name(docker_image)

            # Run docker with the env vars and the task ID
            vm.run_docker(image_name=docker_image, env_vars=vm_env_vars)

        logger.info(f"Triggered Docker runs for {virtual_machine_count} VMs")

    except Exception as e:
        raise e
    finally:
        # Cleanup VMs
        logger.info("Cleaning up resources")
        import pdb

        pdb.set_trace()
        azure_manager.cleanup()


if __name__ == "__main__":
    main()

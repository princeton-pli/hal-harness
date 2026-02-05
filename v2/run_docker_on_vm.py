#!/usr/bin/env python3

"""Run Docker containers on Azure VMs."""

import argparse
import logging
import os
import uuid

from azure_manager import AzureManager
from logging_utils import setup_logging


logger = logging.getLogger(__name__)


def main():
    """Main entry point for running Docker on Azure VMs."""
    parser = argparse.ArgumentParser(description="Run Docker containers on Azure VMs")
    parser.add_argument(
        "--image",
        # Note: right now this pulls a docker image that's on the invoker/orchestrator machine; we may want these to be
        # in a image repository at some point
        # FIXME: This image exists locally on the orchestrator but needs to be transferred to the VM
        # Options: 1) docker save + scp + docker load, 2) push to registry + pull on VM, 3) rebuild on VM
        help="This is the name of the docker image that you want to run",
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
    run_id = str(uuid.uuid4())[:20]

    # Setup logging (with Azure Monitor if running in VM)
    log_dir = os.path.join(os.getcwd(), "logs")
    setup_logging(log_dir=log_dir, run_id=run_id, use_azure=True)

    logger.info(
        f"Starting run {run_id}. virtual_machine_count={virtual_machine_count}, use_gpu={use_gpu}, image={docker_image}"
    )

    # Initialize Azure manager
    azure_manager = AzureManager(
        run_id=run_id,
        virtual_machine_count=virtual_machine_count,
        use_gpu=use_gpu,
        dcr_id=args.dcr_id,
    )

    try:
        # FIXME: Before running docker, we need to transfer the image to each VM
        # Current flow is broken: docker_image exists locally but VM doesn't have it
        # Need to add: azure_manager.transfer_image(docker_image) or vm.transfer_image(docker_image)

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
            task_id = str(uuid.uuid4())[:20]
            vm_env_vars = docker_env_vars.copy()
            vm_env_vars["HAL_TASK_ID"] = task_id

            # Run docker with the env vars and the task ID
            vm.run_docker(image=docker_image, env_vars=vm_env_vars)

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

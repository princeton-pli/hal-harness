"""Azure Virtual Machine representation."""

import logging
import os
import subprocess

from utils import run_command


logger = logging.getLogger(__name__)


class AzureVirtualMachine:
    """Represents a single Azure VM."""

    def __init__(self, name: str, public_ip: str, resource_group: str):
        self.name = name
        self.public_ip = public_ip
        self.resource_group = resource_group

    def run_docker(
        self,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        image = "hal-agent-runner:latest"
        """Run Docker container on this VM via SSH."""
        logger.info(f"Running Docker on VM {self.name}")

        ssh_key_path = os.getenv("SSH_PRIVATE_KEY_PATH")
        if not ssh_key_path:
            raise ValueError("SSH_PRIVATE_KEY_PATH not set")

        # Build docker run command
        env_flags = ""
        if env_vars:
            env_flags = " ".join([f"-e {k}={v}" for k, v in env_vars.items()])

        docker_cmd = f"docker run --rm {env_flags} {image}"

        # SSH and run Docker
        ssh_cmd = [
            "ssh",
            "-i",
            ssh_key_path,
            "-o",
            "StrictHostKeyChecking=no",
            f"agent@{self.public_ip}",
            docker_cmd,
        ]

        # Run in background (non-blocking)
        subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Docker started on VM {self.name}")

    def delete(self) -> None:
        """Delete this VM and all resources."""
        logger.info(f"Deleting VM {self.name}")

        resources = [
            ("vm", self.name),
            ("nic", f"{self.name}-nic"),
            ("public-ip", f"{self.name}-public-ip"),
            ("vnet", f"{self.name}-vnet"),
        ]

        for resource_type, resource_name in resources:
            if resource_type == "vm":
                cmd = [
                    "az",
                    "vm",
                    "delete",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    resource_name,
                    "--yes",
                ]
            elif resource_type == "nic":
                cmd = [
                    "az",
                    "network",
                    "nic",
                    "delete",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    resource_name,
                ]
            elif resource_type == "public-ip":
                cmd = [
                    "az",
                    "network",
                    "public-ip",
                    "delete",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    resource_name,
                ]
            elif resource_type == "vnet":
                cmd = [
                    "az",
                    "network",
                    "vnet",
                    "delete",
                    "--resource-group",
                    self.resource_group,
                    "--name",
                    resource_name,
                ]

            run_command(cmd, check=False)

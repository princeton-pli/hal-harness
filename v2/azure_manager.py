"""Azure resource manager for VM and Docker orchestration."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from azure_virtual_machine import AzureVirtualMachine
from utils import run_command


logger = logging.getLogger(__name__)


class AzureManager:
    """Manages Azure resources for a run."""

    def __init__(self, run_id: str):
        # Load environment variables from .env
        load_dotenv()

        self.run_id = run_id
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP_NAME")
        self.location = os.getenv("AZURE_LOCATION", "eastus")
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.nsg_name = os.getenv("NETWORK_SECURITY_GROUP_NAME")
        self.ssh_public_key = self._read_ssh_key()

        if not all([self.resource_group, self.subscription_id, self.nsg_name]):
            raise ValueError("Missing required Azure environment variables")

        self.template_dir = Path(__file__).parent / "infrastructure"
        self.virtual_machines: list[AzureVirtualMachine] = []

        # Configure the network and VMs
        self._create_network()
        self._create_virtual_machines()

    def _read_ssh_key(self) -> str:
        """Read SSH public key from path in env."""
        ssh_key_path = os.getenv("SSH_PUBLIC_KEY_PATH")
        if not ssh_key_path:
            raise ValueError("SSH_PUBLIC_KEY_PATH not set")
        return open(ssh_key_path).read().strip()

    def _create_network(self) -> dict:
        """Deploy NSG infrastructure."""
        logger.info(f"Creating network infrastructure for run {self.run_id}")

        cmd = [
            "az",
            "deployment",
            "group",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            f"nsg-{self.run_id}",
            "--template-file",
            str(self.template_dir / "main.bicep"),
            "--parameters",
            f"networkSecurityGroupName={self.nsg_name}",
            f"tags={{run_id:{self.run_id}}}",
        ]

        run_command(cmd)
        return {"status": "created"}

    def _create_virtual_machines(
        self, count: int, use_gpu: bool = False
    ) -> list[AzureVirtualMachine]:
        """Create multiple VMs."""
        logger.info(f"Creating {count} VMs (GPU: {use_gpu})")

        # FIXME: do this in parallel
        vms = []
        for i in range(count):
            vm_name = f"vm-{self.run_id}-{i}"[:32].replace("_", "-")
            # FIXME: use Azure here to initiate VMs and return AzureVirtualMachine
            vm = self._create_vm(vm_name, use_gpu)
            vms.append(vm)

        self.vms.extend(vms)
        return vms

    def cleanup(self) -> None:
        """Delete all VMs created in this run."""
        logger.info(f"Cleaning up {len(self.vms)} VMs")
        for vm in self.vms:
            vm.delete()

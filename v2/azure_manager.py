"""Azure resource manager for VM and Docker orchestration."""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from azure_virtual_machine import AzureVirtualMachine


logger = logging.getLogger(__name__)


class AzureManager:
    """Manages Azure resources for a run."""

    def __init__(self, run_id: str, virtual_machine_count: int, use_gpu: bool = False):
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
        self.nsg_id = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/networkSecurityGroups/{self.nsg_name}"

        # Create VMs in parallel
        self.virtual_machines = asyncio.run(
            self._create_virtual_machines_async(virtual_machine_count, use_gpu)
        )

    def _read_ssh_key(self) -> str:
        """Read SSH public key from path in env."""
        ssh_key_path = os.getenv("SSH_PUBLIC_KEY_PATH")
        if not ssh_key_path:
            raise ValueError("SSH_PUBLIC_KEY_PATH not set")
        return open(ssh_key_path).read().strip()

    async def _create_virtual_machines_async(
        self, count: int, use_gpu: bool
    ) -> list[AzureVirtualMachine]:
        """Create multiple VMs in parallel."""
        logger.info(f"Creating {count} VMs in parallel (GPU: {use_gpu})")

        async def create_vm_async(index: int) -> AzureVirtualMachine:
            """Create a single VM in a thread."""
            vm_name = f"vm-{self.run_id}-{index}"[:32].replace("_", "-")

            # Run VM creation in thread pool (blocking Azure SDK calls)
            return await asyncio.to_thread(
                AzureVirtualMachine,
                name=vm_name,
                resource_group=self.resource_group,
                location=self.location,
                subscription_id=self.subscription_id,
                nsg_id=self.nsg_id,
                ssh_public_key=self.ssh_public_key,
                gpu=use_gpu,
            )

        # Create all VMs concurrently
        vms = await asyncio.gather(*[create_vm_async(i) for i in range(count)])
        return list(vms)

    def cleanup(self) -> None:
        """Delete all VMs created in this run (in parallel)."""
        logger.info(f"Cleaning up {len(self.virtual_machines)} VMs")

        async def cleanup_async():
            tasks = [
                asyncio.to_thread(vm.delete) for vm in self.virtual_machines
            ]
            await asyncio.gather(*tasks)

        asyncio.run(cleanup_async())

"""Azure Virtual Machine representation."""

import base64
import logging
import os
import subprocess
import time

from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import (
    HardwareProfile,
    ImageReference,
    LinuxConfiguration,
    NetworkInterfaceReference,
    NetworkProfile,
    OSDisk,
    OSProfile,
    SshConfiguration,
    SshPublicKey,
    StorageProfile,
    VirtualMachine,
)
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential

from .utils import run_command


logger = logging.getLogger(__name__)


class AzureVirtualMachine:
    """Represents a single Azure VM."""

    def __init__(
        self,
        name: str,
        resource_group: str,
        location: str,
        subscription_id: str,
        nsg_id: str,
        ssh_public_key: str,
        gpu: bool,
        timeout: int,
    ):
        self.name = name
        self.resource_group = resource_group
        self.location = location
        self.vm_size = "Standard_NC4as_T4_v3" if gpu else "Standard_E2as_v5"
        self.gpu = gpu
        self.timeout = timeout

        # Azure clients
        credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(credential, subscription_id)
        self.network_client = NetworkManagementClient(credential, subscription_id)

        # Store for later use
        self.nsg_id = nsg_id
        self.ssh_public_key = ssh_public_key
        self.public_ip = None

        # Create the VM
        self._create()

    def _run_ssh(
        self, remote_command: str, *, timeout: int = 10
    ) -> subprocess.CompletedProcess:
        """Run a command on the VM over SSH."""
        ssh_key_path = os.getenv("SSH_PRIVATE_KEY_PATH")
        cmd = [
            "ssh",
            "-i",
            ssh_key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=5",
            f"agent@{self.public_ip}",
            remote_command,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def check_for_file_presence_by_path(self, file_path: str) -> bool:
        """
        Checks if a file is present on the virtual machine.

        :param self: the virtual machine
        :param file_path: the file path to check
        :type file_path: str
        :return: whether or not the file is present
        :rtype: bool
        """
        try:
            result = self._run_ssh(f"test -f {file_path}")
        except (subprocess.TimeoutExpired, OSError):
            return False

        return result.returncode == 0

    def _create(self) -> None:
        """Create VM using Azure SDK."""
        logger.info(
            f"Creating VM {self.name} {'with' if self.gpu else 'WITHOUT'} a GPU"
        )

        # Create VNet
        vnet_name = f"{self.name}-vnet"
        subnet_name = f"{self.name}-subnet"
        vnet = self.network_client.virtual_networks.begin_create_or_update(
            self.resource_group,
            vnet_name,
            {
                "location": self.location,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]},
                "subnets": [{"name": subnet_name, "address_prefix": "10.0.0.0/24"}],
            },
        ).result()
        subnet = vnet.subnets[0]

        # Create public IP
        public_ip_name = f"{self.name}-public-ip"
        public_ip = self.network_client.public_ip_addresses.begin_create_or_update(
            self.resource_group,
            public_ip_name,
            {
                "location": self.location,
                "sku": {"name": "Standard"},
                "public_ip_allocation_method": "Static",
            },
        ).result()
        self.public_ip = public_ip.ip_address

        # Create NIC
        nic_name = f"{self.name}-nic"
        nic = self.network_client.network_interfaces.begin_create_or_update(
            self.resource_group,
            nic_name,
            {
                "location": self.location,
                "ip_configurations": [
                    {
                        "name": "default",
                        "subnet": {"id": subnet.id},
                        "public_ip_address": {"id": public_ip.id},
                    }
                ],
                "network_security_group": {"id": self.nsg_id},
            },
        ).result()

        # Load cloud-init config
        cloud_init_path = os.path.join(os.path.dirname(__file__), "cloud_init.yaml")
        with open(cloud_init_path, "r") as f:
            cloud_init_config = f.read()

        # Azure requires custom_data to be base64 encoded
        custom_data = base64.b64encode(cloud_init_config.encode()).decode()

        # Create VM
        # Use Ubuntu-HPC image for GPU VMs (has NVIDIA drivers pre-installed)
        if self.gpu:
            image_reference = ImageReference(
                publisher="microsoft-dsvm",
                offer="ubuntu-hpc",
                sku="2204",
                version="latest",
            )
            logger.info(
                f"Using Ubuntu-HPC 22.04 image with pre-installed GPU drivers for {self.name}"
            )
        else:
            image_reference = ImageReference(
                publisher="Canonical",
                offer="0001-com-ubuntu-server-jammy",
                sku="22_04-lts-gen2",
                version="latest",
            )
            logger.info(f"Using standard Ubuntu 22.04 image for {self.name}")

        vm = VirtualMachine(
            location=self.location,
            hardware_profile=HardwareProfile(vm_size=self.vm_size),
            storage_profile=StorageProfile(
                image_reference=image_reference,
                os_disk=OSDisk(create_option="FromImage", disk_size_gb=80),
            ),
            os_profile=OSProfile(
                computer_name=self.name,
                admin_username="agent",
                custom_data=custom_data,
                linux_configuration=LinuxConfiguration(
                    disable_password_authentication=True,
                    ssh=SshConfiguration(
                        public_keys=[
                            SshPublicKey(
                                path="/home/agent/.ssh/authorized_keys",
                                key_data=self.ssh_public_key,
                            )
                        ]
                    ),
                ),
            ),
            network_profile=NetworkProfile(
                network_interfaces=[NetworkInterfaceReference(id=nic.id)]
            ),
        )

        self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group, self.name, vm
        ).result()

        logger.info(f"VM {self.name} created at {self.public_ip}")

        # Wait for startup script to complete
        startup_start = time.time()
        self._wait_for_setup_to_complete()
        startup_duration = int(time.time() - startup_start)
        logger.info(
            f"Startup script completed for VM {self.name} in {startup_duration} seconds"
        )

    def _wait_for_setup_to_complete(self) -> None:
        """Wait for startup script to complete by checking for sentinel file.

        Uses self.timeout (passed from VirtualMachineRunner.task_timeout).
        Streams new lines from /home/agent/cloud_init.log to the runner logs.
        """
        start_time = time.time()
        sentinel_file = "/home/agent/startup_complete"
        startup_log = "/home/agent/cloud_init.log"
        log_offset = 0

        logger.info(
            f"Waiting for startup script to complete on {self.name} (timeout: {self.timeout}s)"
        )
        while time.time() - start_time < self.timeout:
            if self.check_for_file_presence_by_path(sentinel_file):
                self._stream_startup_log(startup_log, log_offset)
                logger.info(
                    f"Startup script completed on {self.name} at {self.public_ip}"
                )
                return

            log_offset = self._stream_startup_log(startup_log, log_offset)
            elapsed = int(time.time() - start_time)
            started = self.check_for_file_presence_by_path(
                "/home/agent/startup_started"
            )
            logger.info(
                f"Startup in progress on {self.name} ({self.public_ip}): "
                f"{elapsed}s elapsed, cloud-init started={started}"
            )
            time.sleep(10)

        self._stream_startup_log(startup_log, log_offset)
        raise TimeoutError(
            f"Startup script did not complete on {self.name} ({self.public_ip}) within {self.timeout} seconds"
        )

    def _stream_startup_log(self, remote_path: str, offset: int) -> int:
        """Print new bytes from the remote startup log; return the new offset."""
        try:
            result = self._run_ssh(
                f"tail -c +{offset + 1} {remote_path} 2>/dev/null || true"
            )
        except (subprocess.TimeoutExpired, OSError):
            return offset

        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"{self.name} cloud-init: {line}")
            return offset + len(result.stdout.encode())
        return offset

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

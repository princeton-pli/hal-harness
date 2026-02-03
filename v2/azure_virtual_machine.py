"""Azure Virtual Machine representation."""

import logging
import os
import subprocess

from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential

from utils import run_command


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
        vm_size: str = "Standard_E2as_v5",
        gpu: bool = False,
    ):
        self.name = name
        self.resource_group = resource_group
        self.location = location
        self.vm_size = vm_size if not gpu else "Standard_NC4as_T4_v3"
        self.gpu = gpu

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

    def _create(self) -> None:
        """Create VM using Azure SDK."""
        logger.info(f"Creating VM {self.name} ({'GPU' if self.gpu else 'standard'})")

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

        # Create VM
        vm_params = {
            "location": self.location,
            "storage_profile": {
                "image_reference": {
                    "publisher": "Canonical",
                    "offer": "0001-com-ubuntu-server-focal",
                    "sku": "20_04-lts-gen2",
                    "version": "latest",
                },
                "os_disk": {"createOption": "FromImage", "diskSizeGB": 80},
            },
            "hardware_profile": {"vm_size": self.vm_size},
            "os_profile": {
                "computer_name": self.name,
                "admin_username": "agent",
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": "/home/agent/.ssh/authorized_keys",
                                "key_data": self.ssh_public_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id}]},
        }

        if self.gpu:
            vm_params["security_profile"] = {
                "uefi_settings": {"secure_boot_enabled": False},
                "security_type": "TrustedLaunch",
            }

        self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group, self.name, vm_params
        ).result()

        # Install GPU driver if needed
        if self.gpu:
            logger.info(f"Installing NVIDIA GPU driver on {self.name} (~12 min)")
            self.compute_client.virtual_machine_extensions.begin_create_or_update(
                self.resource_group,
                self.name,
                "NvidiaGpuDriverLinux",
                {
                    "location": self.location,
                    "publisher": "Microsoft.HpcCompute",
                    "type_properties_type": "NvidiaGpuDriverLinux",
                    "type_handler_version": "1.9",
                    "auto_upgrade_minor_version": True,
                    "settings": {},
                },
            ).result()

        logger.info(f"VM {self.name} created at {self.public_ip}")

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

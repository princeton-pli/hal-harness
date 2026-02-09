"""Azure Virtual Machine representation."""

import base64
import logging
import os
import subprocess
import time
import tempfile

from fabric import Connection
from invoke.runners import Result

from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.identity import DefaultAzureCredential

from hal.utils import run_command


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
        dcr_id: str,
        vm_size: str = "Standard_E2as_v5",
        gpu: bool = False,
    ):
        self.name = name
        self.resource_group = resource_group
        self.location = location
        self.subscription_id = subscription_id
        self.vm_size = vm_size if not gpu else "Standard_NC4as_T4_v3"
        self.gpu = gpu
        self.dcr_id = dcr_id

        # Azure clients
        credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(credential, subscription_id)
        self.network_client = NetworkManagementClient(credential, subscription_id)
        self.monitor_client = MonitorManagementClient(credential, subscription_id)

        # Store for later use
        self.nsg_id = nsg_id
        self.public_ip = None
        self.ssh_public_key = ssh_public_key

        # Get private key path from env
        self.ssh_private_key_path = os.getenv("SSH_PRIVATE_KEY_PATH")
        if not self.ssh_private_key_path:
            raise ValueError("SSH_PRIVATE_KEY_PATH not set")

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
        logging.info(f"Public IP {self.public_ip} created for VM {self.name}")

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
        cloud_init_path = os.path.join(
            os.path.dirname(__file__), "virtual_machine_cloud_init.yaml"
        )
        with open(cloud_init_path, "r") as f:
            cloud_init_config = f.read()

        custom_data = base64.b64encode(cloud_init_config.encode()).decode()

        # Create VM
        # Use Ubuntu-HPC image for GPU VMs (has NVIDIA drivers pre-installed)
        if self.gpu:
            image_reference = {
                "publisher": "microsoft-dsvm",
                "offer": "ubuntu-hpc",
                "sku": "2404",
                "version": "latest",
            }
            logger.info(
                f"Using Ubuntu-HPC 24.04 image with pre-installed GPU drivers for {self.name}"
            )
        else:
            image_reference = {
                "publisher": "Canonical",
                "offer": "ubuntu-24_04-lts",
                "sku": "server",
                "version": "latest",
            }
            logger.info(f"Using standard Ubuntu 24.04 LTS image for {self.name}")

        vm_params = {
            "location": self.location,
            "storage_profile": {
                "image_reference": image_reference,
                "os_disk": {"createOption": "FromImage", "diskSizeGB": 80},
            },
            "hardware_profile": {"vm_size": self.vm_size},
            "os_profile": {
                "computer_name": self.name,
                "admin_username": "agent",
                "custom_data": custom_data,
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

        self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group, self.name, vm_params
        ).result()

        # FIXME: restore Azure Monitor Agent and DCR associationonce we have the permissions
        # Install Azure Monitor Agent
        # logger.info(f"Installing Azure Monitor Agent on {self.name}")
        # self.compute_client.virtual_machine_extensions.begin_create_or_update(
        #     self.resource_group,
        #     self.name,
        #     "AzureMonitorLinuxAgent",
        #     {
        #         "location": self.location,
        #         "publisher": "Microsoft.Azure.Monitor",
        #         "type_properties_type": "AzureMonitorLinuxAgent",
        #         "type_handler_version": "1.0",
        #         "auto_upgrade_minor_version": True,
        #         "settings": {},
        #     },
        # ).result()

        # logger.info(
        #     f"VM {self.name} created at {self.public_ip}; waiting for startup script"
        # )

        # # Wait for SSH to be ready
        # self._wait_for_setup_to_complete()
        # logger.info(f"Got SSH for VM {self.name}")

        # # Associate Data Collection Rule if provided
        # # FIXME: get the permissions to run _associate_dcr !!!
        # self._associate_dcr()

    def _wait_for_setup_to_complete(self, timeout: int = 600) -> None:
        """Wait for startup script to complete by checking for sentinel file.

        Args:
            timeout: Maximum time to wait in seconds (default 600)
        """
        logger.info(f"Waiting for startup script to complete on {self.name}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if sentinel file exists
                result = self.run_command(
                    "test -f /home/agent/startup_complete", hide=True, warn=True
                )

                if result.exited == 0:
                    logger.info(
                        f"Startup script completed on {self.name} at {self.public_ip}"
                    )
                    return
            except Exception:
                pass

            time.sleep(10)

        raise TimeoutError(
            f"Startup script did not complete on {self.name} ({self.public_ip}) within {timeout} seconds"
        )

    def _associate_dcr(self) -> None:
        """Associate a Data Collection Rule with this VM for log collection."""
        logger.info(f"Associating DCR with VM {self.name}")

        vm_resource_id = (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.Compute/virtualMachines/{self.name}"
        )

        association_name = f"{self.name}-dcr-association"

        try:
            self.monitor_client.data_collection_rule_associations.create(
                resource_uri=vm_resource_id,
                association_name=association_name,
                body={"properties": {"dataCollectionRuleId": self.dcr_id}},
            )
            logger.info(f"DCR associated with VM {self.name}")
        except Exception as e:
            logger.error(f"Failed to associate DCR with VM {self.name}: {e}")
            raise

    def send_docker_image_by_name(self, image: str) -> None:
        """Transfer a Docker image from local machine to this VM.

        Uses docker save + scp + docker load approach for simplicity.
        For production, consider using Azure Container Registry instead.

        Args:
            image: Docker image name (e.g., "my-agent:latest")

        Raises:
            RuntimeError: If image doesn't exist locally or transfer fails
        """

        # FIXME: get timestamp for local Docker image to confirm when it was built
        # FIXME: log the sha for the docker image on local and on the remote (throw an error if they don't match)
        # ^^^ this is the command: docker images --digests --format '{{.Digest}}' hal-core-agent-docker:latest

        # Save image to temporary tar file
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_file:
            tar_path = tmp_file.name
        # FIXME: only make the tempfile once if there are multiple possible versions
        try:
            logger.info(f"Saving image {image} to tar file at {tar_path}...")
            subprocess.run(
                ["docker", "save", "-o", tar_path, image],
                check=True,
                capture_output=True,
            )

            remote_tar_path = f"/tmp/{os.path.basename(tar_path)}"

            image_size_gb = os.path.getsize(tar_path) / (1024**3)
            logger.info(f"Image size: {image_size_gb:.2f} GB")

            logger.info(f"Transferring image to VM {self.name}...")

            subprocess.run(
                [
                    "scp",
                    "-i",
                    self.ssh_private_key_path,
                    "-o",
                    "StrictHostKeyChecking=no",
                    tar_path,
                    f"agent@{self.public_ip}:{remote_tar_path}",
                ],
                check=True,
                capture_output=True,
            )

            # Load image on VM
            logger.info(f"Loading image on VM {self.name}...")
            self.run_command(
                f"docker load -i {remote_tar_path} && rm {remote_tar_path}", hide=True
            )

            logger.info(f"Image {image} successfully transferred to VM {self.name}")

        finally:
            # Clean up local tar tempfile
            if os.path.exists(tar_path):
                os.remove(tar_path)

    def _check_if_docker_image_exists(self, image_name: str) -> bool:
        # Check if the docker image exists on the VM
        result = self.run_command(
            f"docker image inspect {image_name} > /dev/null 2>&1", hide=True, warn=True
        )
        if result.exited != 0:
            raise RuntimeError(
                f"Docker image '{image_name}' not found on VM {self.name}. "
                f"Call send_docker_image_by_name() first to transfer the image."
            )

    def run_docker(
        self,
        image_name: str,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        """Run Docker container on this VM via SSH.

        Note: Image must already exist on the VM. Call send_docker_image_by_name()
        first to transfer the image from the local machine.

        Args:
            image: Docker image name to run
            env_vars: Optional environment variables to pass to the container
        """
        logger.info(f"Running Docker image {image_name} on VM {self.name}")

        # Create log directory on VM (flat structure: run_id-task_id)
        run_id = env_vars.get("HAL_RUN_ID", "unknown") if env_vars else "unknown"
        task_id = env_vars.get("HAL_TASK_ID", "unknown") if env_vars else "unknown"
        log_dir = f"/home/agent/logging/docker_run/{run_id}-{task_id}"

        # Build docker run command with mounted volume
        # Quote env var values to handle special characters (URLs, IDs, etc.)
        # Strip newlines and extra whitespace from values to avoid breaking the command
        env_flags = ""
        if env_vars:
            cleaned_vars = {
                k: v.replace("\n", "").replace("  ", "") for k, v in env_vars.items()
            }
            env_flags = " ".join([f'-e {k}="{v}"' for k, v in cleaned_vars.items()])

        # Run docker in background, redirect all output to log directory
        docker_cmd = (
            f"mkdir -p {log_dir} && "
            f"nohup docker run --rm -v {log_dir}:/workspace/logs {env_flags} {image_name} "
            f"> {log_dir}/docker_output.log 2>&1 &"
        )

        # Run Docker via SSH (spawns background process on VM)
        result = self.run_command(docker_cmd, warn=True)
        if result.exited != 0:
            logger.error(f"Failed to start Docker on {self.name}: {result.stderr}")
        else:
            logger.info(f"Docker started on VM {self.name} (logs in: {log_dir}/)")

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

    def run_command(self, command: str, **kwargs) -> Result:
        """Run a command on the remote VM via SSH.

        Args:
            command: Shell command to run on the remote VM
            **kwargs: Additional arguments to pass to Connection.run()

        Returns:
            Result object from fabric
        """
        logger.info(f"Running command on {self.name}: {command}")
        connection = Connection(
            host=self.public_ip,
            user="agent",
            connect_kwargs={"key_filename": self.ssh_private_key_path},
        )
        result = connection.run(command, **kwargs)
        logger.info(f"Command completed with exit code {result.exited}")
        return result

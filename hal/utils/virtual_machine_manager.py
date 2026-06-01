from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.identity import DefaultAzureCredential
import paramiko
import os
import shlex
import tarfile
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from dotenv import dotenv_values
from .vm.azure_virtual_machine import AzureVirtualMachine

# Mount names for core_agent: used only under VM_AGENT_HOME/environment/ (e.g. environment/data, environment/code, environment/results from task payload).
VM_AGENT_HOME = "/home/agent"
VM_ENVIRONMENT_MOUNT_NAMES = ("data", "code", "results")

RUN_AGENT_SCRIPT_PATH = Path(__file__).resolve().parent / "vm" / "run_agent.py"


def _vm_resolve_env_var_from_host(name: str) -> str | None:
    """Resolve a secret from the host process env or local .env (for VM payload files)."""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    environment_file = Path.cwd() / ".env"
    if environment_file.is_file():
        raw_environment_variables = dotenv_values(environment_file).get(name)
        if raw_environment_variables:
            s = str(raw_environment_variables).strip()
            if s:
                return s
    return None


# Injected into run_agent.env so the remote agent sees keys when the host has no .env file
# to copy to the VM (e.g. CI: secrets exist only on the runner environment).
_VM_RUN_AGENT_SECRET_NAMES = (
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHERAI_API_KEY",
    "OPENROUTER_API_KEY",
)

# Set up base logger
_base_logger = logging.getLogger(__name__)


class VMLoggerAdapter(logging.LoggerAdapter):
    """Prefix all messages in the file with the vm_name."""

    def process(self, msg, kwargs):
        vm_name = self.extra.get("vm_name", "unknown")
        return f"VM Manager {vm_name}: {msg}", kwargs


def _get_logger(vm_name: str) -> logging.LoggerAdapter:
    """Get a logger adapter that prefixes messages with the VM name."""
    return VMLoggerAdapter(_base_logger, {"vm_name": vm_name})


class VirtualMachineManager:
    """
    Manages virtual machine operations for agent execution.

    This class provides stateless methods for creating, managing, and deleting Azure VMs.
    Each method operates on a specific VM identified by vm_name parameter.

    Thread-safe for concurrent operations on different VMs when using the same instance.
    All methods require vm_name as the first parameter to avoid state conflicts.

    Environment Variables Required:
        - AZURE_SUBSCRIPTION_ID: Azure subscription ID
        - AZURE_RESOURCE_GROUP_NAME: Resource group for VMs
        - AZURE_LOCATION: Azure region (e.g., 'eastus')
        - SSH_PRIVATE_KEY_PATH: Path to SSH private key file
        - SSH_PUBLIC_KEY_PATH: Path to SSH public key file
        - NETWORK_SECURITY_GROUP_NAME: Network security group name
    """

    def __init__(self):
        # Load required environment variables
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group_name = os.getenv("AZURE_RESOURCE_GROUP_NAME")
        self.location = os.getenv("AZURE_LOCATION")
        self.ssh_private_key_path = os.getenv("SSH_PRIVATE_KEY_PATH")
        self.ssh_public_key_path = os.getenv("SSH_PUBLIC_KEY_PATH")
        self.network_security_group_name = os.getenv("NETWORK_SECURITY_GROUP_NAME")

        # Validate all required environment variables
        missing_vars = []
        if not self.subscription_id:
            missing_vars.append("AZURE_SUBSCRIPTION_ID")
        if not self.resource_group_name:
            missing_vars.append("AZURE_RESOURCE_GROUP_NAME")
        if not self.location:
            missing_vars.append("AZURE_LOCATION")
        if not self.ssh_private_key_path:
            missing_vars.append("SSH_PRIVATE_KEY_PATH")
        if not self.ssh_public_key_path:
            missing_vars.append("SSH_PUBLIC_KEY_PATH")
        if not self.network_security_group_name:
            missing_vars.append("NETWORK_SECURITY_GROUP_NAME")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please set them in your .env file or environment."
            )

        # Validate SSH key files exist
        if not os.path.exists(self.ssh_private_key_path):
            raise FileNotFoundError(
                f"SSH private key not found at: {self.ssh_private_key_path}. "
                f"Please ensure SSH_PRIVATE_KEY_PATH points to a valid private key file."
            )
        if not os.path.exists(self.ssh_public_key_path):
            raise FileNotFoundError(
                f"SSH public key not found at: {self.ssh_public_key_path}. "
                f"Please ensure SSH_PUBLIC_KEY_PATH points to a valid public key file."
            )

        # Read SSH public key
        with open(self.ssh_public_key_path, "r") as f:
            self.ssh_public_key = f.read().strip()

        # Calculate NSG ID
        self.nsg_id = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group_name}/providers/Microsoft.Network/networkSecurityGroups/{self.network_security_group_name}"

        # Initialize Azure clients (for backwards compatibility with existing code)
        self.credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(
            self.credential, self.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential, self.subscription_id
        )

        # Store created VMs for tracking
        self._vms = {}

    @contextmanager
    def _get_sftp_client(
        self,
        vm_name,
        network_client,
        resource_group_name,
    ):
        """
        Context manager for SFTP client that automatically handles connection and cleanup.

        Args:
            vm_name: Name of the VM to connect to
            network_client: Azure network client
            resource_group_name: Azure resource group name

        Usage:
            with self._get_sftp_client(vm_name, network_client, rg_name) as (sftp, ssh):
                sftp.put(local_file, remote_file)
        """
        logger = _get_logger(vm_name)
        ssh_client = None
        sftp_client = None

        try:
            # Get the public IP address of the VM
            public_ip_address = network_client.public_ip_addresses.get(
                resource_group_name, f"{vm_name}-public-ip"
            ).ip_address

            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the VM using SSH (key_filename lets Paramiko auto-detect RSA/Ed25519/ECDSA)
            ssh_client.connect(
                hostname=public_ip_address,
                username="agent",
                key_filename=self.ssh_private_key_path,
            )

            # Create SFTP client
            sftp_client = ssh_client.open_sftp()

            yield sftp_client, ssh_client

        finally:
            # Close connections
            if sftp_client:
                try:
                    sftp_client.close()
                except Exception as e:
                    logger.error(f"Error closing SFTP client: {e}")

            if ssh_client:
                try:
                    ssh_client.close()
                except Exception as e:
                    logger.error(f"Error closing SSH client: {e}")

    def create_virtual_machine_by_name(
        self, vm_name, has_gpu: bool = False, setup_timeout: int = 0
    ):
        """Create a standard Azure VM without GPU.

        Args:
            vm_name: Name of the VM.
            has_gpu: Whether the VM should have a GPU.
            setup_timeout: Seconds to wait for startup script (passed from
                VirtualMachineRunner.task_timeout).
        """
        logger = _get_logger(vm_name)
        if has_gpu:
            logger.info(f"Creating virtual machine {vm_name} with a GPU")
        else:
            logger.info(f"Creating virtual machine {vm_name} with *no* GPU")

        # Create VM using new AzureVirtualMachine class
        vm = AzureVirtualMachine(
            name=vm_name,
            resource_group=self.resource_group_name,
            location=self.location,
            subscription_id=self.subscription_id,
            nsg_id=self.nsg_id,
            ssh_public_key=self.ssh_public_key,
            gpu=has_gpu,
            timeout=setup_timeout,
        )

        # Store for tracking
        self._vms[vm_name] = vm

        if has_gpu:
            logger.info(f"Created virtual machine {vm_name} with GPU")
        else:
            logger.info(f"Created virtual machine {vm_name} with *no* GPU")
        return vm

    def delete_virtual_machine_by_name(self, vm_name):
        """Delete an Azure VM and all associated resources."""
        # Use the AzureVirtualMachine delete method if we have it
        if vm_name in self._vms:
            self._vms[vm_name].delete()
            del self._vms[vm_name]

    def compress_and_copy_files_to_vm(self, vm_name, source_directory):
        """Copy files from a local directory to the VM."""
        logger = _get_logger(vm_name)
        try:
            # Compress the source directory
            source_directory = os.path.abspath(source_directory)
            tar_file_path = f"{source_directory}.tar.gz"

            logger.info(
                f"Creating tar archive from {source_directory} in {tar_file_path}"
            )
            with tarfile.open(tar_file_path, "w:gz") as tar:
                tar.add(source_directory, arcname=os.path.basename(source_directory))

            tar_size = os.path.getsize(tar_file_path)

            # Copy the compressed file to the VM
            remote_tar_file_path = f"/home/agent/{os.path.basename(tar_file_path)}"
            with self._get_sftp_client(
                vm_name,
                self.network_client,
                self.resource_group_name,
            ) as (sftp_client, ssh_client):
                logger.info(f"Uploading {tar_size} bytes")
                sftp_client.put(tar_file_path, remote_tar_file_path)

                # Extract the compressed file on the VM
                logger.info("Extracting files on the VM")
                _, stdout, stderr = ssh_client.exec_command(
                    f"tar -xzf {remote_tar_file_path} --strip-components=1 -C /home/agent"
                )

                # Block until the tar command completes and check for errors
                exit_status = stdout.channel.recv_exit_status()
                stderr_text = stderr.read().decode()

                if exit_status != 0:
                    raise Exception(
                        f"Tar extraction failed with exit status {exit_status}. stderr: {stderr_text}"
                    )

                if stderr_text:
                    logger.warning(f"Warning during tar extraction: {stderr_text}")

                logger.info(f"Successfully copied files from {source_directory}")

                # Remove the compressed file from the VM and the local machine
                sftp_client.remove(remote_tar_file_path)
                os.remove(tar_file_path)

        except Exception as e:
            logger.error(f"Error copying files: {e}")
            raise

    def copy_files_from_vm(
        self,
        vm_name,
        destination_directory,
        *,
        download_environment: bool = True,
    ):
        """Copy files from the VM to local directory.

        When download_environment is False, omits ``/home/agent/environment`` from
        the archive (task data/code/results) to speed up SFTP; ``output.json`` and
        other home-directory files are still included.
        """
        logger = _get_logger(vm_name)
        with self._get_sftp_client(
            vm_name,
            self.network_client,
            self.resource_group_name,
        ) as (sftp_client, ssh_client):
            # Remove ./miniconda3 directory from the VM
            _, stdout, _ = ssh_client.exec_command("rm -rf /home/agent/miniconda3")
            for _ in stdout:
                pass  # Block until the rm command completes

            # Compress all files in the home directory on the VM
            remote_tar_file_path = (
                f"/home/agent/{os.path.basename(destination_directory)}_back.tar.gz"
            )
            remote_home_directory = "/home/agent"
            exclude_command_line_argument_string = ""
            if not download_environment:
                logger.info(
                    "Excluding environment/ from VM results archive (faster download)"
                )
                exclude_command_line_argument_string = "--exclude=environment "
            quoted_tar = shlex.quote(remote_tar_file_path)
            _, stdout, _ = ssh_client.exec_command(
                f"tar {exclude_command_line_argument_string}-czf {quoted_tar} -C {remote_home_directory} ."
            )
            for _ in stdout:
                pass  # Block until the tar command completes

            # Copy the compressed file from the VM
            sftp_client.get(remote_tar_file_path, f"{destination_directory}.tar.gz")

            # Extract the compressed file on the local machine
            with tarfile.open(f"{destination_directory}.tar.gz", "r:gz") as tar:
                tar.extractall(destination_directory)

            # Remove the compressed file from the VM and the local machine
            # sftp_client.remove(remote_tar_file_path)
            os.remove(f"{destination_directory}.tar.gz")

    def check_task_completion(self, vm_name: str) -> bool:
        """
        Check if task is complete by checking for output.json file.

        :param self: the virtual machine manager
        :param vm_name: the virtual machine whose task we want to check
        :type vm_name: str
        :return: whether or not the task is complete
        :rtype: bool
        """
        task_completed_filepath = "/home/agent/output.json"
        vm = self._vms[vm_name]
        return vm.check_for_file_presence_by_path(task_completed_filepath)

    def run_agent_on_virtual_machine(
        self,
        vm_name,
        agent_function,
        task_id,
        input_data,
        agent_args,
        run_id,
        log_dir,
        benchmark,
    ):
        """
        Run agent on VM with improved monitoring and error handling.
        """
        logger = _get_logger(vm_name)

        def copy_env_and_run_setup_script(
            vm_name: str,
            log_dir: str,
            benchmark,
            task_id: str,
        ) -> None:
            """
            Set up the VM environment using uv and a setup script.
            """
            try:
                with self._get_sftp_client(
                    vm_name,
                    self.network_client,
                    self.resource_group_name,
                ) as (sftp_client, ssh_client):
                    # Copy .env file to VM first
                    if os.path.exists(".env"):
                        logger.info("Copying .env file")
                        sftp_client.put(".env", "/home/agent/.env")

                    # Copy setup script to VM
                    setup_script_path = os.path.join(
                        os.path.dirname(__file__), "setup_vm.sh"
                    )
                    remote_setup_path = "/home/agent/setup_vm.sh"
                    sftp_client.put(setup_script_path, remote_setup_path)

                    # Make setup script executable
                    ssh_client.exec_command(f"chmod +x {remote_setup_path}")

                    # Run setup script with sudo
                    logger.info("Running setup_vm.sh on remote")
                    _, stdout, stderr = ssh_client.exec_command(
                        f"sudo bash {remote_setup_path}"
                    )

                    # Create log directory if it doesn't exist and write log file
                    os.makedirs(log_dir, exist_ok=True)
                    with open(f"{log_dir}/setup_vm_log_{task_id}.log", "w") as f:
                        f.write(stdout.read().decode())
                        f.write(stderr.read().decode())

                    # Run setup script if it exists
                    if benchmark and benchmark.setup_script:
                        setup_script = os.path.join(benchmark.setup_script)
                        if os.path.exists(setup_script):
                            logger.info("Running setup script")
                            try:
                                cmd = """
                                source /home/agent/miniconda3/etc/profile.d/conda.sh && \
                                cd /home/agent && \
                                bash setup_script.sh
                                """
                                _, stdout, stderr = ssh_client.exec_command(cmd)
                                with open(
                                    f"{log_dir}/setup_script_log_{task_id}.log", "w"
                                ) as f:
                                    f.write(stdout.read().decode())
                                    f.write(stderr.read().decode())
                            except Exception as e:
                                logger.error(f"Error running setup script: {e}")

            except Exception as e:
                logger.error(f"Error setting up VM environment: {e}")
                raise

        try:
            # Setup conda environment if it exists
            copy_env_and_run_setup_script(
                vm_name,
                log_dir,
                benchmark,
                task_id,
            )

            with self._get_sftp_client(
                vm_name,
                self.network_client,
                self.resource_group_name,
            ) as (sftp_client, ssh_client):
                # Write input data and agent args to files
                with sftp_client.open("/home/agent/input.json", "w") as f:
                    f.write(json.dumps({task_id: input_data}))
                with sftp_client.open("/home/agent/agent_args.json", "w") as f:
                    f.write(json.dumps(agent_args))

                # Write run-specific env vars for static run_agent.py
                run_agent_env = f"RUN_ID={run_id}\nAGENT_FUNCTION={agent_function}\nTASK_ID={task_id}\n"
                for secret_name in _VM_RUN_AGENT_SECRET_NAMES:
                    secret_value = _vm_resolve_env_var_from_host(secret_name)
                    if secret_value:
                        run_agent_env += f"{secret_name}={secret_value}\n"
                with sftp_client.open("/home/agent/run_agent.env", "w") as f:
                    f.write(run_agent_env)

                script_path = "/home/agent/run_agent.py"
                ssh_client.exec_command(f"chmod +x {script_path}")

                # Construct command to run script
                cmd = f"source /home/agent/miniconda3/etc/profile.d/conda.sh && conda activate agent_env && python {script_path} > agent_trace.log 2>&1"

                # Execute script
                logger.info("Running agent")
                _, stdout, stderr = ssh_client.exec_command(cmd)

                # Close the channel to prevent hanging
                stdout.channel.close()
                stderr.channel.close()

        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise

    def get_agent_trace(self, vm_name):
        """
        Fetch the current agent trace log from a VM.

        Args:
            vm_name: Name of the VM to fetch logs from

        Returns:
            str: Contents of the agent trace log, or None if not available
        """
        logger = _get_logger(vm_name)
        try:
            with self._get_sftp_client(
                vm_name,
                self.network_client,
                self.resource_group_name,
            ) as (sftp_client, _):
                # Try to read the agent trace file
                try:
                    with sftp_client.open("/home/agent/agent_trace.log") as f:
                        return f.read().decode("utf-8")
                except FileNotFoundError:
                    return None

        except Exception as e:
            logger.error(f"Error fetching agent trace: {e}")
            return None

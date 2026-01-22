from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.identity import DefaultAzureCredential
import paramiko
import os
import tarfile
import json
from tenacity import retry, stop_after_attempt, wait_exponential


# Define retry decorator with tenacity
def get_retry_decorator(max_attempts=3, initial_wait=1, max_wait=30):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        reraise=True,
    )


class VirtualMachineManager:
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group_name = os.getenv("AZURE_RESOURCE_GROUP_NAME")
        self.location = os.getenv("AZURE_LOCATION")
        self.credential = DefaultAzureCredential()
        self.compute_client = ComputeManagementClient(
            self.credential, self.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential, self.subscription_id
        )
        self.resource_client = ResourceManagementClient(
            self.credential, self.subscription_id
        )

    @get_retry_decorator()
    def create_vm(
        self,
        vm_name,
        username,
        ssh_public_key_path,
        network_security_group_name,
        vm_size="Standard_E2as_v5",
        image_reference=None,
        disk_size=80,
    ):
        # Create a virtual network and subnet
        vnet_name = f"{vm_name}-vnet"
        subnet_name = f"{vm_name}-subnet"
        vnet = self.network_client.virtual_networks.begin_create_or_update(
            self.resource_group_name,
            vnet_name,
            {
                "location": self.location,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]},
                "subnets": [{"name": subnet_name, "address_prefix": "10.0.0.0/24"}],
            },
        ).result()
        subnet = vnet.subnets[0]

        # Create a public IP address
        public_ip_name = f"{vm_name}-public-ip"
        public_ip = self.network_client.public_ip_addresses.begin_create_or_update(
            self.resource_group_name,
            public_ip_name,
            {
                "location": self.location,
                "sku": {"name": "Standard"},
                "public_ip_allocation_method": "Static",
            },
        ).result()

        # Get the existing network security group
        network_security_group = self.network_client.network_security_groups.get(
            self.resource_group_name, network_security_group_name
        )

        # Create a network interface
        nic_name = f"{vm_name}-nic"
        nic = self.network_client.network_interfaces.begin_create_or_update(
            self.resource_group_name,
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
                "network_security_group": {"id": network_security_group.id},
            },
        ).result()

        # Read the SSH public key from the specified file
        with open(ssh_public_key_path, "r") as file:
            ssh_public_key = file.read().strip()

        # Define the VM configuration
        if image_reference is None:
            image_reference = {
                "publisher": "Canonical",
                "offer": "0001-com-ubuntu-server-focal",
                "sku": "20_04-lts-gen2",
                "version": "latest",
            }

        vm_parameters = {
            "location": self.location,
            "storage_profile": {
                "image_reference": image_reference,
                "os_disk": {"createOption": "FromImage", "diskSizeGB": disk_size},
            },
            "hardware_profile": {"vm_size": vm_size},
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": username,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{username}/.ssh/authorized_keys",
                                "key_data": ssh_public_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id}]},
        }

        # Create the VM
        vm = self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group_name, vm_name, vm_parameters
        ).result()

        return vm

    @get_retry_decorator()
    def create_gpu_vm(
        self,
        vm_name,
        username,
        ssh_public_key_path,
        network_security_group_name,
        vm_size="Standard_NC4as_T4_v3",
        image_reference=None,
        disk_size=80,
    ):
        # Create a virtual network and subnet
        vnet_name = f"{vm_name}-vnet"
        subnet_name = f"{vm_name}-subnet"
        vnet = self.network_client.virtual_networks.begin_create_or_update(
            self.resource_group_name,
            vnet_name,
            {
                "location": self.location,
                "address_space": {"address_prefixes": ["10.0.0.0/16"]},
                "subnets": [{"name": subnet_name, "address_prefix": "10.0.0.0/24"}],
            },
        ).result()
        subnet = vnet.subnets[0]

        # Create a public IP address
        public_ip_name = f"{vm_name}-public-ip"
        public_ip = self.network_client.public_ip_addresses.begin_create_or_update(
            self.resource_group_name,
            public_ip_name,
            {
                "location": self.location,
                "sku": {"name": "Standard"},
                "public_ip_allocation_method": "Static",
            },
        ).result()

        # Get the existing network security group
        network_security_group = self.network_client.network_security_groups.get(
            self.resource_group_name, network_security_group_name
        )

        # Create a network interface
        nic_name = f"{vm_name}-nic"
        nic = self.network_client.network_interfaces.begin_create_or_update(
            self.resource_group_name,
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
                "network_security_group": {"id": network_security_group.id},
            },
        ).result()

        # Read the SSH public key from the specified file
        if not ssh_public_key_path:
            raise ValueError(
                "SSH public key path is empty. Check the SSH_PUBLIC_KEY_PATH environment variable and try again."
            )

        try:
            with open(ssh_public_key_path, "r") as file:
                ssh_public_key = file.read().strip()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"The SSH public key file at '{ssh_public_key_path}' cannot be found. Check the SSH_PUBLIC_KEY_PATH environment variable and try again."
            ) from e

        # Define the GPU VM configuration
        if image_reference is None:
            image_reference = {
                "publisher": "Canonical",
                "offer": "0001-com-ubuntu-server-focal",
                "sku": "20_04-lts-gen2",
                "version": "latest",
            }

        vm_parameters = {
            "location": self.location,
            "storage_profile": {
                "image_reference": image_reference,
                "os_disk": {"createOption": "FromImage", "diskSizeGB": disk_size},
            },
            "hardware_profile": {"vm_size": vm_size},
            "os_profile": {
                "computer_name": vm_name,
                "admin_username": username,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{username}/.ssh/authorized_keys",
                                "key_data": ssh_public_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {"network_interfaces": [{"id": nic.id}]},
            "uefi_settings": {"secure_boot_enabled": False},
        }

        # Create the GPU VM
        vm = self.compute_client.virtual_machines.begin_create_or_update(
            self.resource_group_name, vm_name, vm_parameters
        ).result()

        # Define the NVIDIA GPU driver extension configuration
        extension_name = "NvidiaGpuDriverLinux"
        extension_publisher = "Microsoft.HpcCompute"
        extension_type = "NvidiaGpuDriverLinux"
        type_handler_version = "1.9"

        extension_parameters = {
            "location": self.location,
            "publisher": extension_publisher,
            "type_properties_type": extension_type,
            "type_handler_version": type_handler_version,
            "auto_upgrade_minor_version": True,
            "settings": {},
        }

        # Add the NVIDIA GPU driver extension to the VM
        self.compute_client.virtual_machine_extensions.begin_create_or_update(
            self.resource_group_name, vm_name, extension_name, extension_parameters
        ).result()

        return vm

    def delete_vm(self, vm_name):
        # Get the VM
        vm = self.compute_client.virtual_machines.get(self.resource_group_name, vm_name)

        # Delete the VM
        self.compute_client.virtual_machines.begin_delete(
            self.resource_group_name, vm_name
        ).result()

        # Delete the associated disks
        for disk in vm.storage_profile.data_disks:
            self.compute_client.disks.begin_delete(
                self.resource_group_name, disk.name
            ).result()

        # Delete the OS disk
        os_disk_name = vm.storage_profile.os_disk.name
        self.compute_client.disks.begin_delete(
            self.resource_group_name, os_disk_name
        ).result()

        # Delete the network interface
        nic_name = f"{vm_name}-nic"
        self.network_client.network_interfaces.begin_delete(
            self.resource_group_name, nic_name
        ).result()

        # Delete the public IP address
        public_ip_name = f"{vm_name}-public-ip"
        self.network_client.public_ip_addresses.begin_delete(
            self.resource_group_name, public_ip_name
        ).result()

        # Delete the virtual network (if not used by other resources)
        vnet_name = f"{vm_name}-vnet"
        try:
            self.network_client.virtual_networks.begin_delete(
                self.resource_group_name, vnet_name
            ).result()
        except Exception as e:
            print(f"Failed to delete virtual network {vnet_name}: {str(e)}")

    @get_retry_decorator()
    def copy_files_to_vm(
        self, source_directory, vm_name, username, ssh_private_key_path
    ):
        # Get the public IP address of the VM
        vm = self.compute_client.virtual_machines.get(self.resource_group_name, vm_name)
        public_ip_address = self.network_client.public_ip_addresses.get(
            self.resource_group_name, f"{vm_name}-public-ip"
        ).ip_address

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the SSH private key
        ssh_private_key = paramiko.RSAKey.from_private_key_file(ssh_private_key_path)

        # Connect to the VM using SSH
        ssh_client.connect(
            hostname=public_ip_address, username=username, pkey=ssh_private_key
        )

        # Create an SFTP client
        sftp_client = ssh_client.open_sftp()

        # Copy files from the source directory to the VM
        # Compress the source directory
        source_directory = os.path.abspath(source_directory)
        tar_file_path = f"{source_directory}.tar.gz"
        with tarfile.open(tar_file_path, "w:gz") as tar:
            tar.add(source_directory, arcname=os.path.basename(source_directory))

        # Copy the compressed file to the VM
        remote_tar_file_path = f"/home/{username}/{os.path.basename(tar_file_path)}"
        sftp_client.put(tar_file_path, remote_tar_file_path)

        # Extract the compressed file on the VM
        _, stdout, _ = ssh_client.exec_command(
            f"tar -xzf {remote_tar_file_path} --strip-components=1 -C /home/{username}"
        )
        for _ in stdout:
            pass  # Block until the tar command completes

        # Remove the compressed file from the VM and the local machine
        sftp_client.remove(remote_tar_file_path)
        os.remove(tar_file_path)

        # Close the SFTP client and SSH connection
        sftp_client.close()
        ssh_client.close()

    @get_retry_decorator()
    def copy_files_from_vm(
        self, vm_name, username, ssh_private_key_path, destination_directory
    ):
        # Get the public IP address of the VM
        vm = self.compute_client.virtual_machines.get(self.resource_group_name, vm_name)
        public_ip_address = self.network_client.public_ip_addresses.get(
            self.resource_group_name, f"{vm_name}-public-ip"
        ).ip_address

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the SSH private key
        ssh_private_key = paramiko.RSAKey.from_private_key_file(ssh_private_key_path)

        # Connect to the VM using SSH
        ssh_client.connect(
            hostname=public_ip_address, username=username, pkey=ssh_private_key
        )

        # Create an SFTP client
        sftp_client = ssh_client.open_sftp()

        # Remove ./miniconda3 directory from the VM
        _, stdout, _ = ssh_client.exec_command(f"rm -rf /home/{username}/miniconda3")
        for _ in stdout:
            pass  # Block until the rm command completes

        # Compress all files in the home directory on the VM
        remote_tar_file_path = (
            f"/home/{username}/{os.path.basename(destination_directory)}_back.tar.gz"
        )
        remote_home_directory = f"/home/{username}"
        _, stdout, _ = ssh_client.exec_command(
            f"tar -czf {remote_tar_file_path} -C {remote_home_directory} ."
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

        # Close the SFTP client and SSH connection
        sftp_client.close()
        ssh_client.close()

    @get_retry_decorator(max_attempts=2, initial_wait=5)
    def check_task_completion(
        self,
        vm_name,
        username,
        ssh_private_key_path,
        task_completed_filename="output.json",
        agent_trace_filename="agent_trace.log",
    ):
        # Get the public IP address of the VM
        vm = self.compute_client.virtual_machines.get(self.resource_group_name, vm_name)
        public_ip_address = self.network_client.public_ip_addresses.get(
            self.resource_group_name, f"{vm_name}-public-ip"
        ).ip_address

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load the SSH private key
        ssh_private_key = paramiko.RSAKey.from_private_key_file(ssh_private_key_path)

        # Connect to the VM using SSH
        ssh_client.connect(
            hostname=public_ip_address,
            username=username,
            pkey=ssh_private_key,
            timeout=15,
        )

        # Create an SFTP client
        sftp_client = ssh_client.open_sftp()

        # Check for task completion via existence of output.json
        task_completed_filepath = f"/home/{username}/{task_completed_filename}"
        agent_trace_filepath = f"/home/{username}/{agent_trace_filename}"

        try:
            with sftp_client.open(task_completed_filepath) as file:
                result = json.loads(file.read().decode("utf-8"))
        except FileNotFoundError:
            result = None  # output.json does not exist

        # Close the SFTP client and SSH connection
        sftp_client.close()
        ssh_client.close()

        return result

    @get_retry_decorator()
    def setup_vm_environment(
        self,
        vm_name: str,
        username: str,
        ssh_private_key_path: str,
        agent_dir: str,
        log_dir: str,
        benchmark,
        task_id: str,
    ) -> None:
        """
        Set up the VM environment using uv and a setup script.
        """
        try:
            # Get the public IP address of the VM
            vm = self.compute_client.virtual_machines.get(
                self.resource_group_name, vm_name
            )
            public_ip_address = self.network_client.public_ip_addresses.get(
                self.resource_group_name, f"{vm_name}-public-ip"
            ).ip_address

            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_private_key = paramiko.RSAKey.from_private_key_file(
                ssh_private_key_path
            )
            ssh_client.connect(
                hostname=public_ip_address, username=username, pkey=ssh_private_key
            )

            # Create SFTP client
            sftp_client = ssh_client.open_sftp()

            try:
                # Copy .env file to VM first
                if os.path.exists(".env"):
                    print(f"Copying .env file to VM {vm_name}")
                    sftp_client.put(".env", f"/home/{username}/.env")

                # Copy setup script to VM
                setup_script_path = os.path.join(
                    os.path.dirname(__file__), "setup_vm.sh"
                )
                remote_setup_path = f"/home/{username}/setup_vm.sh"
                sftp_client.put(setup_script_path, remote_setup_path)

                # Make setup script executable
                ssh_client.exec_command(f"chmod +x {remote_setup_path}")

                # Run setup script with sudo (passing username as argument)
                print(f"Setting up environment on VM {vm_name}")
                _, stdout, stderr = ssh_client.exec_command(
                    f"sudo bash {remote_setup_path} {username}"
                )

                # Create log file
                with open(f"{log_dir}/setup_vm_log_{task_id}.log", "w") as f:
                    f.write(stdout.read().decode())
                    f.write(stderr.read().decode())

                # Run setup script if it exists
                if benchmark and benchmark.setup_script:
                    setup_script = os.path.join(benchmark.setup_script)
                    if os.path.exists(setup_script):
                        print(f"Running setup script on VM {vm_name}")
                        try:
                            cmd = f"""
                            source /home/{username}/miniconda3/etc/profile.d/conda.sh && \
                            cd /home/{username} && \
                            bash setup_script.sh
                            """
                            _, stdout, stderr = ssh_client.exec_command(cmd)
                            with open(
                                f"{log_dir}/setup_script_log_{task_id}.log", "w"
                            ) as f:
                                f.write(stdout.read().decode())
                                f.write(stderr.read().decode())
                        except Exception as e:
                            print(f"Error running setup script on VM {vm_name}: {e}")

            finally:
                sftp_client.close()
                ssh_client.close()

        except Exception as e:
            print(f"Error setting up VM environment: {e}")
            raise

    def run_agent_on_vm(
        self,
        agent_function,
        vm_name,
        task_id,
        input_data,
        agent_args,
        agent_dir,
        run_id,
        username,
        log_dir,
        ssh_private_key_path,
        benchmark,
        timeout=7200,
    ):
        """
        Run agent on VM with improved monitoring and error handling.
        """
        try:
            # Setup conda environment if it exists
            self.setup_vm_environment(
                vm_name,
                username,
                ssh_private_key_path,
                agent_dir,
                log_dir,
                benchmark,
                task_id,
            )

            # Get the public IP address of the VM
            public_ip_address = self.network_client.public_ip_addresses.get(
                self.resource_group_name, f"{vm_name}-public-ip"
            ).ip_address

            # Create SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_private_key = paramiko.RSAKey.from_private_key_file(
                ssh_private_key_path
            )
            ssh_client.connect(
                hostname=public_ip_address, username=username, pkey=ssh_private_key
            )

            # Create SFTP client
            sftp_client = ssh_client.open_sftp()

            try:
                # Write input data and agent args to files
                with sftp_client.open(f"/home/{username}/input.json", "w") as f:
                    f.write(json.dumps({task_id: input_data}))
                with sftp_client.open(f"/home/{username}/agent_args.json", "w") as f:
                    f.write(json.dumps(agent_args))

                # Create Python script for agent execution
                script_content = f'''#!/usr/bin/env python3
import os
import json
import importlib.util
import weave
import traceback

try:
    weave.init("{run_id}")
    
    # Load input data
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent args
    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)
    
    # Load the agent module
    module_name = "{agent_function.rsplit(".", 1)[0]}"
    function_name = "{agent_function.rsplit(".", 1)[1]}"
    
    spec = importlib.util.spec_from_file_location(module_name, os.path.join("/home/{username}", module_name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    agent = getattr(module, function_name)
    
    # Run agent with weave task_id attribute
    with weave.attributes({{"weave_task_id": "{task_id}"}}):
        result = agent(input_data, **agent_args)
    
    # Save result
    with open("output.json", "w") as f:
        json.dump(result, f)

except Exception as e:
    with open("error.log", "w") as f:
        f.write(f"ERROR: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    raise
'''

                # Write script to VM
                script_path = f"/home/{username}/run_agent.py"
                with sftp_client.open(script_path, "w") as f:
                    f.write(script_content)

                # Make script executable
                ssh_client.exec_command(f"chmod +x {script_path}")

                # Construct command to run script
                cmd = f"source /home/{username}/init_conda.sh && conda activate agent_env && python {script_path} > agent_trace.log 2>&1"

                # Execute script
                print(f"Running agent on VM {vm_name}")
                stdin, stdout, stderr = ssh_client.exec_command(cmd)

                # Close the channel to prevent hanging
                stdout.channel.close()
                stderr.channel.close()

            finally:
                sftp_client.close()
                ssh_client.close()

        except Exception as e:
            print(f"Error running agent on VM {vm_name}: {e}")
            raise

    @get_retry_decorator(max_attempts=2, initial_wait=5)
    def get_agent_trace(self, vm_name, username, ssh_private_key_path):
        """
        Fetch the current agent trace log from a VM.

        Returns:
            str: Contents of the agent trace log, or None if not available
        """
        try:
            # Get the public IP address of the VM
            vm = self.compute_client.virtual_machines.get(
                self.resource_group_name, vm_name
            )
            public_ip_address = self.network_client.public_ip_addresses.get(
                self.resource_group_name, f"{vm_name}-public-ip"
            ).ip_address

            # Create an SSH client
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Load the SSH private key
            ssh_private_key = paramiko.RSAKey.from_private_key_file(
                ssh_private_key_path
            )

            # Connect to the VM using SSH
            ssh_client.connect(
                hostname=public_ip_address,
                username=username,
                pkey=ssh_private_key,
                timeout=15,
            )

            # Create an SFTP client
            sftp_client = ssh_client.open_sftp()

            # Try to read the agent trace file
            try:
                with sftp_client.open(f"/home/{username}/agent_trace.log") as f:
                    return f.read().decode("utf-8")
            except FileNotFoundError:
                return None
            finally:
                sftp_client.close()
                ssh_client.close()

        except Exception as e:
            print(f"Error fetching agent trace from {vm_name}: {e}")
            return None


if __name__ == "__main__":
    vm_manager = VirtualMachineManager()
    vm_manager.create_vm(
        "test-vm", "agent", SSH_PUBLIC_KEY_PATH, NETWORK_SECURITY_GROUP_NAME
    )
    vm_manager.copy_files_to_vm("agent", "test-vm", "agent", SSH_PRIVATE_KEY_PATH)
    vm_manager.copy_files_from_vm(
        "test-vm", "agent", SSH_PRIVATE_KEY_PATH, "/home/azureuser/agent-back"
    )
    vm_manager.delete_vm("test-vm")

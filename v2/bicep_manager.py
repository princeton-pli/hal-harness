"""Minimal Bicep deployment manager for Azure VMs."""

import subprocess
from pathlib import Path


class BicepManager:
    """Manages Bicep template deployments."""

    def __init__(
        self,
        resource_group: str,
        location: str = "eastus",
        subscription_id: str = "",
        nsg_name: str = "",
    ):
        self.resource_group = resource_group
        self.location = location
        self.subscription_id = subscription_id
        self.nsg_name = nsg_name
        self.template_dir = Path(__file__).parent / "infrastructure"

    def deploy(
        self,
        template_path: str,
        deployment_name: str,
        parameters: dict[str, str],
    ) -> dict:
        """Deploy a Bicep template.

        Args:
            template_path: Path to .bicep file relative to infrastructure/
            deployment_name: Unique name for this deployment
            parameters: Dict of parameter name -> value

        Returns:
            Deployment outputs as dict
        """
        template_file = self.template_dir / template_path

        if not template_file.exists():
            raise FileNotFoundError(f"Template not found: {template_file}")

        # Build parameter arguments
        param_args = []
        for key, value in parameters.items():
            param_args.extend(["--parameters", f"{key}={value}"])

        cmd = [
            "az",
            "deployment",
            "group",
            "create",
            "--resource-group",
            self.resource_group,
            "--name",
            deployment_name,
            "--template-file",
            str(template_file),
            *param_args,
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "deployed", "name": deployment_name}

    def delete(self, deployment_name: str) -> None:
        """Delete a deployment (does NOT delete resources, only deployment record)."""
        cmd = [
            "az",
            "deployment",
            "group",
            "delete",
            "--resource-group",
            self.resource_group,
            "--name",
            deployment_name,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def delete_resource_group(self) -> None:
        """Delete entire resource group and all resources."""
        cmd = [
            "az",
            "group",
            "delete",
            "--name",
            self.resource_group,
            "--yes",
            "--no-wait",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    def deploy_vm(
        self, vm_name: str, ssh_public_key: str, gpu: bool = False
    ) -> dict:
        """Deploy a VM (standard or GPU).

        Args:
            vm_name: Name for the VM
            ssh_public_key: SSH public key content
            gpu: True for GPU VM, False for standard VM

        Returns:
            Deployment info with public IP
        """
        nsg_id = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/networkSecurityGroups/{self.nsg_name}"

        template = "modules/gpu-vm-v2.bicep" if gpu else "modules/standard-vm-v2.bicep"

        result = self.deploy(
            template_path=template,
            deployment_name=f"{vm_name}-deployment",
            parameters={
                "vmName": vm_name,
                "networkSecurityGroupId": nsg_id,
                "sshPublicKey": ssh_public_key,
            },
        )

        # Get public IP
        cmd = [
            "az",
            "deployment",
            "group",
            "show",
            "--resource-group",
            self.resource_group,
            "--name",
            f"{vm_name}-deployment",
            "--query",
            "properties.outputs.publicIpAddress.value",
            "--output",
            "tsv",
        ]
        ip_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        public_ip = ip_result.stdout.strip()

        return {"vm_name": vm_name, "public_ip": public_ip, **result}

    def delete_vm(self, vm_name: str) -> None:
        """Delete a VM and all associated resources.

        Args:
            vm_name: Name of the VM to delete
        """
        resources = [
            ("vm", vm_name),
            ("nic", f"{vm_name}-nic"),
            ("public-ip", f"{vm_name}-public-ip"),
            ("vnet", f"{vm_name}-vnet"),
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

            subprocess.run(cmd, capture_output=True)  # Don't fail on missing resources

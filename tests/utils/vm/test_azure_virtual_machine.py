"""Tests for AzureVirtualMachine class."""

import os
from unittest.mock import Mock, patch

from hal.utils.vm.azure_virtual_machine import AzureVirtualMachine


class TestAzureVirtualMachineCheckForFilePresence:
    """Tests for check_for_file_presence_by_path method."""

    @patch.dict(os.environ, {"SSH_PRIVATE_KEY_PATH": "/path/to/key"})
    @patch("subprocess.run")
    def test_calls_the_right_subprocess_command_when_the_test_file_exists(
        self, mock_run
    ):
        """Test when file exists on VM."""
        # Setup
        mock_run.return_value = Mock(returncode=0)
        vm = self._create_mock_vm()
        vm.public_ip = "1.2.3.4"

        # Execute
        result = vm.check_for_file_presence_by_path("/path/to/file.txt")

        # Assert
        assert result is True
        mock_run.assert_called_once_with(
            [
                "ssh",
                "-i",
                "/path/to/key",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=5",
                "agent@1.2.3.4",
                "test -f /path/to/file.txt",
            ],
            capture_output=True,
        )

    @patch.dict(os.environ, {"SSH_PRIVATE_KEY_PATH": "/path/to/key"})
    @patch("subprocess.run")
    def test_calls_the_right_subprocess_command_when_the_test_file_does_not_exist(
        self, mock_run
    ):
        """Test when file does not exist on VM."""
        # Setup
        mock_run.return_value = Mock(returncode=1)
        vm = self._create_mock_vm()
        vm.public_ip = "1.2.3.4"

        # Execute
        result = vm.check_for_file_presence_by_path("/path/to/missing.txt")

        # Assert
        assert result is False
        mock_run.assert_called_once_with(
            [
                "ssh",
                "-i",
                "/path/to/key",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=5",
                "agent@1.2.3.4",
                "test -f /path/to/missing.txt",
            ],
            capture_output=True,
        )

    @patch.dict(
        os.environ, {"SSH_PRIVATE_KEY_PATH": "/home/user/.ssh/id_rsa_CUSTOMPATH"}
    )
    @patch("subprocess.run")
    def test_pulls_ssh_key_path_from_env(self, mock_run):
        """Test that correct SSH key path from environment is used."""
        # Setup
        mock_run.return_value = Mock(returncode=0)
        vm = self._create_mock_vm()
        vm.public_ip = "10.0.0.1"

        # Execute
        vm.check_for_file_presence_by_path("/test")

        # Assert
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ssh"
        assert call_args[1] == "-i"
        assert call_args[2] == "/home/user/.ssh/id_rsa_CUSTOMPATH"

    def _create_mock_vm(self) -> AzureVirtualMachine:
        """Create a mock AzureVirtualMachine instance without calling __init__."""
        vm = object.__new__(AzureVirtualMachine)
        return vm


class TestAzureVirtualMachineInit:
    """Tests for __init__ method GPU configuration."""

    @patch("hal.utils.vm.azure_virtual_machine.DefaultAzureCredential")
    @patch("hal.utils.vm.azure_virtual_machine.ComputeManagementClient")
    @patch("hal.utils.vm.azure_virtual_machine.NetworkManagementClient")
    @patch.object(AzureVirtualMachine, "_create")
    def test_gpu_enabled_uses_gpu_vm_size(self, mock_create, _, __, ___):
        """Test that gpu=True results in GPU VM size."""
        # Execute
        vm = AzureVirtualMachine(
            name="test-vm",
            resource_group="test-rg",
            location="eastus",
            subscription_id="sub-123",
            nsg_id="nsg-123",
            ssh_public_key="ssh-rsa AAAAB3...",
            gpu=True,
        )

        # Assert
        assert vm.vm_size == "Standard_NC4as_T4_v3"
        assert vm.gpu is True
        mock_create.assert_called_once()

    @patch("hal.utils.vm.azure_virtual_machine.DefaultAzureCredential")
    @patch("hal.utils.vm.azure_virtual_machine.ComputeManagementClient")
    @patch("hal.utils.vm.azure_virtual_machine.NetworkManagementClient")
    @patch.object(AzureVirtualMachine, "_create")
    def test_gpu_disabled_uses_provided_vm_size(self, mock_create, _, __, ___):
        """Test that gpu=False uses the provided VM size."""
        # Execute
        vm = AzureVirtualMachine(
            name="test-vm",
            resource_group="test-rg",
            location="westus",
            subscription_id="sub-456",
            nsg_id="nsg-456",
            ssh_public_key="ssh-rsa AAAAB3...",
            gpu=False,
        )

        # Assert
        assert vm.vm_size == "Standard_E2as_v5"
        assert vm.gpu is False
        mock_create.assert_called_once()

    @patch("hal.utils.vm.azure_virtual_machine.DefaultAzureCredential")
    @patch("hal.utils.vm.azure_virtual_machine.ComputeManagementClient")
    @patch("hal.utils.vm.azure_virtual_machine.NetworkManagementClient")
    @patch.object(AzureVirtualMachine, "_create")
    def test_default_gpu_false_uses_default_vm_size(self, mock_create, _, __, ___):
        """Test that default gpu=False uses the default VM size."""
        # Execute
        vm = AzureVirtualMachine(
            name="test-vm",
            resource_group="test-rg",
            location="centralus",
            subscription_id="sub-789",
            nsg_id="nsg-789",
            ssh_public_key="ssh-rsa AAAAB3...",
            gpu=False,
        )

        # Assert
        assert vm.vm_size == "Standard_E2as_v5"
        mock_create.assert_called_once()

    @patch("hal.utils.vm.azure_virtual_machine.DefaultAzureCredential")
    @patch("hal.utils.vm.azure_virtual_machine.ComputeManagementClient")
    @patch("hal.utils.vm.azure_virtual_machine.NetworkManagementClient")
    @patch.object(AzureVirtualMachine, "_create")
    def test_gpu_true_overrides_vm_size(self, mock_create, _, __, ___):
        """Test that gpu=True overrides any provided vm_size."""
        # Execute
        vm = AzureVirtualMachine(
            name="test-vm",
            resource_group="test-rg",
            location="eastus",
            subscription_id="sub-abc",
            nsg_id="nsg-abc",
            ssh_public_key="ssh-rsa AAAAB3...",
            gpu=True,
        )

        # Assert - GPU size should override the provided size
        assert vm.vm_size == "Standard_NC4as_T4_v3"
        assert vm.gpu is True
        mock_create.assert_called_once()

    @patch("hal.utils.vm.azure_virtual_machine.DefaultAzureCredential")
    @patch("hal.utils.vm.azure_virtual_machine.ComputeManagementClient")
    @patch("hal.utils.vm.azure_virtual_machine.NetworkManagementClient")
    @patch.object(AzureVirtualMachine, "_create")
    def test_stores_all_initialization_parameters(self, mock_create, _, __, ___):
        """Test that all initialization parameters are stored correctly."""
        # Execute
        vm = AzureVirtualMachine(
            name="my-vm",
            resource_group="my-rg",
            location="westeurope",
            subscription_id="sub-xyz",
            nsg_id="nsg-xyz",
            ssh_public_key="ssh-rsa AAAAB3NzaC1...",
            gpu=False,
        )

        # Assert
        assert vm.name == "my-vm"
        assert vm.resource_group == "my-rg"
        assert vm.location == "westeurope"
        assert vm.gpu is False
        assert vm.nsg_id == "nsg-xyz"
        assert vm.ssh_public_key == "ssh-rsa AAAAB3NzaC1..."
        assert vm.public_ip is None  # Not set until _create runs
        mock_create.assert_called_once()

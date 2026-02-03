// Bicep module for creating a GPU-enabled VM
// This module creates all networking resources and the GPU VM with NVIDIA driver extension
// Matches the configuration in VirtualMachineManager.create_gpu_vm()

@description('Name of the VM (will be used as prefix for all resources)')
param vmName string

@description('Azure region for resources')
param location string

@description('Network Security Group resource ID')
param networkSecurityGroupId string

@description('SSH public key data')
@secure()
param sshPublicKey string

@description('VM size - GPU-enabled SKU')
param vmSize string = 'Standard_NC4as_T4_v3'

@description('Admin username for the VM')
param adminUsername string = 'agent'

@description('OS disk size in GB')
param osDiskSizeGB int = 80

@description('Tags to apply to resources')
param tags object = {}

// Virtual Network
resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name: '${vmName}-vnet'
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: '${vmName}-subnet'
        properties: {
          addressPrefix: '10.0.0.0/24'
        }
      }
    ]
  }
}

// Public IP Address
resource publicIp 'Microsoft.Network/publicIPAddresses@2023-11-01' = {
  name: '${vmName}-public-ip'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
  }
}

// Network Interface
resource nic 'Microsoft.Network/networkInterfaces@2023-11-01' = {
  name: '${vmName}-nic'
  location: location
  tags: tags
  properties: {
    ipConfigurations: [
      {
        name: 'default'
        properties: {
          subnet: {
            id: vnet.properties.subnets[0].id
          }
          publicIPAddress: {
            id: publicIp.id
          }
          privateIPAllocationMethod: 'Dynamic'
        }
      }
    ]
    networkSecurityGroup: {
      id: networkSecurityGroupId
    }
  }
}

// GPU Virtual Machine
resource vm 'Microsoft.Compute/virtualMachines@2024-03-01' = {
  name: vmName
  location: location
  tags: tags
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-focal'
        sku: '20_04-lts-gen2'
        version: 'latest'
      }
      osDisk: {
        createOption: 'FromImage'
        diskSizeGB: osDiskSizeGB
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
      }
    }
    osProfile: {
      computerName: vmName
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
        }
      ]
    }
    // Required for GPU VMs
    additionalCapabilities: {
      ultraSSDEnabled: false
    }
    securityProfile: {
      uefiSettings: {
        secureBootEnabled: false
        vTpmEnabled: false
      }
      securityType: 'TrustedLaunch'
    }
  }
}

// NVIDIA GPU Driver Extension
resource gpuExtension 'Microsoft.Compute/virtualMachines/extensions@2024-03-01' = {
  name: 'NvidiaGpuDriverLinux'
  parent: vm
  location: location
  tags: tags
  properties: {
    publisher: 'Microsoft.HpcCompute'
    type: 'NvidiaGpuDriverLinux'
    typeHandlerVersion: '1.9'
    autoUpgradeMinorVersion: true
    settings: {}
  }
}

output vmId string = vm.id
output vmName string = vm.name
output publicIpAddress string = publicIp.properties.ipAddress
output vnetId string = vnet.id
output nicId string = nic.id
output extensionName string = gpuExtension.name

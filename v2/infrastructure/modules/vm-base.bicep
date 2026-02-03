// Base VM module - shared VM configuration
@description('VM name')
param vmName string

@description('Azure region')
param location string

@description('Network Interface ID')
param nicId string

@description('SSH public key')
@secure()
param sshPublicKey string

@description('VM size')
param vmSize string

@description('Admin username')
param adminUsername string = 'agent'

@description('OS disk size in GB')
param osDiskSizeGB int = 80

@description('Enable GPU settings (UEFI, security profile)')
param isGpuVm bool = false

@description('Tags')
param tags object = {}

// VM
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
          id: nicId
        }
      ]
    }
    additionalCapabilities: isGpuVm ? {
      ultraSSDEnabled: false
    } : null
    securityProfile: isGpuVm ? {
      uefiSettings: {
        secureBootEnabled: false
        vTpmEnabled: false
      }
      securityType: 'TrustedLaunch'
    } : null
  }
}

output vmId string = vm.id
output vmName string = vm.name

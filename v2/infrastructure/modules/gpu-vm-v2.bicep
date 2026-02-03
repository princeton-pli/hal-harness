// GPU VM - composed from networking + base VM + GPU extension modules
@description('VM name')
param vmName string

@description('Azure region')
param location string

@description('Network Security Group resource ID')
param networkSecurityGroupId string

@description('SSH public key')
@secure()
param sshPublicKey string

@description('VM size - GPU SKU')
param vmSize string = 'Standard_NC4as_T4_v3'

@description('Admin username')
param adminUsername string = 'agent'

@description('OS disk size in GB')
param osDiskSizeGB int = 80

@description('Tags')
param tags object = {}

// Networking
module networking 'networking.bicep' = {
  name: '${vmName}-networking'
  params: {
    vmName: vmName
    location: location
    networkSecurityGroupId: networkSecurityGroupId
    tags: tags
  }
}

// GPU VM
module vm 'vm-base.bicep' = {
  name: '${vmName}-vm'
  params: {
    vmName: vmName
    location: location
    nicId: networking.outputs.nicId
    sshPublicKey: sshPublicKey
    vmSize: vmSize
    adminUsername: adminUsername
    osDiskSizeGB: osDiskSizeGB
    isGpuVm: true
    tags: tags
  }
}

// NVIDIA GPU Driver Extension
resource gpuExtension 'Microsoft.Compute/virtualMachines/extensions@2024-03-01' = {
  name: '${vmName}/NvidiaGpuDriverLinux'
  location: location
  tags: tags
  properties: {
    publisher: 'Microsoft.HpcCompute'
    type: 'NvidiaGpuDriverLinux'
    typeHandlerVersion: '1.9'
    autoUpgradeMinorVersion: true
    settings: {}
  }
  dependsOn: [
    vm
  ]
}

output vmId string = vm.outputs.vmId
output vmName string = vm.outputs.vmName
output publicIpAddress string = networking.outputs.publicIpAddress
output vnetId string = networking.outputs.vnetId
output nicId string = networking.outputs.nicId
output extensionName string = gpuExtension.name

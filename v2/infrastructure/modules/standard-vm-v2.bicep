// Standard VM - composed from networking + base VM modules
@description('VM name')
param vmName string

@description('Azure region')
param location string

@description('Network Security Group resource ID')
param networkSecurityGroupId string

@description('SSH public key')
@secure()
param sshPublicKey string

@description('VM size')
param vmSize string = 'Standard_E2as_v5'

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

// VM
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
    isGpuVm: false
    tags: tags
  }
}

output vmId string = vm.outputs.vmId
output vmName string = vm.outputs.vmName
output publicIpAddress string = networking.outputs.publicIpAddress
output vnetId string = networking.outputs.vnetId
output nicId string = networking.outputs.nicId

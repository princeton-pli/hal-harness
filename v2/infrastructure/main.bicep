// Main Bicep template for HAL Harness Azure Infrastructure
targetScope = 'resourceGroup'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Network Security Group name')
param networkSecurityGroupName string

@description('Data Collection Endpoint ID')
param dataCollectionEndpointId string

@description('Log Analytics Workspace resource ID')
param logAnalyticsWorkspaceId string

@description('Tags to apply to all resources')
param tags object = {
  application: 'hal-harness'
  environment: 'production'
}

// Network Security Group
resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name: networkSecurityGroupName
  location: location
  tags: tags
  properties: {
    securityRules: [
      {
        name: 'AllowSSH'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '22'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 1000
          direction: 'Inbound'
          description: 'Allow SSH access for VM management'
        }
      }
      {
        name: 'DenyAllInbound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: '*'
          destinationAddressPrefix: '*'
          access: 'Deny'
          priority: 4096
          direction: 'Inbound'
          description: 'Deny all other inbound traffic'
        }
      }
    ]
  }
}

// Data Collection Rule for log monitoring
module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring-deployment'
  params: {
    location: location
    dataCollectionEndpointId: dataCollectionEndpointId
    logAnalyticsWorkspaceId: logAnalyticsWorkspaceId
    tags: tags
  }
}

output networkSecurityGroupId string = nsg.id
output networkSecurityGroupName string = nsg.name
output dataCollectionRuleId string = monitoring.outputs.dataCollectionRuleId
output dataCollectionRuleName string = monitoring.outputs.dataCollectionRuleName
output location string = location

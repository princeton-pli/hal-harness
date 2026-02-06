// Azure Monitor Data Collection Rule for HAL Docker Logs
targetScope = 'resourceGroup'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Data Collection Endpoint ID')
param dataCollectionEndpointId string

@description('Log Analytics Workspace resource ID')
param logAnalyticsWorkspaceId string

@description('Tags to apply to resources')
param tags object = {
  application: 'hal-harness'
  environment: 'production'
}

// Extract workspace name from resource ID
var workspaceName = last(split(logAnalyticsWorkspaceId, '/'))
var workspaceResourceGroup = split(logAnalyticsWorkspaceId, '/')[4]
var workspaceSubscriptionId = split(logAnalyticsWorkspaceId, '/')[2]

// Custom table for container logs
resource customTable 'Microsoft.OperationalInsights/workspaces/tables@2022-10-01' = {
  name: '${workspaceName}/ContainerLogs_CL'
  properties: {
    schema: {
      name: 'ContainerLogs_CL'
      columns: [
        {
          name: 'TimeGenerated'
          type: 'datetime'
        }
        {
          name: 'RawData'
          type: 'string'
        }
        {
          name: 'FilePath'
          type: 'string'
        }
        {
          name: 'Computer'
          type: 'string'
        }
        {
          name: 'RunID'
          type: 'string'
        }
        {
          name: 'TaskID'
          type: 'string'
        }
      ]
    }
  }
}

// Data Collection Rule for file-based log collection
resource dcr 'Microsoft.Insights/dataCollectionRules@2022-06-01' = {
  name: 'hal-docker-logs-dcr'
  location: location
  tags: tags
  kind: 'Linux'
  dependsOn: [
    customTable
  ]
  properties: {
    dataCollectionEndpointId: dataCollectionEndpointId
    streamDeclarations: {
      'Custom-ContainerLogs_CL': {
        columns: [
          {
            name: 'TimeGenerated'
            type: 'datetime'
          }
          {
            name: 'RawData'
            type: 'string'
          }
          {
            name: 'FilePath'
            type: 'string'
          }
          {
            name: 'Computer'
            type: 'string'
          }
          {
            name: 'RunID'
            type: 'string'
          }
          {
            name: 'TaskID'
            type: 'string'
          }
        ]
      }
    }
    dataSources: {
      logFiles: [
        {
          streams: [
            'Custom-ContainerLogs_CL'
          ]
          filePatterns: [
            '/home/agent/logging/agent_run/*/*.log'
          ]
          format: 'text'
          settings: {
            text: {
              recordStartTimestampFormat: 'ISO 8601'
            }
          }
          name: 'containerLogs'
        }
      ]
    }
    destinations: {
      logAnalytics: [
        {
          workspaceResourceId: logAnalyticsWorkspaceId
          name: 'halWorkspace'
        }
      ]
    }
    dataFlows: [
      {
        streams: [
          'Custom-ContainerLogs_CL'
        ]
        destinations: [
          'halWorkspace'
        ]
        transformKql: 'source | extend RunTaskID = extract(@\'hal_logs/([^/]+)/\', 1, FilePath) | extend RunID = tostring(split(RunTaskID, \'-\')[0]) | extend TaskID = tostring(split(RunTaskID, \'-\')[1]) | project TimeGenerated, RawData, FilePath, Computer, RunID, TaskID'
        outputStream: 'Custom-ContainerLogs_CL'
      }
    ]
  }
}

output dataCollectionRuleId string = dcr.id
output dataCollectionRuleName string = dcr.name

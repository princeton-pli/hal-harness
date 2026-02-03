// Parameters file for HAL Harness infrastructure
using './main.bicep'

// Network Security Group name - reads from environment variable or defaults
param networkSecurityGroupName = readEnvironmentVariable('NETWORK_SECURITY_GROUP_NAME', 'hal-harness-nsg')

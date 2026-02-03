"""
1. run bicep code to make the desired VM
2. a docker file that is run on that VM and logs to Azure Monitor
"""

virtual_machine_count = 3

run_id = random_hash

azure_manager = AzureManager(run_id=run_id)

# tags resources with run_id
network = azure_manager.create_network()

# uses network that's already part of the instance now
# tags resources with run_id
# returns a list[AzureVirtualMachine] class
virtual_machines = azure_manager.create_virtual_machines(
    count=virtual_machine_count, use_gpu=False
)

for virtual_machine in virtual_machines:
    virtual_machine.run_docker(
        # TODO: add arguments here; for now skip
    )

logger.info(f"Triggered Docker runs for {virtual_machine_count} VMs")
logger.info(f"Track detailled logs on {azure_manager.azure_monitor.url}")

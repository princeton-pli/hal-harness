import os

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Args:
        input: Dictionary mapping task IDs to task data
        **kwargs: Additional arguments passed via -A flags
    
    Returns:
        Dictionary mapping task IDs to submissions
    """
    # Print all tasks
    print("\n=== TASKS ===")
    for instance_id, task_data in input.items():
        print(f"\nTask ID: {instance_id}")
        print(f"Task Data: {task_data}")
    
    # List all files in the current directory recursively
    print("\n=== FILES IN ENVIRONMENT ===")
    for root, dirs, files in os.walk('.'):
        for file in files:
            print(os.path.join(root, file))
    
    # Return empty patches since we're just listing files
    return {instance_id: "Test" for instance_id in input.keys()}

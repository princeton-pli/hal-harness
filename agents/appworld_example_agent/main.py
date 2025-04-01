from appworld import AppWorld

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:

    assert 'model_name' in kwargs, 'model_name is required'
    
    task_id = list(input.keys())[0]
    
    with AppWorld(task_id=task_id, experiment_name="output", remote_environment_url="http://0.0.0.0:8000") as world:
        world.task.instruction # To see task instruction.

        # indicate task completion:
        world.execute("apis.supervisor.complete_task()")
        
    return {task_id: "Completed"}


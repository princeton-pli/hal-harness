from science_agent import ScienceAgent

def format_task_dict(example):
    task = {
        "instance_id": example["instance_id"],
        "task_inst": example["task_inst"],
        "dataset_path": "benchmark/datasets/" + example["dataset_folder_tree"].split("\n")[0][4:],
        "dataset_folder_tree": example["dataset_folder_tree"],
        "dataset_preview": example["dataset_preview"],
        "output_fname": example["output_fname"],
        "domain_knowledge": example["domain_knowledge"],
        "gold_program_name": example["gold_program_name"],
    }

    return task


def run(input_dict: dict[str, dict], **kwargs) -> dict[str, str]:
    assert 'model_name' in kwargs, 'model_name is required'
    assert len(input_dict) == 1, 'input must contain only one task'

    agent = ScienceAgent(
        kwargs['model_name'],
        context_cutoff=28000,
        use_self_debug=kwargs['use_self_debug'],
        use_knowledge=kwargs['use_knowledge']
    )

    task_id = list(input_dict.keys())[0]
    task = format_task_dict(list(input_dict.values())[0])
    out_fname = "pred_programs/pred_" + task["gold_program_name"]
    trajectory = agent.solve_task(task, out_fname=out_fname)

    return {task_id: trajectory}  # The output must be {task_id: result}

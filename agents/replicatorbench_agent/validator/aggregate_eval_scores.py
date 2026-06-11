import json


def summarize_eval_execute(eval_data):
    eval_scores = {}
    for sub_stage, sub_stage_eval_data in eval_data.items():
        eval_scores[f"execute_{sub_stage}"] = {
            "aspect_scores": {}
        }
        sub_stage_scores = []
        for aspect in sub_stage_eval_data:
            aspect_scores = []
            for rubric_id, rubric_info in sub_stage_eval_data[aspect].items():
                aspect_scores.append(rubric_info['score'])
            aspect_avg = sum(aspect_scores)/len(aspect_scores)
            eval_scores[f"execute_{sub_stage}"]["aspect_scores"][aspect] = aspect_avg
            sub_stage_scores.append(aspect_avg)
        eval_scores[f"execute_{sub_stage}"]["avg_score"] = sum(sub_stage_scores)/len(sub_stage_scores)
    return eval_scores

def _to_float_or_none(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s.upper() in {"NA", "N/A", ""}:
            return None
        return float(s)  # will still raise if it's something else
    return None

def summarize_eval_scores(study_path):
    stages = ["extract", "design", "execute", "interpret"]
    eval_summary = {}
    for stage in stages:
        with open(f"{study_path}/llm_eval/{stage}_llm_eval.json") as f:
            eval_json = json.load(f)
        if stage == "execute":
            eval_data = {
                "design": eval_json["evaluate_design"],
                "execute": eval_json["execute"] 
            }
            eval_summary.update(summarize_eval_execute(eval_data))
        else:
            aspect_totals = {}
            for eval_field, eval_info in eval_json.items():
                aspect = eval_field.split(".")[0]
                if aspect not in aspect_totals:
                    aspect_totals[aspect] = [0.0, 0.0]
                score = _to_float_or_none(eval_info.get("score"))
                if score is None:
                    continue
                aspect_totals[aspect][0] += score
                aspect_totals[aspect][1] += 3.0

            eval_summary[stage] = {"aspect_scores": {}}
            stage_scores = []
            for aspect, (score_sum, max_sum) in aspect_totals.items():
                aspect_avg = (score_sum / max_sum) if max_sum else 0.0
                eval_summary[stage]["aspect_scores"][aspect] = aspect_avg
                stage_scores.append(aspect_avg)

            eval_summary[stage]["avg_score"] = (
                sum(stage_scores) / len(stage_scores) if stage_scores else 0.0
            )
        with open(f"{study_path}/llm_eval/eval_summary.json", "w") as fout:
            json.dump(eval_summary, fout, indent =2)
            
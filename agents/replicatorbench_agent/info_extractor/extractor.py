"""
LLM_Benchmarking__
|
info_extractor--|extractor.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""
import os
import re
import json
import time
from openai import OpenAI
from openai.types.beta.threads import TextContentBlock
from core.utils import get_logger

from info_extractor.file_utils import split_models, read_file_contents, save_output, find_required_file, call_search_model_once, parse_json_strict
from info_extractor.prompt_builder import build_prompt, build_context_and_message
from core.constants import API_KEY, TEMPLATE_PATHS, FILE_SELECTION_RULES
from core.utils import configure_file_logging
from core.agent import update_metadata, messages_to_responses_input

from core.tools import read_and_summarize_pdf

client = OpenAI(api_key=API_KEY)
logger, formatter = get_logger()

def is_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "gpt-5"))

def run_stage_1(study_path, difficulty, show_prompt=False, model_name: str="gpt-4o"):
    """
    Extract original study information and save to post_registration.json
    """
    start_time = time.time()
    configure_file_logging(logger, study_path, f"extract.log")
    print(f"\n\nmodel name for extractor stage: {model_name}\n\n")

    logger.info("Running Stage 1: original study extraction")
    # Load post-registration template
    with open(TEMPLATE_PATHS['post_registration_template']) as f:
        template = json.load(f)

    # Load instructions for stage_1 / difficulty
    with open(TEMPLATE_PATHS['info_extractor_instructions']) as f:
        instructions = json.load(f).get(difficulty, {}).get("stage_1", {})

    file_context, datasets_original, datasets_replication, code_file_descriptions, original_study_data = read_file_contents(
        study_path, difficulty, FILE_SELECTION_RULES, stage="stage_1"
    )

    if not file_context:
        print(f"No content was read from {study_path}")

    context_message, full_message = build_context_and_message(
        study_path, template, file_context, stage="stage_1"
    )
    prompt = build_prompt(template, instructions, stage="stage_1")

    print("=== GENERATED PROMPT (Stage 1) ===")
    logger.info(f"=== GENERATED PROMPT (Stage 1) ===\n{prompt}")
    logger.info(f"\n\n=== GENERATED MESSAGE (Stage 1) ===\n{full_message}")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": full_message},
    ]

    response = client.responses.create(
        model=model_name,
        input=messages_to_responses_input(messages),
    )

    duration = time.time() - start_time
    usage = response.usage
    json_text = response.output_text.strip()

    # metric collection
    metric_data = {
        "total_time_seconds": round(duration, 2),
        "total_tokens": usage.total_tokens if usage else 0,
        "prompt_tokens": (
            usage.input_tokens if is_reasoning_model(model_name) else usage.input_tokens
        ) if usage else 0,
        "completion_tokens": (
            usage.output_tokens if is_reasoning_model(model_name) else usage.output_tokens
        ) if usage else 0,
        "total_turns": 1
    }
    update_metadata(study_path, "extract_stage_1", metric_data)

    extracted_json = None

    # Remove markdown-style code fences if present
    if json_text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    try:
        extracted_json = json.loads(json_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw text was:", json_text)
        logger.info(f"\n\n=== RAW TEXT (Stage 1) ===\n{json_text}")
        extracted_json = None

    save_output(extracted_json, study_path, stage="stage_1")
    return extracted_json

def run_web_search(study_path,model_name,show_prompt=False):
    """
    Reads initial_details.txt + original_paper.pdf, then calls the paired search model once
    to return URLs needed for replication. Saves found_urls.json.
    """
    start_time = time.time()
    configure_file_logging(logger, study_path, "find_urls.log")

    details_path = find_required_file(study_path, "initial_details.txt")
    paper_path = find_required_file(study_path, "original_paper.pdf")

    # read claim
    with open(details_path, "r", encoding="utf-8", errors="ignore") as f:
        claim_text = f.read()
    
    summarizer_model, search_model = split_models(model_name)
    print(f"[web-search] summarizer_model={summarizer_model} -> search_model={search_model}")
    
    paper_text = read_and_summarize_pdf(paper_path, summarizer_model=summarizer_model, for_data=True)
    raw = ""
    try:
    	raw = call_search_model_once(search_model, claim_text, paper_text)
    except Exception as e:
    	print(f"search model call failed: {search_model}")
    	
    parsed = parse_json_strict(raw)

    duration = time.time() - start_time

    study_id = None
    # extract id "/10/" from the study path. 
    match = re.search(r"[/\\](\d+)[/\\]", str(study_path))
    if match:
        study_id = int(match.group(1))

    # Save output
    out_obj = {
    	"id": study_id,
        "requested_model": model_name,
        "summarizer_model": summarizer_model,
        "search_model": search_model,
        "details_path": details_path,
        "paper_path": paper_path,
        "result": parsed,
        "raw_response": raw,
    }

    # out_dir = os.path.join("data", "results", "web-search")
    # os.makedirs(out_dir, exist_ok=True)

    # out_path = os.path.join(out_dir, f"merged_{model_name}.jsonl")
    # with open(out_path, "a", encoding="utf-8") as f:
    #     f.write(json.dumps(out_obj) + "\n")

    # Save the task artifact where HAL expects it
    out_path = os.path.join(study_path, "merged-urls.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Stage 'web_search' output saved to {out_path}")

    # Optional: also append a global audit log
    audit_dir = os.path.join("data", "results", "web-search")
    os.makedirs(audit_dir, exist_ok=True)
    audit_path = os.path.join(audit_dir, f"merged_{model_name}.jsonl")
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out_obj) + "\n")
        
    # Metadata
    metric_data = {
        "total_time_seconds": round(duration, 2),
        "total_turns": 1,
        # token usage maybe
    }
    update_metadata(study_path, "extract_stage_find_urls", metric_data)

    return out_obj



def run_extraction(study_path, difficulty, stage, model_name, show_prompt=False):

    if stage == "stage_1":
        return run_stage_1(study_path, difficulty, show_prompt, model_name)
    if stage == "web_search":
    	return run_web_search(study_path,model_name, show_prompt)
    else:
        raise ValueError(f"Unknown stage: {stage}")



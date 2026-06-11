"""
LLM_Benchmarking__
|
info_extractor--|prompt_builder.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import json
from core.utils import get_logger

logger, formatter = get_logger()


def build_prompt(template, instruction, stage="stage_1"):
    
    prompt = (
        instruction + "\n\n"
        + "Here is the JSON template, and its values represent descriptions of what is expected to be stored in each key:\n\n"
        + json.dumps(template, indent=2)
        + "\n\nPlease return only a completed JSON object appropriate for this stage."
    )
    return prompt


def build_context_and_message(study_path, template, file_context, stage, original_study=None):
    context_message = ""
    
    if stage == "stage_1":
        context_message = "Extract stage 1 (original study) information."
    
    elif stage == "stage_2":
        context_message = (
            "You are extracting structured information about the replication study. "
            "You also have access to the original study's extracted data and should integrate or reference it as specified by the replication_info schema."
        )

    # original_block = ""
    # if original_study:
    #     original_block = "\n=== ORIGINAL STUDY EXTRACTED DATA ===\n" + json.dumps(original_study, indent=2)

    full_message = (
        f"{context_message}\n\n"
        "You are tasked with extracting structured information from the following text "
        "based on the given instructions.\n\n"
        "=== START OF FILE CONTENT ===\n"
        f"{file_context}\n"
        "=== END OF FILE CONTENT ==="
        # f"{original_block}"
    )

    return context_message, full_message





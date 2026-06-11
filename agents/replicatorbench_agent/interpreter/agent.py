# interpreter/agent.py

import os
import json
import re
import logging
import sys
import tiktoken
import copy

from core.constants import API_KEY, INTERPRET_CONSTANTS
from openai import OpenAI

from info_extractor.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx

from core.prompts import PREAMBLE, INTERPRET, EXAMPLE
from core.agent import run_react_loop, save_output
from core.actions import base_known_actions, get_interpret_tool_definitions
from core.utils import get_logger, configure_file_logging, build_file_description  

logger, formatter = get_logger()

client = OpenAI(api_key=API_KEY)

MAX_TOKENS = 20000

def _count_tokens(text: str, model_name="gpt-4o"):
    enc = tiktoken.encoding_for_model(model_name if model_name else "gpt-4")
    return len(enc.encode(text))

def discover_interpretable_files(study_path: str):
    """
    Walks the study_path directory to discover new files that might usefule to interpetation
    """
    interpretable_exts = {".txt", ".docx", ".log"}
    auto_files = {}

    for root, dirs, files in os.walk(study_path):
        # Skip obviously unhelpful dirs
        basename = os.path.basename(root)
        if basename in {".git", "__pycache__"}:
            continue

        rel_dir = os.path.relpath(root, study_path)
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in interpretable_exts:
                # Build a path relative to study_path so tools can use it
                if rel_dir == ".":
                    rel_path = fname
                else:
                    rel_path = os.path.join(rel_dir, fname)

                # Avoid double-listing things already in INTERPRET_CONSTANTS if you want
                if "interpret" not in rel_path and "human" not in rel_path:
                    auto_files[rel_path] = (
                        f"Auto-discovered {ext} file in the study directory. "
                        f"May contain information relevant for interpreting the replication."
                    )

    return auto_files

def read_log(file_path: str, model_name: str = "gpt-4o"):
    """
    Tool: read a potentially very long log. If too big, chunk and summarize progressively.
    Returns a string (full text or summarized).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_log = f.read()
    except Exception as e:
        return f"[read_log error] {e}"

    if _count_tokens(full_log, model_name=model_name) <= MAX_TOKENS:
        return full_log

    # Chunk + summarize
    lines = full_log.splitlines(keepends=True)
    chunk_size = 800  # lines per chunk (tweakable)
    chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

    sys_prompt = (
        "You are an effective log reader. Summarize the provided log chunk. "
        "Focus on errors, exceptions, warnings, commands executed, and results."
    )

    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"CHUNK {idx}/{len(chunks)}:\n{chunk}"}
        ]
        try:
            out = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=messages
            )
            summaries.append(out.choices[0].message.content)
        except Exception as e:
            summaries.append(f"[summarization error on chunk {idx}] {e}")

    # One final synthesis pass (bounded)
    final_messages = [
        {"role": "system", "content": "Synthesize the chunk summaries into a concise but detailed overall summary."},
        {"role": "user", "content": "\n\n".join(summaries)}
    ]
    try:
        final = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=final_messages
        ).choices[0].message.content
    except Exception as e:
        final = "\n\n".join(summaries) + f"\n[final synthesis error] {e}"
    return final

system_prompt = "\n\n".join([PREAMBLE, INTERPRET, EXAMPLE])

# Map action names to their functions
known_actions = {
    **base_known_actions(),
    "read_log": read_log
}

def run_interpret(study_path, show_prompt=False, tier="easy", model="gpt-4o"):
    configure_file_logging(logger, study_path, f"interpret_{tier}.log")
    logger.info(f"Starting execution evaluation for study path: {study_path}")

    eval_prompt_template = read_txt(INTERPRET_CONSTANTS['prompt_template'])
    json_schema = read_json(INTERPRET_CONSTANTS['json_template'])
    claim_docs_for_evaluator = build_file_description(INTERPRET_CONSTANTS['claim_files'], study_path)
    agent_docs_for_evaluator = build_file_description(INTERPRET_CONSTANTS['agent_files'], study_path)

    auto_files_map = discover_interpretable_files(study_path)
    auto_files_for_evaluator = build_file_description(auto_files_map, study_path) if auto_files_map else ""
    logger.info(f"ADDITIONAL FILES FOUND: {auto_files_for_evaluator}")


    variables = {
        'interpret_json_schema': json_schema,
        'claim_docs_for_evaluator': claim_docs_for_evaluator,
        'agent_docs_for_evaluator': agent_docs_for_evaluator,
    }

    base_question = eval_prompt_template.format(**variables)

    print(f"\n\nfiles for interpreter: {auto_files_for_evaluator} \n\n")
    extra_instructions = ""
    if auto_files_for_evaluator:
        extra_instructions = f"""        
In addition to the documents listed above, the following files were automatically discovered
in the study directory and may contain useful information (logs, reports, outputs, datasets, etc.):

{auto_files_for_evaluator}

You should consider exploring these files when needed, using the available tools such as
`list_files_in_folder`, `read_log`, `read_txt`, `read_pdf`, `read_docx`, `read_json`,
`read_image`, and the dataset tools (`load_dataset`, `get_dataset_head`, `get_dataset_info`, etc.).
Only inspect what you think is necessary to complete the interpretation.

 Rememeber, every response needs to have the the following one of the two formats:
----- FORMAT 1 (For when you need to call actions to help accomplish the given task) -------
Thought: [Your thinking/planning process for completing the task based on interactions so far]
Action: [call next action to help you solve the task]
PAUSE
----- FORMAT 2 (For when you are ready to give a final response)-------
Thought: [Your thinking/planning process for completing the task based on interactions so far]
Answer: [Execute necessary next action to help you solve the task]
""".rstrip()

    question = "Question: " + base_question + extra_instructions

    tool_definitions = get_interpret_tool_definitions()

    return run_react_loop(
        system_prompt,
        known_actions,
        tool_definitions,
        question,
        study_path=study_path,
        session_state={"analyzers": {}},
        on_final=lambda ans: save_output(
            ans,
            study_path=study_path,
            filename="interpret_results.json",
            stage_name="interpret",
        ),
        model_name=model
    )

"""
LLM_Benchmarking__
|
validator--|extract_from_human_replication_study.py
Created on Wed Jun 18 00:32:19 2025
@author: Rochana Obadage
"""

import os
import fitz  # PyMuPDF
import openai
import json
import re
from docx import Document
from openai import OpenAI

from core.constants import API_KEY, TEMPLATE_PATHS
from info_extractor.file_utils import read_json


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_report_text(path):
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        return extract_text_from_docx(path)
    else:
        raise ValueError("Unsupported format")


def build_extraction_prompt(preregistration, score_report, expected_schema):
    return f"""
        You are an assistant that extracts structured metadata from replication study reports. The output should match the JSON schema used in the `replication_info_schema.json` file. Extract the information as accurately and concisely as possible.
        
        Please return a JSON object with the following structure:
        
        === replication_info_schema.json ===
        {expected_schema}
        
        If any field is unavailable or unclear, use \"N/A\".
        
        === REPLICATION STUDY PRE-REGISTRATION DOCUMENT START ===
        {preregistration}
        === REPLICATION STUDY PRE-REGISTRATION DOCUMENT END ===
        
        === REPLICATION STUDY REPORT TEXT START ===
        {score_report}
        === REPLICATION STUDY REPORT TEXT END ===

        Output Requirements:\n- Return a valid JSON object only.\n- Do NOT wrap the output in markdown (no ```json).\n- Do NOT include extra text, commentary, or notes.\n\nBegin extraction using the provided schema below and the file contents. Ensure accuracy and completeness.\n- Strictly use provided sources as specified
        """

def save_prompt_log(study_path, prompt):
    case_name = os.path.basename(os.path.normpath(study_path))

    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", study_path)
        if match:
            case_name = match.group()
            
    log_dir = "logs"
    # log_dir = os.path.join(study_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{case_name}_validator_ehri_log.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")

    print(f"[INFO] Prompt logged to {log_file}")
    

def generate_expected_json(preregistration, score_report, expected_schema, client, log_path):
    prompt = build_extraction_prompt(preregistration, score_report, expected_schema)

    save_prompt_log(log_path, prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    content = response.choices[0].message.content
    return json.loads(content)


def save_json(data, path):
    save_path = os.path.join(path, "llm_eval", "design_llm_eval.json")
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
        print(f"extract_from_human_replication_study.py output saved to {save_path}")


def extract_from_human_replication_study(preregistration_path, score_report_path, output_path):
    client = OpenAI(api_key=API_KEY)
    
    expected_schema = read_json(TEMPLATE_PATHS['replication_info_template'])
    
    preregistration = load_report_text(preregistration_path)
    score_report = load_report_text(score_report_path)
    
    log_path = os.path.dirname(output_path)
    expected_json = generate_expected_json(preregistration, score_report, expected_schema, client, log_path)
    save_json(expected_json, output_path)

import os
import fitz  # PyMuPDF
import openai
import json
import re
from docx import Document
from openai import OpenAI

from hal.benchmarks.replicatorbench.constants import API_KEY, TEMPLATE_PATHS
from hal.benchmarks.replicatorbench.validator.file_utils import read_json, read_txt


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



def build_extract_evaluate_prompt(eval_prompt_template, extraction_schema, extracted_json, expected_json):
    variables = {
        'extraction_schema': extraction_schema,
        'extracted_json': extracted_json,
        'expected_json': expected_json,
    }

    final_prompt = eval_prompt_template.format(**variables)
    return final_prompt

def save_prompt_log(study_path, prompt):
    case_name = os.path.basename(os.path.normpath(study_path))

    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", study_path)
        if match:
            case_name = match.group()
            
    log_dir = os.path.join(study_path, "llm_eval")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"extract_llm_eval.log")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")

    print(f"[INFO] Prompt logged to {log_file}")
    

def generate_evaluation_json(eval_prompt_template, expected_schema, extracted_json, expected_json, client, log_path):
    prompt = build_extract_evaluate_prompt(eval_prompt_template, expected_schema, extracted_json, expected_json)

    save_prompt_log(log_path, prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    content = response.choices[0].message.content
    return json.loads(content)


def save_json(data, path):
    save_path = os.path.join(path, "llm_eval", "extract_llm_eval.json")
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
        print(f"extract_and_evaluate_from_human_rep.py output saved to {save_path}")


def extract_from_human_postreg(extracted_json_path, expected_json_path, output_path):
    client = OpenAI(api_key=API_KEY)
    
    with open(TEMPLATE_PATHS['extract_eval_prompt_template'], "r", encoding="utf-8", errors="ignore") as f:
        eval_prompt_template = f.read()
    # eval_prompt_template = read_txt()
    
    
    with open(TEMPLATE_PATHS['post_registration_template'], "r", encoding="utf-8", errors="ignore") as f:
        expected_schema = f.read()
    # expected_schema = read_json()
    
    
    with open(extracted_json_path, "r", encoding="utf-8", errors="ignore") as f:
        extracted_json = f.read()
    # extracted_json = read_json(extracted_json_path)
    
    with open(expected_json_path, "r", encoding="utf-8", errors="ignore") as f:
        expected_json = f.read()
    # expected_json = read_json(expected_json_path)
    
    log_path = output_path
    evaluated_json = generate_evaluation_json(eval_prompt_template, expected_schema, extracted_json, expected_json, client, log_path)
    save_json(evaluated_json, output_path)
import os
import fitz  # PyMuPDF
import json
from docx import Document
from openai import OpenAI
from hal.benchmarks.replicatorbench.constants import API_KEY, EVALUATE_INTERPRET_CONSTANTS
from hal.benchmarks.replicatorbench.validator.file_utils import read_json, read_txt, read_pdf, read_docx


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



def build_extract_evaluate_prompt(eval_prompt_template, interpret_schema, reported_json, reference_doc):
    variables = {
        'interpret_schema': interpret_schema,
        'reported_json': reported_json,
        'reference_report_doc': reference_doc,
    }

    final_prompt = eval_prompt_template.format(**variables)
    return final_prompt

def save_prompt_log(study_path, prompt):
    log_dir = os.path.join(study_path,  "llm_eval")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"interpret_eval.log")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")

    print(f"[INFO] Prompt logged to {log_file}")
    

def generate_evaluation_json(eval_prompt_template, expected_schema, extracted_json, reference_doc, client, log_path):
    prompt = build_extract_evaluate_prompt(eval_prompt_template, expected_schema, extracted_json, reference_doc)

    save_prompt_log(log_path, prompt)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    content = response.choices[0].message.content
    return json.loads(content)


def save_json(data, path):
    save_path = os.path.join(path, "llm_eval", "interpret_llm_eval.json")
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
        print(f"Interpret Evaluation output saved to {save_path}")


def extract_from_human_report(interpret_results_path, reference_report_path, study_path):
    client = OpenAI(api_key=API_KEY)
    
    eval_prompt_template = read_txt(EVALUATE_INTERPRET_CONSTANTS['prompt_template'])
    
    interpret_schema = read_json(EVALUATE_INTERPRET_CONSTANTS['json_template'])
    reported_json = read_json(interpret_results_path)
    reference_doc = ""
    if reference_report_path.lower().endswith(".pdf"):
        reference_doc = read_pdf(reference_report_path)
    elif reference_report_path.lower().endswith(".docx"):
        reference_doc = read_docx(reference_report_path)
    
    log_path = study_path
    evaluated_json = generate_evaluation_json(eval_prompt_template, interpret_schema, reported_json, reference_doc, client, log_path)
    save_json(evaluated_json, study_path)
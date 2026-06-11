"""
LLM_Benchmarking__
|
info_extractor--|file_utils.py
Created on Mon Jun  9 15:36:52 2025
@author: Rochana Obadage
"""

import os
import pymupdf
import json
import pandas as pd
import pyreadr
import io
import re
import docx
from pathlib import Path
from core.utils import get_logger
from core.tools import read_and_summarize_pdf
import tiktoken

logger, formatter = get_logger()

def find_required_file(study_path: str, filename: str) -> str:
    """
    Finds filename inside study_path (direct child first, then recursive).
    """
    direct = os.path.join(study_path, filename)
    if os.path.exists(direct):
        return direct

    for root, _, files in os.walk(study_path):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(f"Required file not found: {filename} under {study_path}")

URL_FINDER_SYSTEM_PROMPT = """
You are a replication assistant with web search.
Definitions (important):
- Reproduction: re-running the ORIGINAL authors' analysis on the SAME dataset/sample (often using the same code) to verify the published results.
- Replication (our goal): independently re-testing the claim using NEW data or an independent sample/population/time period, while following the paper's design/operationalization as closely as feasible.


You will receive:
- The replication claim/hypotheses (initial_details.txt)
- The original paper text (extracted from original_paper.pdf; may be summarized if very long)

Task:
Find ALL URLs needed to replicate the claim (not just reproduce the original run):
- data sources (datasets, archives, portals, OSF/Zenodo/Dataverse/MIT, etc.)
- code sources (GitHub repos, MIT, OSF code, supplemental code archives)

If the original dataset is restricted, outdated, or a one-off sample, also find the closest feasible data source that can provide a NEW sample with the same required variables and measurement definitions (e.g., the same survey instrument in a later wave, a similar administrative dataset, or a new cohort drawn from the same population).
Why this matters: replication often requires collecting/drawing a fresh sample; the key is that the data allow construction of the necessary variables and sample criteria, not that it is the identical original sample.

Return ONLY JSON in the following format:

{
  "urls": [
    {
      "url": "https://...",
      "kind": "data|code",
      "resource_name": "short name",
      "why_needed": "one sentence"
    }
  ],
  "missing": [
    {
      "resource_name": "what's missing",
      "search_query": "a query to find it"
    }
  ]
}

Rules:
- Use web search when the paper describes a dataset/repo but doesn't give a URL.
- Prefer official/stable landing pages (DOI landing, archive page, repository homepage).
- Do not include prose outside the JSON.
""".strip()

def split_models(requested_model: str) -> tuple[str, str]:
    # If requested_model is a search/deep-research variant, summarizer uses the base model.
    # Examples:
    #   o3 -> (o3, o3)
    #   o3-deep-research -> (o3, o3-deep-research)
    #   gpt-4o-search-review -> (gpt-4o, gpt-4o-search-review)
    #   gpt-5-search-api -> (gpt-5, gpt-5-search-api)
    base = requested_model
    for suffix in ("-deep-research", "-search-preview", "-search-review", "-search-api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base, requested_model

def call_search_model_once(search_model: str, claim_text: str, paper_text: str) -> str:
    """
    DIFF: uses Chat Completions because search-preview models are documented for Chat Completions. :contentReference[oaicite:2]{index=2}
    Keep arguments minimal for compatibility.
    """
    user_msg = (
        "CLAIM / HYPOTHESES (from initial_details.txt):\n"
        f"{claim_text}\n\n"
        "ORIGINAL PAPER TEXT (from original_paper.pdf):\n"
        f"{paper_text}\n"
    )
    messages = [
        {"role": "system", "content": URL_FINDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        if "-search-" in search_model:
            resp = client.chat.completions.create(
                model=search_model,
                messages=messages,
            )
            return (resp.choices[0].message.content or "").strip()
        else:
            resp = client.responses.create(
                model=search_model,
                input=messages,
                tools=[{"type": "web_search"}],
            )
            return (resp.output_text or "").strip()
    except Exception as e:
        logger.warning(
            f"[web-search] Responses API failed for model={search_model}; "
            f"falling back to Chat Completions (no web_search tool). Error: {e}"
        )
        if "-search-" in search_model:
            resp = client.chat.completions.create(
                model=search_model,
                messages=messages,
            )
        else:
            resp = client.chat.completions.create(
                model=search_model,
                messages=messages,
                tools=[{"type": "web_search"}],
            )
        return (resp.choices[0].message.content or "").strip()


def parse_json_strict(text: str):
    """
    Removes ``` fences if present and parses JSON. Returns None on failure.
    """
    t = text.strip()
    if t.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
        if match:
            t = match.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        return None


from openai import OpenAI
from core.constants import API_KEY
client = OpenAI(api_key=API_KEY)

def read_txt(file_path, model_name: str = "gpt-4o"):
    file_path = str(file_path)
    if not file_path.endswith(".txt"):
        return "not a .txt file"
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        file_content = f.read()
    return check_long_logs(file_content, model_name=model_name)

def check_long_logs(full_doc_content: str, model_name: str = "gpt-4o"):
    """
    Tool: read a potentially very long log. If too big, chunk and summarize progressively.
    Returns a string (full text or summarized).
    """
    def _count_tokens(text: str, model_name="gpt-4o") -> int:
        try:
            enc = tiktoken.encoding_for_model(model_name if model_name else "gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    
    MAX_TOKENS = 20000
    if _count_tokens(full_doc_content, model_name=model_name) <= MAX_TOKENS:
        return full_doc_content
    
    print(f"Document has > 20000 tokens. Summarizing content to prevent overflow...")

    # Chunk + summarize
    # lines = full_doc_content.splitlines(keepends=True)
    # chunk_size = 800  # lines per chunk (tweakable)
    # chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    chunk_size = 12000
    chunks = [full_doc_content[i:i+chunk_size] for i in range(0, len(full_doc_content), chunk_size)]

    sys_prompt = (
        "You are an effective file reader. Summarize the provided log chunk. "
        "Focus on errors, exceptions, warnings, commands executed, and results."
    )

    running_summary = "No logs read yet."
    for idx, chunk in enumerate(chunks, 1):
        print(f"Summarizing {idx}/{len(chunks)} chunks")
        user_content = (
            f"--- EXISTING SUMMARY (Context from previous {idx-1} chunks) ---\n"
            f"{running_summary}\n\n"
            f"--- NEW LOG CHUNK ({idx}/{len(chunks)}) ---\n"
            f"{chunk}"
            f"Only return the summary of the new log chunk."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
        try:
            out = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=messages
            )
            running_summary += "\n" + (out.choices[0].message.content or "")
            #running_summary.append(out.choices[0].message.content)
        except Exception as e:
            running_summary += f"\n[Error processing chunk {idx}: {e}]"


    return running_summary


def read_pdf(file_path):
    try:
        with pymupdf.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"[PDF read error: {e}]"
    
def read_docx(file_path: str, model_name: str = "gpt-4o") -> str:
    try:
        doc = docx.Document(file_path)
        # Extract text from each paragraph and join with a newline
        full_text = [para.text for para in doc.paragraphs]
        full_text = '\n'.join(full_text)
        return check_long_logs(full_text, model_name)
    except FileNotFoundError:
        return f"Error: The file at {file_path} was not found."
    except Exception as e:
        # This can catch errors from corrupted or non-standard docx files
        return f"An error occurred while reading the docx file: {e}"


def read_json(file_path, model_name: str = "gpt-4o"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        json_str =  json.dumps(data, indent=2)
        return check_long_logs(json_str)
    except Exception as e:
        return f"[JSON read error: {e}]"


def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    except Exception as e:
        return f"[CSV read error: {e}]"

FILE_READERS = {
    ".txt": read_txt,
    ".pdf": read_and_summarize_pdf,
    ".json": read_json,
    ".csv": read_csv,
}
    
    
def summarize_dataset(file_path):
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame.from_dict(data)
            else:
                return {"columns": None, "info": "[JSON parsing error: Unsupported structure]", "describe": None}
        elif ext == ".rdata":
            result = pyreadr.read_r(file_path)
            if result:
                df = next(iter(result.values()))
            else:
                return {"columns": None, "info": "[RData parsing error: No data frames found]", "describe": None}
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return {"columns": None, "info": f"[Unsupported format: {ext}]", "describe": None}
        
        # Capture df.info()
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        
        # Try describe
        try:
            describe = df.describe(include='all', datetime_is_numeric=True).to_string()
        except Exception as e:
            describe = f"[describe() failed: {e}]"

        return {
            "columns": df.columns.tolist(),
            "info": info,
            "describe": describe
        }

    except Exception as e:
        return {
            "columns": None,
            "info": f"[Data summary error: {e}]",
            "describe": None
        }



def read_file_contents(folder, difficulty, selection_rules, stage="stage_1"):
    folder = Path(folder)

    tier_rules = selection_rules['info_extractor'].get(difficulty, {})
    stage_rules = tier_rules.get(stage, {})
    allowed_files = stage_rules.get("files", [])
    allowed_folders = stage_rules.get("folders", {})

    aggregated_content = []
    code_section = ["\n=== CODE RELATED FILES ==="]
    dataset_section = ["\n=== DATASET FILES ==="]
    datasets_original = []
    datasets_replication = []
    code_file_descriptions = {}

    for file in folder.iterdir():

        if file.is_dir():
            # CODEBASE folder (only include if allowed via folder rules or conventionally named)
            if (file.name.lower() == "code") and ("code" in allowed_folders or not allowed_folders):
                for code_file in file.glob("*.*"):
									
                    try:
                        with open(code_file, "r", encoding="utf-8", errors="ignore") as f:
                            code_content = f.read(3000)
                        code_section.append(f"\n---\n**{code_file.name}**\n{code_content}")
                        code_file_descriptions[code_file.name] = code_content
                    except Exception as e:
                        code_section.append(f"\n---\n**{code_file.name}**\n[Error reading file: {e}]")
                        logger.warning(f"\n---\n**{code_file.name}**\n[Error reading file: {e}]")

            # DATASET folder
            if file.name.lower() in ["dataset_folder", "datasets", "data"] and \
               ("data" in allowed_folders or not allowed_folders):
                for data_file in file.glob("*.*"):
                    summary = summarize_dataset(data_file)
                    head = ""
                    try:
                        if data_file.suffix == ".csv":
                            df = pd.read_csv(data_file)
                            head = df.head().to_string()
                        elif data_file.suffix == ".xlsx":
                            df = pd.read_excel(data_file)
                            head = df.head().to_string()
                        elif data_file.suffix == ".json":
                            with open(data_file, "r") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                                head = df.head().to_string()
                            else:
                                head = json.dumps(data, indent=2)[:1000]
                        elif data_file.suffix == ".rdata":
                            result = pyreadr.read_r(data_file)
                            if result:
                                df = next(iter(result.values()))
                                head = df.head().to_string()
                    except Exception as e:
                        head = f"[Error reading dataset head: {e}]"
																			   
                        logger.exception(f"[Error reading dataset head: {e}]")

																		
                    dataset_section.append(f"\n---\n**{data_file.name}**\n"
                                           f"=== HEAD ===\n{head}\n"
                                           f"=== INFO ===\n{summary.get('info')}\n"
                                           f"=== DESCRIBE ===\n{summary.get('describe')}")

                    dataset_obj = {
                        "name": data_file.stem,
                        "filename": data_file.name,
                        "type": "original" if "replication" not in data_file.name.lower() else "replication",
                        "file_format": data_file.suffix[1:],
                        "columns": summary.get("columns"),
                        "summary_statistics": {
                            "info": summary.get("info"),
                            "describe": summary.get("describe")
                        },
                        "access": {
                            "url": None,
                            "restrictions": None
                        },
                        "notes": None
                    }

                    if dataset_obj["type"] == "replication":
                        datasets_replication.append(dataset_obj)
                    else:
                        datasets_original.append(dataset_obj)

        # non-directory core files
        if file.name not in allowed_files:
            continue
        																				 
        elif file.suffix in FILE_READERS:
            try:
                reader = FILE_READERS[file.suffix]
                content = reader(file)
                aggregated_content.append(f"\n---\n**{file.name}**\n{content}")
            except Exception as e:
                aggregated_content.append(f"\n---\n**{file.name}**\n[Error reading file: {e}]")
                logger.exception(f"\n---\n**{file.name}**\n[Error reading file: {e}]")
        
    original_study_data = None
    if stage == 'stage_2':                
        # Load existing original study output
        post_reg_path = os.path.join(folder, "post_registration.json")
        
        
        if os.path.exists(post_reg_path):
            try:
                with open(post_reg_path, "r", encoding="utf-8") as f:
                    original_study_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read existing post_registration.json: {e}")
        else:
            raise ValueError("post_registration.json not found")

													   
    file_context = "\n".join(aggregated_content + code_section + dataset_section)

    return file_context, datasets_original, datasets_replication, code_file_descriptions, original_study_data 

    

def save_output(extracted_json, study_path, stage):
    if stage == "stage_1":
        output_filename = "post_registration.json"
    elif stage == "stage_2":
        output_filename = "replication_info.json"


    output_path = os.path.join(study_path, output_filename)
    try:
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(extracted_json, f, indent=2)
        print(f"[INFO] Stage '{stage}' output saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save output for stage {stage}: {e}")
        logger.exception(f"Failed to save output for stage {stage}: {e}")
    

def save_prompt_log(study_path, stage, prompt, full_message):
									   
    case_name = os.path.basename(os.path.normpath(study_path))
    
    if "case_study" not in case_name:
        match = re.search(r"case_study_\d+", study_path)
        if match:
            case_name = match.group()

					   
    log_dir = "logs"
												
    os.makedirs(log_dir, exist_ok=True)

				   
    log_file = os.path.join(log_dir, f"{case_name}_{stage}_log.txt")

				  
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== GENERATED PROMPT ===\n")
        f.write(prompt + "\n\n")
        f.write("=== GENERATED FULL MESSAGE ===\n")
        f.write(full_message + "\n")

    print(f"[INFO] Prompt and message logged to {log_file}")



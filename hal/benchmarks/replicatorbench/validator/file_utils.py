import os
import pymupdf
import json
import pandas as pd
import pyreadr
import io
import re
import docx
from pathlib import Path
from hal.benchmarks.replicatorbench.core.utils import get_logger
from hal.benchmarks.replicatorbench.core.tools import read_and_summarize_pdf
import tiktoken

from openai import OpenAI
from hal.benchmarks.replicatorbench.constants import API_KEY


client = OpenAI(api_key=API_KEY)
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




import base64
from openai import OpenAI
import os
import json
import pandas as pd
import pyreadr
from hal.benchmarks.replicatorbench.core.constants import API_KEY
from typing import Dict, Any, Optional, Tuple
import io # Add this import at the top of your file
from pathlib import Path
import difflib

from pypdf import PdfReader

client = OpenAI(api_key=API_KEY)

class DataFrameAnalyzer:
    """
    A class to load and analyze a tabular dataset from a file.

    Loads a DataFrame once upon initialization and provides methods
    to perform common exploratory analysis tasks.
    """
    def __init__(self, file_path: str):
        """
        Initializes the analyzer and loads the data.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = self._load_data()

    def _load_data(self) -> Optional[pd.DataFrame]:
        """
        Private method to load data from the file_path.
        Handles both .csv, .xlsx, and .dta files and potential errors.
        """
        # Get the file extension from the file path
        _, file_extension = os.path.splitext(self.file_path)
        
        try:
            print(f"Loading data from {self.file_path}...")
            
            # Choose the correct pandas function based on the extension
            if file_extension == '.csv':
                return pd.read_csv(self.file_path)
            elif file_extension in ['.xlsx', '.xls']:
                # You might need to install openpyxl: pip install openpyxl
                return pd.read_excel(self.file_path)
            elif file_extension == '.dta':
                return pd.read_stata(self.file_path)
            elif file_extension.lower() == '.rds':
                return pyreadr.read_r(self.file_path)[None]
            else:
                print(f"Error: Unsupported file type '{file_extension}'.")
                return None

        except FileNotFoundError:
            print(f"Error: The file at {self.file_path} was not found.")
            return None
        except (pd.errors.ParserError, ValueError, Exception) as e:
            # Catch pandas parsing errors and other potential issues
            print(f"An error occurred while reading the file: {e}")
            return None

    def get_head(self, n: int = 5) -> Optional[pd.DataFrame]:
        """Returns the first n rows of the DataFrame."""
        if self.df is not None:
            return self.df.head(n)
        return None

    def get_shape(self) -> Optional[Tuple[int, int]]:
        """Returns the shape (rows, columns) of the DataFrame."""
        if self.df is not None:
            # .shape is an attribute, not a function
            return self.df.shape
        return None

    def get_info(self) -> str: # Change the return type hint to str
        """
        Returns a concise summary of the DataFrame as a string.
        """
        if self.df is not None:
            # Create an in-memory text buffer
            buffer = io.StringIO()
            
            # Tell df.info() to write its output to the buffer instead of the console
            self.df.info(buf=buffer)
            
            # Get the string from the buffer and return it
            return buffer.getvalue()
        return "Error: DataFrame not loaded."

    def get_description(self) -> Optional[pd.DataFrame]:
        """Returns descriptive statistics of the DataFrame."""
        if self.df is not None:
            return self.df.describe()
        return None
    
    def get_variable_summary(self, variable_name) -> str:
        """
        Rreturns summary statistics for a specific variable.
        - Numeric: Returns the 5-number summary (Min, Q1, Median, Q3, Max).
        - Categorical: Returns counts of unique categories (capped at top 20).
        """
        
        # 1. Load the Data

        # 2. Check if variable exists
        if variable_name not in self.df.columns:
            available_cols = ", ".join(self.df.columns[:5]) # Show first 5 as hint
            return f"Error: Variable '{variable_name}' not found. (First few columns: {available_cols}...)"

        series = self.df[variable_name]
        
        # 3. Handle Numeric Variables (5-number summary)
        if pd.api.types.is_numeric_dtype(series):
            # clean data (drop NAs for accurate stats)
            clean_series = series.dropna()
            
            if clean_series.empty:
                return f"Variable '{variable_name}' contains only NaN values."

            quartiles = clean_series.quantile([0.25, 0.5, 0.75])
            
            summary = (
                f"--- Numeric Summary for '{variable_name}' ---\n"
                f"Min:    {clean_series.min()}\n"
                f"Q1:     {quartiles[0.25]}\n"
                f"Median: {quartiles[0.5]}\n"
                f"Q3:     {quartiles[0.75]}\n"
                f"Max:    {clean_series.max()}\n"
                f"missing_values: {series.isna().sum()}"
            )
            return summary

        # 4. Handle Categorical/Character Variables
        else:
            # Get value counts
            counts = series.value_counts(dropna=False)
            unique_count = len(counts)
            
            # Guardrail: Don't print 10,000 rows if it's high cardinality
            top_n = 20
            truncated = unique_count > top_n
            display_counts = counts.head(top_n)
            
            output_lines = [f"--- Categorical Summary for '{variable_name}' ---",
                            f"Total Unique Categories: {unique_count}"]
            
            for cat, count in display_counts.items():
                output_lines.append(f"- {cat}: {count}")
                
            if truncated:
                output_lines.append(f"... (and {unique_count - top_n} more categories)")
                
            return "\n".join(output_lines)
    
    
def load_dataset(session_state: Dict[str, Any], file_path: str) -> str:
    """
    Loads a dataset and stores its analyzer in the session state.
    """
    analyzers = session_state["analyzers"]
    if file_path in analyzers:
        return f"Dataset '{file_path}' is already loaded."
    
    analyzer = DataFrameAnalyzer(file_path)
    if analyzer.df is not None:
        analyzers[file_path] = analyzer
        return f"Successfully loaded dataset '{file_path}'."
    else:
        return f"Failed to load dataset from '{file_path}'."

def get_dataset_shape(session_state: Dict[str, Any], file_path: str) -> str:
    """
    Gets the shape from an analyzer in the session state.
    """
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_shape())

def get_dataset_head(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_head())

def get_dataset_info(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_info())

def get_dataset_description(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_description())

def get_dataset_columns(session_state: Dict[str, Any], file_path: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(list(analyzers[file_path].df.columns))

def get_dataset_variable_summary(session_state: Dict[str, Any], file_path: str, variable_name: str) -> str:
    analyzers = session_state["analyzers"]
    if file_path not in analyzers:
        return "Error: Dataset not loaded. Please call load_dataset() first."
    return str(analyzers[file_path].get_variable_summary(variable_name))

def read_image(file_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    base64_image = encode_image(file_path)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Describe this image in details." },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ])
    return completion.choices[0].message.content



def list_files_in_folder(study_path, folder_path: str) -> str:
    """
    Recursively lists all files within a specified folder and its subfolders.

    Args:
        folder_path: The absolute or relative path to the folder.

    Returns:
        A string containing the relative paths of all files found,
        each separated by a newline. If the folder does not exist
        or is not a directory, an error message is returned.
    """
    abs_folder = os.path.abspath(folder_path)
    abs_study = os.path.abspath(study_path)
    
    # Check if abs_folder is inside abs_study
    # We check if the common path between them is the study_path itself
    try:
        if os.path.commonpath([abs_folder, abs_study]) != abs_study:
            return f"Error: Access denied. '{folder_path}' is outside of the study directory. You can only search within {study_path}"
    except ValueError:
        # This handles cases where paths are on different drives (Windows)
        return "Error: Paths are on different drives."
    
    # Check if the provided path exists
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' does not exist."

    # Check if the provided path is actually a directory
    if not os.path.isdir(folder_path):
        return f"Error: Path '{folder_path}' is not a directory."

    file_paths = []
    strs2avoid = ["human_preregistration","metadata.json", "human_report", "llm_eval", "expected_post_registration"]

    # Walk through all directories and subdirectories
    for current_root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(current_root, file)
            # Store paths relative to the provided folder
            relative_path = os.path.relpath(full_path, folder_path)
            if not any(s in relative_path for s in strs2avoid): #avoid cheating
                file_paths.append(relative_path)

    if not file_paths:
        return f"Folder path: {folder_path}\nNo files found."

    file_paths.sort()

    file_info = f"Folder path: {folder_path}\n"
    file_info += "All files:\n" + "\n".join(file_paths)
    return file_info






data_summarizer_prompt = (
"You are a careful technical paper summarizer. "
"Summarize this chunk while preserving dataset names, code/data availability clues, "
"repository/archive mentions, DOIs, accession numbers, and any described download procedures."
)

normal_summarization_prompt = (
"You are a helpful research assistant."
"Summarize the following text from a technical paper/document. Capture key methodologies, specific metrics, results, and conclusions. Do not lose specific data points."
)


def read_pdf(file_path: str, summarizer_model: str="gpt-4o", for_data: bool=False) -> str:
    """
    Reads a PDF file. If the PDF is short (<= 15 pages), it returns the full text.
    If the PDF is long (> 15 pages), it splits the text into chunks and uses the
    LLM to summarize each chunk, returning a consolidated summary to save context window.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Full text or a consolidated summary of the PDF.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    try:
        reader = PdfReader(file_path)
        number_of_pages = len(reader.pages)
        
        # Extract all text first
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        # THRESHOLD: If 15 pages or less, just return the text as is.
        if number_of_pages <= 15:
            print(f"PDF is short ({number_of_pages} pages). Returning full text.")
            return f"--- START OF PDF CONTENT ({number_of_pages} pages) ---\n{full_text}\n--- END OF PDF CONTENT ---"

        # LOGIC FOR LONG PDFS
        print(f"PDF is long ({number_of_pages} pages). Summarizing content to prevent overflow...")
        
        # Split text into chunks of roughly 12,000 characters (approx 3-4k tokens)
        chunk_size = 12000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        summaries = []
        total_chunks = len(chunks)
        prompt = data_summarizer_prompt if for_data else normal_summarization_prompt
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{total_chunks}...")
            try: 
                completion = client.chat.completions.create(
                    model=summarizer_model,
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {"role": "user", "content": chunk}
                    ]
                )
                summaries.append(completion.choices[0].message.content)
            except Exception as e:
                print(f"Chunk {i+1} failed: {e}")
                summaries.append(f"Chunk {i+1} fauled: {e}")
        consolidated_summary = "\n\n".join(summaries)
        
        return (f"--- PDF SUMMARY (Document was {number_of_pages} pages long) ---\n"
                f"The document was too long to read directly, so here is a detailed summary of all sections:\n\n"
                f"{consolidated_summary}")

    except Exception as e:
        return f"Error reading or summarizing PDF: {e}"
    
def ask_human_input(question: str) -> str:
    """
    Prompts the human user for input in the terminal.

    Use this tool when you are stuck, need clarification, or require 
    information that you cannot find or deduce from the available files.

    Args:
        question_to_ask (str): The clear, specific question to ask the human user.

    Returns:
        str: The human's response from the terminal.
    """
    # Print a clear message to the user indicating the agent needs help
    print("\n🤔 [AGENT NEEDS HUMAN INPUT] 🤔")
    print(f"Agent's Question: {question}")
    
    # Get input from the user
    human_response = input("Your Response: ")
    
    return human_response
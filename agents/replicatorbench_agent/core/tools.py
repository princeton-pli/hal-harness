import base64
from openai import OpenAI
import os
import json
import pandas as pd
import pyreadr
from core.constants import API_KEY
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



# def ask_human_input(question: str) -> str:
#     """
#     Prompts the human user for input in the terminal.

#     Use this tool when you are stuck, need clarification, or require 
#     information that you cannot find or deduce from the available files.

#     Args:
#         question_to_ask (str): The clear, specific question to ask the human user.

#     Returns:
#         str: The human's response from the terminal.
#     """
#     # Print a clear message to the user indicating the agent needs help
#     print("\n🤔 [AGENT NEEDS HUMAN INPUT] 🤔")
#     print(f"Agent's Question: {question}")
    
#     # Get input from the user
#     human_response = input("Your Response: ")
    
#     return human_response

def ask_human_input(question: str) -> str:
    print("\n🤔 [AGENT NEEDS HUMAN INPUT] 🤔")
    print(f"Agent's Question: {question}")
    return "Human input unavailable in benchmark mode. Proceed using available files only and do not wait for user input."

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

from pathlib import Path

def write_file(file_path: str, file_content: str, overwrite: bool = False) -> str:
    """
    Create a NEW file (default) or overwrite an existing file only if overwrite=True.
    """
    full_path = Path.cwd() / file_path

    file_exists = full_path.exists()

    print("\n📝 [AGENT ASKS TO WRITE FILE] 📝")
    print(f"FULL PATH: {full_path}")
    print(f"EXISTS ALREADY?: {file_exists}")
    print(f"OVERWRITE FLAG?: {overwrite}")
    print(f"FILE CONTENT:\n---\n{file_content}\n---")

    if file_exists and not overwrite:
        msg = (
            "❌ Refusing to overwrite an existing file. "
            "Use edit_file(...) for targeted edits, or call write_file(..., overwrite=True)."
        )
        print(msg)
        return msg

    #user_response = input("Do you approve? (yes/no): ")
    user_response = "yes"
    if user_response.lower().strip() != "yes":
        print("❌ User denied execution.")
        return f"Command execution denied by the user:\n{user_response}"

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        success_message = f"✅ Successfully wrote content to {full_path}"
        print(success_message)
        return success_message
    except Exception as e:
        error_message = f"❌ Error writing file to {full_path}: {e}"
        print(error_message)
        return error_message


def read_file(file_path: str, max_chars: int = 20000) -> str:
    """
    Read a text file (truncated) so the agent can make targeted edits.
    """
    full_path = Path.cwd() / file_path

    if not full_path.exists():
        return f"Error: File not found: {full_path}"
    if full_path.is_dir():
        return f"Error: Path is a directory: {full_path}"

    try:
        content = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = full_path.read_text(encoding="latin-1")

    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n... [TRUNCATED {len(content)-max_chars} chars] ..."
    return content



def edit_file(
    file_path: str,
    edit_type: str,
    *,
    old_text: str = None,
    new_text: str = None,
    start_marker: str = None,
    end_marker: str = None,
    anchor: str = None,
    insert_text: str = None,
    count: int = 1,
) -> str:
    """
    Targeted edits WITHOUT overwriting the whole file.
    Shows a unified diff and requires approval.

    edit_type:
      - "replace"
      - "replace_between" (markers kept; content between replaced)
      - "insert_after"
      - "insert_before"
      - "append"
      - "prepend"
    """
    full_path = Path.cwd() / file_path

    if not full_path.exists():
        return f"Error: File not found: {full_path}"
    if full_path.is_dir():
        return f"Error: Path is a directory: {full_path}"

    try:
        original = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        original = full_path.read_text(encoding="latin-1")

    edited = original

    if edit_type == "replace":
        if old_text is None or new_text is None:
            return "Error: replace requires old_text and new_text."
        if old_text not in edited:
            return "Error: old_text not found."
        edited = edited.replace(old_text, new_text, count)

    elif edit_type == "replace_between":
        if start_marker is None or end_marker is None or new_text is None:
            return "Error: replace_between requires start_marker, end_marker, and new_text."
        s = edited.find(start_marker)
        if s == -1:
            return "Error: start_marker not found."
        e = edited.find(end_marker, s + len(start_marker))
        if e == -1:
            return "Error: end_marker not found (after start_marker)."
        between_start = s + len(start_marker)
        between_end = e
        edited = edited[:between_start] + new_text + edited[between_end:]

    elif edit_type in ("insert_after", "insert_before"):
        if anchor is None or insert_text is None:
            return f"Error: {edit_type} requires anchor and insert_text."
        idx = edited.find(anchor)
        if idx == -1:
            return "Error: anchor not found."
        insert_at = idx + len(anchor) if edit_type == "insert_after" else idx
        edited = edited[:insert_at] + insert_text + edited[insert_at:]

    elif edit_type == "append":
        if insert_text is None:
            return "Error: append requires insert_text."
        if edited and not edited.endswith("\n"):
            edited += "\n"
        edited += insert_text

    elif edit_type == "prepend":
        if insert_text is None:
            return "Error: prepend requires insert_text."
        edited = insert_text + ("" if insert_text.endswith("\n") else "\n") + edited

    else:
        return f"Error: Unknown edit_type '{edit_type}'."

    if edited == original:
        return "No changes made."

    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            edited.splitlines(),
            fromfile=str(full_path) + " (before)",
            tofile=str(full_path) + " (after)",
            lineterm="",
        )
    )

    print("\n✍️ [AGENT PROPOSES A FILE EDIT] ✍️")
    print(f"FULL PATH: {full_path}")
    print(f"DIFF:\n---\n{diff}\n---")

    #user_response = input("Do you approve this edit? (yes/no): ")
    user_response = "yes"
    if user_response.lower().strip() != "yes":
        print("❌ User denied edit.")
        return f"Edit denied by the user:\n{user_response}"

    try:
        full_path.write_text(edited, encoding="utf-8")
        msg = f"✅ Successfully edited {full_path}"
        print(msg)
        return msg
    except Exception as e:
        return f"❌ Error writing edited file to {full_path}: {e}"

data_summarizer_prompt = (
"You are a careful technical paper summarizer. "
"Summarize this chunk while preserving dataset names, code/data availability clues, "
"repository/archive mentions, DOIs, accession numbers, and any described download procedures."
)

normal_summarization_prompt = (
"You are a helpful research assistant."
"Summarize the following text from a technical paper/document. Capture key methodologies, specific metrics, results, and conclusions. Do not lose specific data points."
)


def read_and_summarize_pdf(file_path: str, summarizer_model: str="gpt-4o", for_data: bool=False) -> str:
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
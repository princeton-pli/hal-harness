# main_agent.py

import os
import json
import re
from openai import OpenAI
from hal.benchmarks.replicatorbench.constants import API_KEY, EVALUATE_GENERATE_EXECUTE_CONSTANTS
import logging
import sys
import tiktoken
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set to DEBUG during development to see everything
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO) 
logger.addHandler(console_handler)

client = OpenAI(api_key=API_KEY) 
from hal.benchmarks.replicatorbench.validator.file_utils import read_txt, read_csv, read_json, read_pdf, read_docx # Keep save_output here if the agent orchestrates saving
from hal.benchmarks.replicatorbench.validator.file_utils import load_dataset, get_dataset_head, get_dataset_shape, get_dataset_description, get_dataset_info
from hal.benchmarks.replicatorbench.validator.file_utils import read_image, list_files_in_folder, ask_human_input

MAX_TOKENS = 20000

def count_tokens_in_messages(messages, model_name="gpt-4o"):
    """Counts the number of tokens in a list of messages for a given model."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = 0
    for message in messages:
        # Each message has a role, content, and potentially a name
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens -= 1  # role/name is always followed by token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def read_log(file_path, model_name="gpt-4o"):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_log =  f.read()
    encoding = tiktoken.encoding_for_model(model_name)
    log_token_count =  len(encoding.encode(full_log))
    if log_token_count > MAX_TOKENS:
        log_lines = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                log_lines.append(line)
        processed_chunks = []
        chunk_size = 500
        print("NUMBER OF LOG LINES", len(log_lines))
        # Loop through the list in steps of 50
        for i in range(0, len(log_lines), chunk_size):
            # Get the chunk of 50 lines (e.g., 0:50, 50:100, etc.)
            chunk = log_lines[i : i + chunk_size]
            
            # Join the lines in that chunk into a single string
            combined_string = "".join(chunk)
            
            # Add the combined string to your new list
            processed_chunks.append(combined_string)
            
        summary_messages = [
            {
                "role": "system",
                "content": """You are an effective log reader. 
                Your task is to process a specific chunk of a long log. 
                Generate a concise but informative summary of the current chunk. 
                Only output the summary without anything else.
                """.strip()
            }
        ]
        
        summarized_content = ""
        for chunk_id, chunk in enumerate(processed_chunks, start=1):
            
            current_messages = copy.deepcopy(summary_messages)
            current_messages.append({
                "role": "user",
                "content": f"CURRENT CHUNK:\n {chunk}"
            })
            chunk_summary_completion = client.chat.completions.create(
                                        model="gpt-4o",
                                        temperature=0,
                                        messages=current_messages)
            chunk_summary = chunk_summary_completion.choices[0].message.content
            summary_messages.append({
                "role": "system",
                "content": chunk_summary   
            })
            summarized_content += chunk_summary + "\n"
            print(f"SUMMARIZED CHUNK {chunk_id} out of {len(processed_chunks)} chunks", chunk_summary)
        return summarized_content
    else:
        return full_log
                


class Agent:
    def __init__(self, system="", session_state={}):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
        self.session_state = session_state
        
    def count_tokens_in_messages(self, model_name="gpt-4o"):
        """Counts the number of tokens in a list of messages for a given model."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = 0
        for message in self.messages:
            # Each message has a role, content, and potentially a name
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens -= 1  # role/name is always followed by token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    # Example usage:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I assist you?"}
    ]

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def summarize_messages(self, history):
        messages = copy.deepcopy(history)
        summary_prompt = """
        Your conversation history with the user and the environment will now be soon refreshed. 
        Generate a detailed summary to help you accomplish the main task.
        Simply output the summary without saying anything else.
        """
        
        messages.append({
            "role": "user",
            "content": summary_prompt
        })
        
        summary_completion = client.chat.completions.create(
                                    model="gpt-4o",
                                    temperature=0,
                                    messages=messages)
        return summary_completion.choices[0].message.content

    def execute(self):
        token_count = count_tokens_in_messages(self.messages)
        print(f"Total tokens in messages: {token_count}")
        if  token_count > MAX_TOKENS:
            summary = self.summarize_messages(history=self.messages[:-1])
            self.messages = [
                self.messages[0],
                {
                    "role": "assistant",
                    "content": summary
                },
                self.messages[-1]
            ]
        completion = client.chat.completions.create(
                                model="gpt-4o",
                                temperature=0,
                                messages=self.messages)
        return completion.choices[0].message.content
    
    def _execute_tool_call(self, known_actions, action, action_input_str):
        """
        Robustly parse tool arguments the LLM may emit as:
          - a dict JSON string:        '{"k":"v"}'
          - a JSON-encoded JSON string: "\"{\\\"k\\\":\\\"v\\\"}\""
          - a raw string (path, etc.)
        Backwards-compatible: if parsing fails, we pass the original string like before.
        """
        s = (action_input_str or "").strip()

        # Strip common code-fence wrapping without being strict
        if s.startswith("```") and s.endswith("```"):
            s = s.strip("`").strip()

        parsed = None

        # First attempt: parse JSON once
        try:
            tmp = json.loads(s)
            # If the first parse yields a string (JSON-encoded JSON), try a second pass
            if isinstance(tmp, str):
                try:
                    parsed = json.loads(tmp)
                except Exception:
                    parsed = tmp  # leave as string if inner parse fails
            else:
                parsed = tmp
        except json.JSONDecodeError:
            parsed = s  # not JSON → treat as raw string (legacy behavior)

        observation = None
        if isinstance(parsed, dict):
            # Dict -> kwargs (preserve your dataset-session_state special case)
            if "dataset" in action:
                observation = known_actions[action](self.session_state, **parsed)
            else:
                observation = known_actions[action](**parsed)
        else:
            # String -> single positional arg (legacy behavior)
            try:
                if "dataset" in action:
                    observation = known_actions[action](self.session_state, parsed)
                else:
                    observation = known_actions[action](parsed)
            except Exception as e:
                print(e)

        return observation


# --- Agent System Prompt ---
agent_prompt = """
You are an advanced research assistant specialized in replicating some focal claim in a research paper.
You operate in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output an Answer in JSON format.

Use Thought to describe your reasoning about the question and what actions you need to take.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

1. list_files_in_folder:
    e.g. list_files_in_folder: "data/study_A/datasets"
    Description: Lists all files within a specified folder
    Returns: Names of all files within the specified folder with their names as a single string,
    with each file separated by a comma.

2.  read_txt:
    e.g. read_txt: "data/study_X/abstract.txt"
    Description: Reads the plain text content of a file with .txt or .do (Stata do-file) extensions. This is the default reader if a specific file type is not recognized.
    Returns: The content of the file as a string.

3.  read_pdf:
    e.g. read_pdf: "data/study_Y/methods.pdf"
    Description: Extracts and reads the text content from a PDF (.pdf) file.
    Returns: The extracted text content of the PDF as a string.

4.  read_json:
    e.g. read_json: "data/study_Z/config.json"
    Description: Reads and parses a JSON (.json) file.
    Returns: The content of the JSON file as a Python dictionary (which will be converted to string representation for observation).
    
5. read_docx: 
    e.g. `read_docx: "data/study_Z/protocol.docx"`
    * Description: Extracts and reads the text content from a Microsoft Word (.docx) file.
    * Returns: The extracted text content of the file as a string.
    
6. read_log:
    e.g. `read_log: "data/study_Z/design.log"`
    * Description: Extracts and reads the text content from a log file. If a log is too long, a shorter version, where the full log is separated into chunks. Each chunk is summarized and then combined into a overall summary of log content. 
    * Returns: Full or summarized content of the log file.

7. read_image:
   e.g read_image: "data/study_T/image.png"
   Description: Take in an input image of type .png, .jpeg, .jpg, .webp, or .gif and describe in natural language what the image is about.
   Returns: Textual description of the provided image

8. Dataset Related Tools
   7a.  load_dataset:
    * e.g. `load_dataset: "data/study_A/patient_records.csv"` or  `load_dataset: "data/study_A/patient_records.xlsx"`
    * Description: Loads a dataset from a CSV or Excel file into memory for analysis. This function must be called successfully on a file path before any other `get_dataset_*` tools can be used on it.
    * Returns: A string confirming that the dataset was loaded successfully, or an error message if it failed.

   7b.  get_dataset_head:    
    * e.g. `get_dataset_head: "data/study_A/patient_records.csv"`
    * Description: Retrieves the first 5 rows of a previously loaded CSV dataset. This is useful for quickly inspecting the data's structure, column names, and sample values.
    * Returns: A string containing the first 'n' rows of the dataset in a comma-separated format.

   7c.  get_dataset_shape:
    * e.g. `get_dataset_shape: "data/study_A/patient_records.csv"`
    * Description: Gets the dimensions (number of rows, number of columns) of a previously loaded CSV dataset.
    * Returns: A string representing a tuple, for example, "(150, 4)", indicating (rows, columns).

   7d.  get_dataset_description:
    * e.g. `get_dataset_description: "data/study_A/patient_records.csv"`
    * Description: Calculates descriptive statistics for the numerical columns of a loaded CSV dataset. This includes count, mean, standard deviation, min, max, and percentiles.
    * Returns: A string containing a summary table of the descriptive statistics.

9.  get_dataset_info:
    
    * e.g. `get_dataset_info: "data/study_A/patient_records.csv"`
    * Description: Provides a concise technical summary of a loaded CSV dataset, including column names, data types (e.g., integer, float), and the number of non-missing values for each column.
    * Returns: A string containing the full summary information of the dataset.
    
10. ask_human_input:
    * e.g. `ask_human_input: "Need access permission to download data, please download it and give me the path to the downloaded folder"`
    * Description: Asks a clarifying question to the human user and waits for their text response. Use this tool only when you are stuck, if the instructions are ambiguous, or if you need external information you cannot find in the files.
    * Returns: The human's raw text response as a string.

Remember, you don't have to read all provided files if you don't think they are necessary to fill out the required JSON.

Example Session:

Question: Extract information about the original paper and claim to be replicated from the provided files and fill out this JSON template
    {
      "statement": "The main claim made by the original study.",
      "hypothesis": "A testable hypothesis based on the claim.",
      "original_coefficient": "Numeric value indicating strength/direction of effect.",
      "original_p_value": "P-value for testing statistical significance.",
      "direction": "Positive, negative, or null effect.",
      "study_type": "Type of study (Experimental, Observational, Meta-Analysis)."
    }
You will have access to the following documents:
1. original_paper.pdf: The pdf file containing the full text of the original paper 
2. initial_details.txt: A document containing the following details: (1) the focal claim from the original that needs to be replicated.

Thought: The required JSON centers around the main claim. I need to determine what the claim is from initial_detailst.txt. I should use the 'read_txt' tool.
Action: read_txt: initial_details.txt
PAUSE

You will be called again with this:

Observation:[CLAIM]
The relationship between violence and election fraud follows an inverted U-shape: fraud increases with violence up to a certain level, then decreases.

You then output:

Thought: I now know about the claim to be replicated. I need to look for additional information about the claim from the full paper. I should use the 'read_pdf' tool.
Action: read_pdf: original.pdf
PAUSE

You will be called again with this:
Observation: [FULL PAPER PDF redacted here]

You then output:
Answer: {
    "statement": "The relationship between violence and election fraud follows an inverted U-shape: fraud increases with violence up to a certain level, then decreases.",
    "hypothesis": [
      "H1: The linear association between violence and election fraud will be positive.",
      "H* (SCORE focal test): The quadratic association between violence and election fraud will be negative."
    ],
    "original_coefficients": {
        "linear_term": 8.477,
        "squared_term": -13.748
    },
    "original_p_value": {
        "linear_term": "<0.05",
        "squared_term": "<0.01"
    },
    "direction": "Inverted U-shape effect",
    "study_type": "Observational"
  }

Remember that at the end of each response, you must decide whether to run ONE out of allowed actions or output a final Answer. 
That is, your message must end with either "Action: [Your next action]" Or "Answer: [Your final answer]"
""".strip()

# Map action names to their functions
known_actions = {
    "list_files_in_folder": list_files_in_folder,
    "read_txt": read_txt,
    "read_csv": read_csv,
    "read_pdf": read_pdf,
    "read_json": read_json,
    "read_docx": read_docx,
    "read_log": read_log,
    "read_image": read_image,
    "load_dataset": load_dataset,
    "get_dataset_head": get_dataset_head, 
    "get_dataset_shape": get_dataset_shape, 
    "get_dataset_description": get_dataset_description, 
    "get_dataset_info": get_dataset_info,
    "ask_human_input": ask_human_input,
}

action_re = re.compile(r'^Action: (\w+): (.*)$', re.MULTILINE) # Use re.MULTILINE for multiline parsing
def save_output(extracted_json, study_path):
    final_output = {
        "stage": "execute",
        **extracted_json
    }
    output_path = os.path.join(study_path,"llm_eval", "execute_llm_eval.json")
    extracted_json = final_output
    with open(output_path, 'w') as f:
        json.dump(extracted_json, f, indent=2)

    logger.info(f"Interpret stage output saved to {output_path}")
    
def query_agent(question: str, max_turns: int = 20, study_path_for_saving=None):
    """
    Main function to query the agent and orchestrate the extraction process.
    """
    i = 0
    bot = Agent(agent_prompt, session_state = {"analyzers": {}})
    next_prompt = question

    final_extracted_data = {} # To accumulate results
    
    MAX_DISPLAY_PROMPT_LEN = 2000

    while i < max_turns:
        i += 1
        logger.info(f"\n--- Turn {i} ---")
        # print(f"Agent input: {next_prompt}")
        display_prompt = next_prompt
        if len(display_prompt) > MAX_DISPLAY_PROMPT_LEN:
            display_prompt = display_prompt[:MAX_DISPLAY_PROMPT_LEN] + "\n... (truncated for display)"
        logger.info(f"\n***Agent input: {display_prompt}")

        result = bot(next_prompt) # Get LLM's thought/action/answer
        logger.info(f"\n***Agent output:\n{result}")

        # Check if the LLM provided a final answer
        if "Answer:" in result:
            try:
                answer_match = re.search(r'Answer:\s*(\{.*?\})\s*$', result, re.DOTALL)
                if answer_match:
                    json_answer_str = answer_match.group(1).strip()
                else:
                    json_answer_str = result.split("Answer:", 1)[1].strip()
                    if json_answer_str.strip().startswith('{') and json_answer_str.strip().endswith('}'):
                            pass # Looks like valid JSON, proceed
                    else:
                        logger.warning(f"Warning: Answer found but doesn't look like clean JSON: {json_answer_str[:200]}...")
                        # Try to find the JSON part more aggressively
                        json_start = json_answer_str.find('{')
                        json_end = json_answer_str.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            json_answer_str = json_answer_str[json_start : json_end + 1]
                        else:
                            raise ValueError("Could not find a valid JSON structure after 'Answer:'")
                json_start = json_answer_str.find('{')
                json_end = json_answer_str.rfind('}')
                if json_start == -1 or json_end == -1 or json_end < json_start:
                    raise ValueError("Could not find a valid JSON object (missing curly braces) after cleaning.")

                final_answer = json.loads(json_answer_str[json_start : json_end + 1])
                logger.info("\n--- Final Answer ---")
                logger.info(json.dumps(final_answer, indent=2))
                # Agent decides when to save the output now
                if study_path_for_saving:
                    save_output(final_answer, study_path_for_saving)
                logger.info("Process completed")
                return final_answer
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing final JSON answer: {e}")
                logger.error(f"Raw answer: {json_answer_str}")
                return {"error": "Failed to parse final answer JSON"}
            except Exception as e:
                logger.error(f"An error occurred processing final answer: {e}")
                return {"error": str(e)}
        else:
            actions_matches = [
                action_re.match(line)
                for line in result.split('\n')
                if action_re.match(line)
            ]

            if actions_matches:
                # There is an action to run
                match = actions_matches[0]
                action, action_input_str = match.groups()

                logger.info(f" -- Running Action: {action} with input: {action_input_str}")

                if action not in known_actions:
                    logger.error(f"Unknown action: {action}: {action_input_str}") 
                    raise Exception(f"Unknown action: {action}: {action_input_str}")

                observation = bot._execute_tool_call(known_actions, action, action_input_str)

                # print(f"Observation: {observation}")
                next_prompt = f"Observation: {observation}"
            else:
                logger.warning("Agent did not propose an action. Terminating.")
                # If the agent doesn't provide an action or an answer, something is wrong or it's stuck.
                return {"error": "Agent did not provide a recognized action or final answer."}

    print("Max turns reached. Agent terminated without a final answer.")
    return {"error": "Max turns reached without a final answer."}


def build_file_description(available_files, file_path):
    desc = ""
    for file_id, (file_name, file_desc) in enumerate(available_files.items(), start=1):
        desc += f"{file_id}. {os.path.join(file_path, file_name)}: {file_desc}\n"
    return desc

def _configure_file_logging(study_path: str):
    """
    Configures a file handler for the logger, saving logs to the study_path.
    This function should be called once the study_path is known (e.g., at the start of run_extraction).
    It first removes any existing FileHandlers to avoid duplicate logging if called multiple times.
    """
    # Remove any existing FileHandlers attached to this logger
    # This prevents creating multiple log files or appending to old ones if run_extraction is called multiple times.
    for handler in list(logger.handlers): # Use list() to iterate over a copy, safe to modify
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close() # Important: close the file handle to release the file

    # Construct the log file path within the given study_path
    log_file_name = 'evaluate_execute.log'
    log_directory = os.path.join(study_path, "llm_eval") # Assuming study_path is already the directory where you want the log
    os.makedirs(log_directory, exist_ok=True)
    log_file_full_path = os.path.join(log_directory, log_file_name)

    # Ensure the directory exists before trying to write the log file
    os.makedirs(os.path.dirname(log_file_full_path), exist_ok=True)

    # Create a new FileHandler
    file_handler = logging.FileHandler(log_file_full_path, mode='a') # 'a' for append
    file_handler.setFormatter(formatter) # Use the globally defined formatter
    file_handler.setLevel(logging.DEBUG) # File logs everything (DEBUG level)
    logger.addHandler(file_handler)

    logger.info(f"File logging configured to: '{log_file_full_path}'.")


def run_evaluate_execute(study_path, show_prompt=False):
    _configure_file_logging(study_path)
    # Load json template
    logger.info(f"Starting execution evaluation for study path: {study_path}")
    eval_prompt_template = read_txt(EVALUATE_GENERATE_EXECUTE_CONSTANTS['prompt_template'])
    rubric_schema =  read_json(EVALUATE_GENERATE_EXECUTE_CONSTANTS['json_template'])
    claim_docs_for_evaluator = build_file_description(EVALUATE_GENERATE_EXECUTE_CONSTANTS['claim_files'], study_path)
    agent_docs_for_evaluator = build_file_description(EVALUATE_GENERATE_EXECUTE_CONSTANTS['agent_files'], study_path)
    

    variables = {
        'rubric_schema': rubric_schema,
        'claim_docs_for_evaluator': claim_docs_for_evaluator,
        'agent_docs_for_evaluator': agent_docs_for_evaluator,
    }

    query_question = eval_prompt_template.format(**variables)
    
    
    query_agent(
        query_question,
        study_path_for_saving=study_path,
    )
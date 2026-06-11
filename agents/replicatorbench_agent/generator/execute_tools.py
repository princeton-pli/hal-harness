import base64
from openai import OpenAI
import os
import json
import pandas as pd
from core.constants import API_KEY
from typing import Dict, Any, Optional, Tuple
import io # Add this import at the top of your file
import shlex
import subprocess

def run_shell_command(command: str) -> str:
    """
    Executes a shell command in the local terminal after receiving human confirmation.

    Args:
        command (str): The complete shell command to execute (e.g., "python3 my_script.py --arg value").

    Returns:
        str: The combined standard output and standard error from the command, or a rejection message.
    """
    # 1. Ask for human confirmation, showing the exact command
    print(f"\n🤔 [HUMAN CONFIRMATION REQUIRED] 🤔")
    print(f"Agent wants to execute the command: `{command}`")
    user_response = input("Do you approve? (yes/no): ")

    # 2. Check the user's response
    if user_response.lower().strip() != 'yes':
        print("❌ User denied execution.")
        return "Command execution denied by the user."

    print(f"✅ User approved. Executing command...")
    try:
        # 3. Execute the command securely
        # shlex.split handles arguments with spaces correctly
        args = shlex.split(command)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False # Don't raise an exception on errors
        )
    
        # 4. Return the full output to the agent
        output = f"Exit Code: {result.returncode}\n---STDOUT---\n{result.stdout}\n---STDERR---\n{result.stderr}"
        return output.strip()

    except FileNotFoundError:
        return f"Error: The command '{args[0]}' was not found. Make sure it's installed and in your system's PATH."
    except Exception as e:
        return f"An error occurred while executing the command: {e}"

def run_stata_do_file(file_path: str) -> str:
    """
    Executes a Stata .do file in batch mode after human confirmation,
    captures the output from the corresponding .log file, and returns it as a string.

    Args:
        file_path (str): The path to the Stata .do file.

    Returns:
        str: The full content of the generated .log file, or an error message.
    """
    # 1. Determine the expected log file path from the do-file path
    base_name, _ = os.path.splitext(file_path)
    log_path = base_name.split("/")[-1] + ".log"

    # NOTE: The Stata executable might have a different name on your system (e.g., 'stata-se', 'stata')
    command = f"stata-mp -b do {file_path}"

    # 2. Get human confirmation before executing
    print(f"\n🤔 [HUMAN CONFIRMATION REQUIRED] 🤔")
    print(f"Agent wants to execute the Stata script: `{file_path}`")
    user_response = input(f"This will run the command: `{command}`\nDo you approve? (yes/no): ")

    if user_response.lower().strip() != 'yes':
        return "Command execution denied by the user."
    
    try:
        # 3. Execute the Stata command
        print(f"✅ User approved. Executing Stata script...")
        args = shlex.split(command)
        result = subprocess.run(args, capture_output=True, text=True, check=False)

        # If Stata itself threw an error, return that for debugging
        if result.returncode != 0:
            return f"Stata execution failed.\n---STDERR---\n{result.stderr}"

        # 4. Read the entire contents of the generated log file
        print(f"Execution finished. Reading output from '{log_path}'...")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            log_content = log_file.read()

        # 5. (Good practice) Clean up the log file
        os.remove(log_path)
        
        return log_content

    except FileNotFoundError:
        return f"Error: Could not find the generated log file at '{log_path}'. Make sure Stata is installed and the do-file path is correct."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

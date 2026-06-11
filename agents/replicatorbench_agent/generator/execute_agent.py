# generator/execute_react/agent.py

import os
import json
import logging
from typing import Dict, Any

from core.constants import GENERATE_EXECUTE_REACT_CONSTANTS
from core.actions import base_known_actions, get_execute_tool_definitions
from core.agent import run_react_loop, save_output
from core.prompts import PREAMBLE, EXECUTE, EXAMPLE, EXECUTE_CODE_MODE_POLICY
from core.utils import build_file_description, configure_file_logging, get_logger
from info_extractor.file_utils import read_json

# Execute-stage-only tools
from generator.execute_tools import (
    run_shell_command, run_stata_do_file
)
from generator.orchestrator_tool import (
    orchestrator_generate_dockerfile,
    orchestrator_build_image,
    orchestrator_run_container,
    orchestrator_plan,
    orchestrator_preview_entry,
    orchestrator_execute_entry,
    orchestrator_stop_container,
)

logger, formatter = get_logger()
system_prompt = "\n\n".join([PREAMBLE, EXECUTE, EXAMPLE])

# Map action names to their functions
known_actions = {
    **base_known_actions(),
    "run_shell_command": run_shell_command,
    "run_stata_do_file": run_stata_do_file,

    # Orchestrator tools
    "orchestrator_generate_dockerfile": orchestrator_generate_dockerfile,
    "orchestrator_build_image": orchestrator_build_image,
    "orchestrator_run_container": orchestrator_run_container,
    "orchestrator_plan": orchestrator_plan,
    "orchestrator_preview_entry": orchestrator_preview_entry,
    "orchestrator_execute_entry": orchestrator_execute_entry,
    "orchestrator_stop_container": orchestrator_stop_container,
}

CHECKPOINT_MAP = {
    # PHASE 1
    "orchestrator_generate_dockerfile": "1. Generate Dockerfile",
    "orchestrator_build_image":         "2. Build Image",
    
    # PHASE 2
    "orchestrator_run_container":       "3. Start Container",
    "orchestrator_plan":                "4. Plan & Preview",
    "orchestrator_preview_entry":       "4. Plan & Preview",
    
    # PHASE 3
    "ask_human_input":                  "5. Human Approval",
    
    # PHASE 4
    "orchestrator_execute_entry":       "6. Execute Code",
    
    # PHASE 5
    "orchestrator_stop_container":      "7. Stop Container"
}

def _on_final(ans: dict):
    # Canonical output
    save_output(
        ans,
        study_path=study_path,
        filename="execution_results.json",
        stage_name="execute",
    )
    # Mode-specific copy for side-by-side comparisons
    save_output(
        ans,
        study_path=study_path,
        filename=f"execution_results__{code_mode}.json",
        stage_name="execute",
    )

def run_execute(study_path: str, show_prompt: bool = False, templates_dir: str = "./templates", tier="easy", code_mode: str = "python", model_name: str="gpt-4o"):
    configure_file_logging(logger, study_path, f"execute_{tier}__{code_mode}.log")
    logger.info(f"[agent] dynamic orchestrator run loop for: {study_path}")

    schema_path = os.path.join(templates_dir, "execute_schema.json")
    prev_files = GENERATE_EXECUTE_REACT_CONSTANTS.get("files", {}).copy()
    prev_template = GENERATE_EXECUTE_REACT_CONSTANTS.get("json_template")

    try:
        # Update available files context
        GENERATE_EXECUTE_REACT_CONSTANTS["files"] = {
            "replication_info.json": "Configuration for Docker base image, packages, and code entry point. MODIFY THIS if Docker build fails.",
            "execution_result.json": "Output generated after running 'orchestrator_execute_entry'. Contains stdout/stderr. Read this to debug build errors.",
            "_runtime/Dockerfile": "The generated Dockerfile.",
        }
        GENERATE_EXECUTE_REACT_CONSTANTS["json_template"] = schema_path

        code_policy = EXECUTE_CODE_MODE_POLICY.get(code_mode, EXECUTE_CODE_MODE_POLICY["python"])

        instruction = f"""
Your goal is to successfully execute the replication study inside a Docker container.
You are operating in a DEBUG LOOP. You must assess the result of every action. 

{code_policy}

File operations policy:
 - To modify existing files: ALWAYS call read_file first, then use edit_file for targeted changes.
 - write_file is for creating new files. It will refuse to overwrite unless overwrite=True.
 - Only use write_file(overwrite=True) when you intend to replace the entire file contents.

If an action fails (e.g., Docker build error, Missing Dependency, Code crash), you MUST:
1. Analyze the error message in the Observation.
2. Use `write_file` to FIX the issue (e.g., rewrite `replication_info.json` to add packages, or rewrite the code files). Remember that write_file will overwrite any existing content in the provided file_path if existing. When you use the tool, the provided path file_path to the tool MUST be the study path given to you. But to access other files within the file_content argument, you MUST use the container's directories "app/data". 
3. RETRY the failed step.

**Phases of Execution:**

PHASE 1: BUILD ENVIRONMENT
1. `orchestrator_generate_dockerfile`: Creates _runtime/Dockerfile from replication_info.json.
2. `orchestrator_build_image`: Builds the image.
   * IF BUILD FAILS: Read the error log. It usually means a missing system package or R/Python library. Edit `replication_info.json` to add the missing dependency, regenerate the Dockerfile, and rebuild.

PHASE 2: PREPARE RUNTIME
3. `orchestrator_run_container`: Mounts the code and data and starts the container.
4. `orchestrator_plan` & `orchestrator_preview_entry`: Verify what will run.

PHASE 3: HUMAN APPROVAL (Strict Check)
5. Before running the actual analysis code, you MUST Ask the human:
   Action: ask_human_input: "Ready to execute command: <COMMAND>. Approve? (yes/no)"
   * If they say "no", stop the container and fill the output JSON with status "cancelled".
   * If they say "yes", proceed to Phase 4.

PHASE 4: EXECUTE & DEBUG
6. `orchestrator_execute_entry`: Runs the code.
   * IF EXECUTION FAILS (exit_code != 0): 
     - Read the `stderr` in the observation.
     - Identify if it is a code error or missing library.
     - Use `write_file` to fix the script or `replication_info.json`.
     - If you changed dependencies, you must go back to Phase 1 (Rebuild).
     - If you only changed code, you can retry `orchestrator_execute_entry`.

PHASE 5: FINALIZE
7. `orchestrator_stop_container`: Cleanup.
8. Parse `execution_result.json` and output the Answer in the following required JSON schema.
{json.dumps(read_json(schema_path))}

Current Study Path: "{study_path}"
Start by generating the Dockerfile.

Remember, every response needs to have one of the two following formats:
----- FORMAT 1 (For when you need to call actions to help accomplish the given task) -------
Thought: [Your thinking/planning process for completing the task based on interactions so far]
Action: [call next action to help you solve the task]
PAUSE
----- FORMAT 2 (For when you are ready to give a final response)-------
Thought: [Your thinking/planning process for completing the task based on interactions so far]
Answer: [Execute necessary next action to help you solve the task]
""".strip()

        if show_prompt:
            logger.info("\n\n===== Agent Input (truncated) =====\n" + instruction[:2000])
        print(f"\n\nmodel name for execute agent: {model_name}\n\n")
        tool_definitions = get_execute_tool_definitions()
        return run_react_loop(
            system_prompt,
            known_actions,
            tool_definitions,
            instruction,
            session_state={"analyzers": {}},
            study_path=study_path,
            stage_name="generate-execute",
            checkpoint_map=CHECKPOINT_MAP,
            on_final=lambda ans: save_output(
                ans,
                study_path=study_path,
                filename="execution_results.json",
                stage_name="execute"
            ),
            model_name=model_name
        )
    finally:
        GENERATE_EXECUTE_REACT_CONSTANTS["files"] = prev_files
        if prev_template:
            GENERATE_EXECUTE_REACT_CONSTANTS["json_template"] = prev_template

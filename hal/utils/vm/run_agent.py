#!/usr/bin/env python3
"""
Static entrypoint for running an agent on a VM.

Reads configuration from environment variables (loaded from .env and run_agent.env).
Required env vars: RUN_ID, AGENT_FUNCTION, TASK_ID.
Input data and agent kwargs are read from input.json and agent_args.json in the
current working directory (/home/agent).
"""
import os
import sys
import json
import importlib.util
import traceback

from dotenv import load_dotenv

# Load harness .env and run-specific env (written by VM manager)
load_dotenv("/home/agent/.env")
load_dotenv("/home/agent/run_agent.env")

RUN_ID = os.environ.get("RUN_ID")
AGENT_FUNCTION = os.environ.get("AGENT_FUNCTION")
TASK_ID = os.environ.get("TASK_ID")

missing = [k for k in ("RUN_ID", "AGENT_FUNCTION", "TASK_ID") if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)


def main():
    import weave

    weave.init(RUN_ID)

    with open("/home/agent/input.json", "r") as f:
        input_data = json.load(f)

    with open("/home/agent/agent_args.json", "r") as f:
        agent_args = json.load(f)

    module_name, _, function_name = AGENT_FUNCTION.rpartition(".")
    if not module_name or not function_name:
        print(f"ERROR: AGENT_FUNCTION must be 'module.function', got: {AGENT_FUNCTION!r}", file=sys.stderr)
        sys.exit(1)

    agent_path = os.path.join("/home/agent", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, function_name)

    with weave.attributes({"weave_task_id": TASK_ID}):
        result = agent(input_data, **agent_args)

    with open("/home/agent/output.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("/home/agent/error.log", "w") as f:
            f.write(f"ERROR: {e}\n")
            f.write(traceback.format_exc())
        raise

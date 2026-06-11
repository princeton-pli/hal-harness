import json
import os
import shutil
import subprocess
from typing import Dict, Any
import glob
from typing import Tuple
import logging
from typing import Optional, List

STAGE_TO_MAKE = {
    "extract": "extract-stage1",
    "web_search": "web-search",
    "design": "design-easy",
    "execute": "execute-easy",
    "interpret": "interpret-easy",
}

STAGE_TO_OUTPUT = {
    "extract": ("post_registration.json", "post_registration_path"),
    "web_search": ("merged-urls.json", "merged_urls_path"),
    "design": ("replication_info.json", "replication_info_path"),
    "execute": ("execution_results.json", "execution_results_path"),
    "interpret": ("interpret_results.json", "interpret_results_path"),
}

STAGE_SUFFIXES = ["_extract", "_web_search", "_design", "_execute", "_interpret"]

def _base_id_from_task_id(task_id: str) -> str:
    for suf in STAGE_SUFFIXES:
        if task_id.endswith(suf):
            return task_id[: -len(suf)]
    return task_id

def _find_parent_dir_containing(capsule_dir: str, filename: str) -> Optional[str]:
    for root, _dirs, files in os.walk(capsule_dir):
        if filename in files:
            return root
    return None

def _pick_source_dir(capsule_dir: str, required_files: Optional[List[str]] = None) -> str:

    input_dir = os.path.join(capsule_dir, "input")
    if os.path.isdir(input_dir):
        return input_dir

    if required_files:
        for rf in required_files:
            parent = _find_parent_dir_containing(capsule_dir, rf)
            if parent:
                return parent

    entries = [e for e in os.listdir(capsule_dir) if not e.startswith(".")]
    if len(entries) == 1:
        only_path = os.path.join(capsule_dir, entries[0])
        if os.path.isdir(only_path):
            return only_path

    return capsule_dir

def _symlink_into(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.startswith("."):
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)

        if os.path.islink(dst) or os.path.isfile(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)

        os.symlink(src, dst)


def _infer_stage(task_id: str) -> str:
    if task_id.endswith("_web_search"):
        return "web_search"
    for suf in ("extract", "design", "execute", "interpret"):
        if task_id.endswith("_" + suf):
            return suf
    return "unknown"

def _infer_capsule_id(task_id: str, stage: str) -> str:
    if stage == "web_search":
        return task_id[: -len("_web_search")]
    if stage in ("extract", "design", "execute", "interpret"):
        return task_id[: -(len(stage) + 1)]  
    return task_id

def _load_current_task() -> Tuple[str, Dict[str, Any]]:

    with open("/workspace/input.json", "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("input.json is empty or not a dict")
    task_id = next(iter(payload.keys()))
    task = payload[task_id]
    return task_id, task

def _ensure_tasks_json() -> None:

    dst = "/workspace/tasks.json"
    if os.path.exists(dst):
        return

    candidates = [os.path.join(os.path.dirname(__file__), "tasks.json")]

    for src in candidates:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logging.info("Copied tasks.json into place: %s -> %s", src, dst)
            return

    raise FileNotFoundError(
        "setup_script.sh requires /workspace/tasks.json, but it is missing and no bundled copy was found. "
        "Bundle tasks.json with the agent (e.g., agents/replicatorbench_agent/tasks.json)."
    )

def _prepare_study_dir(task_id: str, stage: str) -> str:

    base_id = _base_id_from_task_id(task_id)
    study_dir = f"/workspace/{base_id}"

    def _has_real_inputs(dir_path: str) -> bool:
        return (
            os.path.isdir(dir_path)
            and os.path.exists(os.path.join(dir_path, "original_paper.pdf"))
            and os.path.exists(os.path.join(dir_path, "initial_details.txt"))
        )

    if _has_real_inputs(study_dir):
        logging.info("Using existing populated study_dir=%s", study_dir)
        return study_dir

    setup_path = "/workspace/setup_script.sh"
    sentinel = "/tmp/replicatorbench_setup_ran"

    setup_stdout = ""
    setup_stderr = ""

    if os.path.exists(setup_path) and not os.path.exists(sentinel):
        logging.info("study_dir missing/empty (%s). Will rerun setup: %s", study_dir, setup_path)
        try:
            with open(sentinel, "w") as f:
                f.write("1")

            _ensure_tasks_json()
            proc = subprocess.run(
                ["bash", setup_path],
                cwd="/workspace",
                env=dict(os.environ),
                text=True,
                capture_output=True,
            )
            setup_stdout = proc.stdout or ""
            setup_stderr = proc.stderr or ""

            if setup_stdout:
                logging.info("setup stdout (tail): %s", setup_stdout[-2000:])
            if setup_stderr:
                logging.info("setup stderr (tail): %s", setup_stderr[-2000:])

            if proc.returncode != 0:
                raise RuntimeError(f"setup_script.sh exited with code {proc.returncode}")

        except Exception as e:

            logging.exception("Setup rerun failed: %s", e)

        if _has_real_inputs(study_dir):
            logging.info("setup produced populated study_dir=%s", study_dir)
            return study_dir

    required_files = ["initial_details.txt", "original_paper.pdf"] if stage in ("extract", "web_search") else []

    candidates = [
        f"/workspace/{task_id}",          
        f"/workspace/capsules/{task_id}", 
        f"/workspace/capsules/{base_id}", 
    ]

    cwd0 = os.getcwd()
    if os.path.basename(cwd0) == task_id:
        candidates.append(cwd0)

    def _candidate_ok(path: str) -> bool:
        if not os.path.isdir(path):
            return False

        if not required_files:
            return any((not x.startswith(".")) for x in os.listdir(path))

        for rf in required_files:
            if _find_parent_dir_containing(path, rf) is not None:
                return True
        return False

    capsule_dir = next((c for c in candidates if _candidate_ok(c)), None)
    if capsule_dir is None:
        ws_listing = os.listdir("/workspace") if os.path.isdir("/workspace") else None
        caps_listing = (
            os.listdir("/workspace/capsules") if os.path.isdir("/workspace/capsules") else None
        )

        raise FileNotFoundError(
            f"No study dir and no capsule/workdir found for task_id={task_id}, stage={stage}.\n"
            f"Tried: {candidates}\n"
            f"/workspace contains: {ws_listing}\n"
            f"/workspace/capsules contains: {caps_listing}\n"
            f"setup_script.sh exists: {os.path.exists(setup_path)} (ran={os.path.exists(sentinel)})\n"
            f"setup stdout tail: {setup_stdout[-2000:]}\n"
            f"setup stderr tail: {setup_stderr[-2000:]}\n"
        )

    source_dir = _pick_source_dir(capsule_dir, required_files=required_files)

    os.makedirs(study_dir, exist_ok=True)
    _symlink_into(source_dir, study_dir)

    logging.info("task_id=%s stage=%s base_id=%s", task_id, stage, base_id)
    logging.info("candidates=%s", candidates)
    logging.info("picked capsule_dir=%s", capsule_dir)
    logging.info("source_dir=%s", source_dir)
    logging.info("study_dir=%s listing=%s", study_dir, os.listdir(study_dir))

    return study_dir

def _ensure_api_env(env: Dict[str, str]) -> Dict[str, str]:

    if env.get("API_KEY") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["API_KEY"]
    return env

def _run_make_target(cwd: str, env: Dict[str, str], study_dir: str, model: str, target: str) -> None:
    cmd = ["make", target, f"STUDY={study_dir}", f"MODEL={model}"]
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def _ensure_prereqs(stage: str, cwd: str, env: Dict[str, str], study_dir: str, model: str) -> None:
    post_reg = os.path.join(study_dir, "post_registration.json")
    merged = os.path.join(study_dir, "merged-urls.json")
    repl = os.path.join(study_dir, "replication_info.json")
    exec_res = os.path.join(study_dir, "execution_results.json")

    if stage in ("extract", "web_search", "design", "execute", "interpret"):
        if not os.path.exists(post_reg):
            _run_make_target(cwd, env, study_dir, model, "extract-stage1")

        if not os.path.exists(merged):
            _run_make_target(cwd, env, study_dir, model, "web-search")

            if not os.path.exists(merged):
                candidates = []
                for root, _, files in os.walk(study_dir):
                    for name in files:
                        if name in ("merged-urls.json", "merged_urls.json"):
                            candidates.append(os.path.join(root, name))

                if len(candidates) == 1:
                    if os.path.abspath(candidates[0]) != os.path.abspath(merged):
                        shutil.copy2(candidates[0], merged)
                elif len(candidates) > 1:
                    raise RuntimeError(f"web-search produced multiple URL files: {candidates}")
                else:
                    raise RuntimeError(f"web-search ran but did not create {merged}")

    if stage in ("design", "execute", "interpret"):
        if not os.path.exists(repl):
            _run_make_target(cwd, env, study_dir, model, "design-easy")

    if stage in ("execute", "interpret"):
        if not os.path.exists(exec_res):
            _run_make_target(cwd, env, study_dir, model, "execute-easy")

def _stage_output_path(stage: str, study_dir: str) -> str:
    stage_to_file = {
        "extract": "post_registration.json",
        "web_search": "merged-urls.json",
        "design": "replication_info.json",
        "execute": "execution_results.json",
        "interpret": "interpret_results.json",
    }
    if stage not in stage_to_file:
        raise ValueError(f"Unknown stage: {stage}")
    return os.path.join(study_dir, stage_to_file[stage])

def run(*args, **kwargs) -> Dict[str, str]:
    task_id, _task = _load_current_task()
    stage = _infer_stage(task_id)
    if stage not in STAGE_TO_MAKE:
        raise RuntimeError(f"Unknown task stage for task_id={task_id}")

    study_dir = _prepare_study_dir(task_id=task_id, stage=stage)

    if stage == "web_search":
        details = os.path.join(study_dir, "initial_details.txt")
        if not os.path.exists(details):
            raise FileNotFoundError(f"Missing {details}. Study dir has: {os.listdir(study_dir)}")


    make_target = STAGE_TO_MAKE[stage]
    out_file, out_key = STAGE_TO_OUTPUT[stage]
    out_path = os.path.join(study_dir, out_file)

    env = _ensure_api_env(dict(os.environ))

    cwd = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(cwd, "Makefile")):
        candidate = "/workspace/agents/replicatorbench_agent"
        if os.path.exists(os.path.join(candidate, "Makefile")):
            cwd = candidate
        else:
            raise FileNotFoundError(f"Makefile not found in {cwd} or {candidate}")

    try:
        subprocess.run(["make", "check-deps"], cwd=cwd, env=env, check=True)
    except Exception:
        subprocess.run(["make", "install-deps"], cwd=cwd, env=env, check=True)

    model = env.get("MODEL") or "gpt-4o"

    _ensure_prereqs(stage, cwd, env, study_dir, model)

    stage_output = _stage_output_path(stage, study_dir)
    if not os.path.exists(stage_output):
        cmd = ["make", make_target, f"STUDY={study_dir}", f"MODEL={model}"]
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    else:
        print(f"[INFO] Skipping {make_target}; output already exists at {stage_output}")
    

    if stage == "web_search" and not os.path.exists(out_path):
        metadata_path = os.path.join(study_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"web_search did not produce {out_path} and metadata.json is missing at {metadata_path}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        payload = meta.get("extract_stage_find_urls")

        if payload is None:
            for v in meta.values():
                if isinstance(v, dict) and "extract_stage_find_urls" in v:
                    payload = v["extract_stage_find_urls"]
                    break

        if payload is None:
            raise RuntimeError(
                f"web_search updated metadata.json but extract_stage_find_urls key not found; cannot build {out_path}. "
                f"Keys: {list(meta.keys())[:50]}"
            )

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    if not os.path.exists(out_path):
        raise RuntimeError(f"Expected output not found: {out_path}")

    return {task_id: {out_key: out_path}}
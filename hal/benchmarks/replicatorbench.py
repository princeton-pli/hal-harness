# hal/benchmarks/replicatorbench.py

import ast
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_benchmark import BaseBenchmark
import os as _os
__path__ = [_os.path.join(_os.path.dirname(__file__), "replicatorbench")]

from hal.benchmarks.replicatorbench.validator.evaluate_extract import extract_from_human_postreg
from hal.benchmarks.replicatorbench.validator.evaluate_design import extract_from_human_prereg
from hal.benchmarks.replicatorbench.validator.evaluate_execute import run_evaluate_execute
from hal.benchmarks.replicatorbench.validator.evaluate_interpret import extract_from_human_report

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

GT_REPO_SLUG = os.getenv("REPLICATORBENCH_GT_REPO_SLUG", "").strip()
GT_REF_OVERRIDE = os.getenv("REPLICATORBENCH_GT_REF", "").strip()
GT_TOKEN = os.getenv("REPLICATORBENCH_GT_TOKEN", "").strip()  # optional, for private repos

REPLICATORBENCH_EVAL_MODE = os.getenv("REPLICATORBENCH_EVAL_MODE", "offline").strip().lower()


JsonObj = Union[Dict[str, Any], List[Any]]


class ReplicatorBenchmark(BaseBenchmark):
    """
    ReplicatorBench benchmark.

    Inside HAL sandbox/container:
      - setup.sh downloads capsules to:
          /workspace/capsules/<task_id>/

    Agent output:
      - agent writes stage outputs under:
          /workspace/<task_id>/
        Required outputs:
          Extract stage:
            - post_registration.json
            - merged-urls.json
          Design stage:
            - replication_info.json
          Execute stage:
            - execution_results.json
          Interpret stage:
            - interpret_results.json

    Agent return to harness:
      - we accept either:
          { "<task_id>": { "execution_results_path": "<abs path>", ... } }
        This evaluator will fall back to the canonical paths under /workspace/<task_id>/.

    Current scoring behavior:
      - Placeholder "eval_summary-like" scoring for all stages:
          stage score = 1.0 if required file exists and parses as JSON (dict/list as appropriate), else 0.0
      - Execute stage can additionally use offline GT compare (met/unmet) if GT exists.
    """

    TARGET_ROOT = "/workspace"

    REQUIRED_OUTPUTS = {
        "extract": ["post_registration.json", "merged-urls.json"],
        "design": ["replication_info.json"],
        "execute": ["execution_results.json"],
        "interpret": ["interpret_results.json"],
    }

    POINTER_KEYS = {
        "post_registration.json": ["post_registration_path"],
        "merged-urls.json": ["merged_urls_path", "merged-urls_path"],
        "replication_info.json": ["replication_info_path"],
        "execution_results.json": ["execution_results_path"],
        "interpret_results.json": ["interpret_results_path"],
    }

    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        if not hasattr(self, "benchmark_name"):
            self.benchmark_name = "replicatorbench"

        self.requires_sandbox = True
        self.setup_script = "hal/benchmarks/replicatorbench/setup.sh"
        self.config = config or {}

        self.benchmark: Dict[str, Dict[str, Any]] = {}

        self._ground_truth: Dict[str, Dict[str, Any]] = {}

        self._bench_dir = os.path.join(os.path.dirname(__file__), "replicatorbench")
        self._tasks_path = os.path.join(self._bench_dir, "tasks.json")

        self._gt_path = os.path.join(self._bench_dir, "ground_truth", "execute_ground_truth.json")

        try:
            super().__init__(
                agent_dir,
                config,
                requires_sandbox=self.requires_sandbox,
                setup_script=self.setup_script,
            )
        except TypeError:
            super().__init__(agent_dir, config)

        self._load_tasks()
        self._load_ground_truth()

    def _load_tasks(self) -> None:
        if not os.path.exists(self._tasks_path):
            raise FileNotFoundError(f"[replicatorbench] tasks.json not found at: {self._tasks_path}")

        with open(self._tasks_path, "r") as f:
            payload = json.load(f)

        if not (isinstance(payload, dict) and isinstance(payload.get("tasks"), list)):
            raise ValueError(
                "[replicatorbench] tasks.json must be a dict with key 'tasks' holding a list, "
                'e.g. {"tasks": [ ... ]}'
            )

        tasks: List[Dict[str, Any]] = payload["tasks"]
        seen_ids = set()

        self._task_meta: Dict[str, Dict[str, str]] = {}

        def _infer_stage(task_id: str) -> str:
            if task_id.endswith("_web_search"):
                return "web_search"
            for suf in ("_extract", "_design", "_execute", "_interpret"):
                if task_id.endswith(suf):
                    return suf.lstrip("_")
            return "unknown"

        def _infer_capsule_id(task_id: str, stage: str) -> str:
            if stage == "web_search":
                return task_id[: -len("_web_search")]
            if stage in ("extract", "design", "execute", "interpret"):
                return task_id[: -(len(stage) + 1)]  # strip "_<stage>"
            return task_id

        for t in tasks:
            if not isinstance(t, dict):
                raise ValueError("[replicatorbench] Each entry in tasks must be an object/dict.")

            task_id = str(t.get("task_id") or "").strip()
            if not task_id:
                raise ValueError("[replicatorbench] Task entry missing required field: task_id")
            if task_id in seen_ids:
                raise ValueError(f"[replicatorbench] Duplicate task_id in tasks.json: {task_id}")
            seen_ids.add(task_id)

            capsule_type = str(t.get("capsule_type") or "").strip()
            capsule_url = str(t.get("capsule_url") or "").strip()
            prompt = str(t.get("prompt") or "").strip()
            gpu = bool(t.get("gpu", False))

            if not capsule_type:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_type")
            if not capsule_url:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: capsule_url")
            if not prompt:
                raise ValueError(f"[replicatorbench:{task_id}] Missing required field: prompt")

            stage = _infer_stage(task_id)
            capsule_id = str(t.get("capsule_id") or "").strip()
            if not capsule_id:
                capsule_id = _infer_capsule_id(task_id, stage)

            capsule_ref = str(t.get("capsule_ref") or "").strip()
            capsule_subdir = str(t.get("capsule_subdir") or "").strip()

            self._task_meta[task_id] = {
                "capsule_id": capsule_id,
                "stage": stage,
                "capsule_url": capsule_url,
                "capsule_ref": capsule_ref,
                "capsule_subdir": capsule_subdir,
            }

            self.benchmark[task_id] = {
                "prompt": prompt,
                "files": {},   # capsules are handled by setup.sh; no per-task file mapping needed here
                "gpu": gpu,
            }

    def _load_ground_truth(self) -> None:
        if not os.path.exists(self._gt_path):
            self._ground_truth = {}
            return

        with open(self._gt_path, "r") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError(f"[replicatorbench] Ground truth must be a JSON object/dict: {self._gt_path}")

        gt: Dict[str, Dict[str, Any]] = {}
        for task_id, entry in obj.items():
            if isinstance(entry, dict):
                gt[str(task_id)] = entry
        self._ground_truth = gt

    #  accept dict, JSON string, or Python dict string
    def _parse_agent_obj(self, solution: Any) -> Optional[Dict[str, Any]]:

        if solution is None:
            return None
        if isinstance(solution, dict):
            return solution
        if not isinstance(solution, str):
            solution = str(solution)
        s = solution.strip()
        if not s:
            return None

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        return None

    def _read_json_any(self, path: str) -> Optional[JsonObj]:
        try:
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, (dict, list)):
                return obj
            return None
        except Exception:
            return None

    def _to_float(self, v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("<"):
                s = s[1:].strip()
            if s.endswith("%"):
                s = s[:-1].strip()
            try:
                return float(s)
            except Exception:
                return None
        return None

    def _extract_primary_effect(self, execution_results: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        results = execution_results.get("results", {}) if isinstance(execution_results, dict) else {}
        findings = results.get("findings_summary", [])
        if not (isinstance(findings, list) and findings and isinstance(findings[0], dict)):
            return None, None, ""
        primary = findings[0]
        est = self._to_float(primary.get("value"))
        pval = self._to_float(primary.get("p_value"))
        direction = str(primary.get("direction", "")).strip().lower()

        if not direction and est is not None:
            if est > 0:
                direction = "positive"
            elif est < 0:
                direction = "negative"
            else:
                direction = "null"

        return est, pval, direction

    def _canonical_task_dir(self, task_id: str) -> str:

        meta = getattr(self, "_task_meta", {}) or {}
        capsule_id = (meta.get(task_id) or {}).get("capsule_id") or task_id
        return os.path.join(self.TARGET_ROOT, capsule_id)


    # Stage checks 
    def _stage_check_json(
        self,
        task_id: str,
        stage: str,
        filename: str,
        parsed_ptr: Optional[Dict[str, Any]],
        allow_list: bool = False,
    ) -> Dict[str, Any]:
        """
        want to check:
          - file exists
          - JSON parses
          - type is dict
        """
        path = os.path.join(self._canonical_task_dir(task_id), filename)
        info: Dict[str, Any] = {"file": filename, "path": path, "ok": False, "error": None}

        if not os.path.exists(path):
            info["error"] = "missing_file"
            return info

        obj = self._read_json_any(path)
        if obj is None:
            info["error"] = "invalid_json"
            return info

        if isinstance(obj, dict):
            
            schema_map = {
                "post_registration.json": "post_registration_schema.json",
                "replication_info.json": "replication_info_schema.json",
                "execution_results.json": "execute_schema.json",
                "interpret_results.json": "interpret_schema.json",
            }
            schema_name = schema_map.get(filename)

            if schema_name:
                schema_path = os.path.join(
                    os.path.dirname(__file__), "replicatorbench", "templates", schema_name
                )
                try:
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema = json.load(f)
                    props = schema.get("properties", {})
                    if isinstance(props, dict) and props:
                        overlap = [k for k in props.keys() if k in obj]
                        if not overlap:
                            info["error"] = "missing_schema_properties"
                            return info

                    required = schema.get("required", [])
                    missing = [k for k in required if k not in obj]
                    if missing:
                        info["error"] = f"missing_required_keys:{missing}"
                        return info
                except Exception as e:
                    info["error"] = f"schema_check_failed:{type(e).__name__}"
                    return info

            info["ok"] = True
            return info
        if allow_list and isinstance(obj, list):
            info["ok"] = True
            return info

        info["error"] = "wrong_json_type"
        return info
    
    def summarize_eval_execute(self, eval_data):
        eval_scores = {}
        for sub_stage, sub_stage_eval_data in eval_data.items():
            eval_scores[f"execute_{sub_stage}"] = {
                "aspect_scores": {}
            }
            sub_stage_scores = []
            for aspect in sub_stage_eval_data:
                aspect_scores = []
                for rubric_id, rubric_info in sub_stage_eval_data[aspect].items():
                    aspect_scores.append(rubric_info['score'])
                aspect_avg = sum(aspect_scores)/len(aspect_scores)
                eval_scores[f"execute_{sub_stage}"]["aspect_scores"][aspect] = aspect_avg
                sub_stage_scores.append(aspect_avg)
            eval_scores[f"execute_{sub_stage}"]["avg_score"] = sum(sub_stage_scores)/len(sub_stage_scores)
        return eval_scores

    def _to_float_or_none(self, x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip()
            if s.upper() in {"NA", "N/A", ""}:
                return None
            return float(s)  # will still raise if it's something else
        return None

    def summarize_eval_scores(self, study_path):
        stages = ["extract", "design", "execute", "interpret"]
        eval_summary = {}
        for stage in stages:
            with open(f"{study_path}/llm_eval/{stage}_llm_eval.json") as f:
                eval_json = json.load(f)
            if stage == "execute":
                eval_data = {
                    "design": eval_json["evaluate_design"],
                    "execute": eval_json["execute"] 
                }
                eval_summary.update(self.summarize_eval_execute(eval_data))
            else:
                aspect_totals = {}
                for eval_field, eval_info in eval_json.items():
                    aspect = eval_field.split(".")[0]
                    if aspect not in aspect_totals:
                        aspect_totals[aspect] = [0.0, 0.0]
                    score = self._to_float_or_none(eval_info.get("score"))
                    if score is None:
                        continue
                    aspect_totals[aspect][0] += score
                    aspect_totals[aspect][1] += 3.0

                eval_summary[stage] = {"aspect_scores": {}}
                stage_scores = []
                for aspect, (score_sum, max_sum) in aspect_totals.items():
                    aspect_avg = (score_sum / max_sum) if max_sum else 0.0
                    eval_summary[stage]["aspect_scores"][aspect] = aspect_avg
                    stage_scores.append(aspect_avg)

                eval_summary[stage]["avg_score"] = (
                    sum(stage_scores) / len(stage_scores) if stage_scores else 0.0
                )
        with open(f"{study_path}/llm_eval/eval_summary.json", "w") as fout:
            json.dump(eval_summary, fout, indent =2)
            
        return eval_summary
            
    
    def _evaluate_task(
        self, task_id: str, parsed_ptr: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate by looking for outputs in:
        /workspace/<capsule_id>/
        and inputs in:
        /workspace/capsules/<capsule_id>/.

        Ground-truth (GT) is fetched at evaluation time into:
        /workspace/_replicatorbench_gt/<capsule_id>/
        """
        import shutil
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        meta = getattr(self, "_task_meta", {}) or {}
        task_meta = (meta.get(task_id) or {})

        capsule_id = task_meta.get("capsule_id") or task_id

        capsule_inputs_dir = os.path.join(self.TARGET_ROOT, "capsules", capsule_id)

        out_dir = self._canonical_task_dir(task_id)

        extract_checks = [
            self._stage_check_json(task_id, "extract", "post_registration.json", parsed_ptr, allow_list=False),
            self._stage_check_json(task_id, "extract", "merged-urls.json", parsed_ptr, allow_list=True),
        ]
        design_checks = [
            self._stage_check_json(task_id, "design", "replication_info.json", parsed_ptr, allow_list=False),
        ]
        execute_check = self._stage_check_json(task_id, "execute", "execution_results.json", parsed_ptr, allow_list=False)
        interpret_checks = [
            self._stage_check_json(task_id, "interpret", "interpret_results.json", parsed_ptr, allow_list=False),
        ]

        def _repo_slug_from_url(capsule_url: str) -> str:
            # https://github.com/org/repo(.git) -> org/repo
            s = (capsule_url or "").strip()
            if s.startswith("https://github.com/"):
                s = s[len("https://github.com/") :]
            s = s.rstrip("/")
            if s.endswith(".git"):
                s = s[:-4]
            return s

        def _raw_url(repo_slug: str, ref: str, path_in_repo: str) -> str:
            return f"https://raw.githubusercontent.com/{repo_slug}/{ref}/{path_in_repo}"

        def _download(url: str, dst_path: str, token: str = "") -> bool:
            try:
                headers = {}
                if token:
                    headers["Authorization"] = f"token {token}"
                req = Request(url, headers=headers)
                with urlopen(req) as r, open(dst_path, "wb") as f:
                    f.write(r.read())
                return True
            except (HTTPError, URLError):
                return False

        GT_REPO_SLUG = os.getenv("REPLICATORBENCH_GT_REPO_SLUG", "").strip()
        GT_REF_OVERRIDE = os.getenv("REPLICATORBENCH_GT_REF", "").strip()
        GT_TOKEN = os.getenv("REPLICATORBENCH_GT_TOKEN", "").strip()

        capsule_url = (task_meta.get("capsule_url") or "").strip()
        capsule_ref = (GT_REF_OVERRIDE or (task_meta.get("capsule_ref") or "")).strip()
        capsule_subdir = (task_meta.get("capsule_subdir") or "").strip()

        gt_ready = False
        gt_note = None

        gt_subdir = None
        if capsule_subdir.endswith("/input"):
            gt_subdir = capsule_subdir[: -len("/input")] + "/gt"


        gt_dir = os.path.join(self.TARGET_ROOT, "_replicatorbench_gt", capsule_id)
        os.makedirs(gt_dir, exist_ok=True)

        gt_post_reg_local = os.path.join(gt_dir, "expected_post_registration.json")
        gt_prereg_pdf = os.path.join(gt_dir, "human_preregistration.pdf")
        gt_prereg_docx = os.path.join(gt_dir, "human_preregistration.docx")
        gt_report_pdf = os.path.join(gt_dir, "human_report.pdf")
        gt_report_docx = os.path.join(gt_dir, "human_report.docx")

        gt_prereg_path = None
        gt_report_path = None

        if GT_REPO_SLUG:
            repo_slug = GT_REPO_SLUG
        else:
            repo_slug = _repo_slug_from_url(capsule_url) if capsule_url else ""

        stage = (task_meta.get("stage") or "").strip()

        if stage != "interpret":
            gt_note = f"Skipping GT download for stage='{stage}'. GT is fetched only for interpret."
        else:
            if repo_slug and capsule_ref and gt_subdir:
                # expected_post_registration.json
                if not os.path.exists(gt_post_reg_local):
                    _download(
                        _raw_url(repo_slug, capsule_ref, f"{gt_subdir}/expected_post_registration.json"),
                        gt_post_reg_local,
                        token=GT_TOKEN,
                    )

                # prereg (try pdf then docx)
                if not (os.path.exists(gt_prereg_pdf) or os.path.exists(gt_prereg_docx)):
                    ok = _download(
                        _raw_url(repo_slug, capsule_ref, f"{gt_subdir}/human_preregistration.pdf"),
                        gt_prereg_pdf,
                        token=GT_TOKEN,
                    )
                    if not ok:
                        _download(
                            _raw_url(repo_slug, capsule_ref, f"{gt_subdir}/human_preregistration.docx"),
                            gt_prereg_docx,
                            token=GT_TOKEN,
                        )

                # report (try pdf then docx)
                if not (os.path.exists(gt_report_pdf) or os.path.exists(gt_report_docx)):
                    ok = _download(
                        _raw_url(repo_slug, capsule_ref, f"{gt_subdir}/human_report.pdf"),
                        gt_report_pdf,
                        token=GT_TOKEN,
                    )
                    if not ok:
                        _download(
                            _raw_url(repo_slug, capsule_ref, f"{gt_subdir}/human_report.docx"),
                            gt_report_docx,
                            token=GT_TOKEN,
                        )

                gt_prereg_path = gt_prereg_pdf if os.path.exists(gt_prereg_pdf) else (
                    gt_prereg_docx if os.path.exists(gt_prereg_docx) else None
                )
                gt_report_path = gt_report_pdf if os.path.exists(gt_report_pdf) else (
                    gt_report_docx if os.path.exists(gt_report_docx) else None
                )

                gt_ready = True
            else:
                gt_note = (
                    "GT download skipped (missing repo_slug/capsule_ref/gt_subdir). "
                    "Ensure _task_meta includes capsule_url, capsule_ref, capsule_subdir, "
                    "and capsule_subdir ends with '/input'."
                )

        # Run GT-based evaluators ONLY during interpret stage
        if (task_meta.get("stage") or "").strip() == "interpret":
            try:
                if extract_checks[0]["ok"] and os.path.exists(gt_post_reg_local):
                    extract_from_human_postreg(extract_checks[0]["path"], gt_post_reg_local, out_dir)
            except Exception:
                pass

            try:
                if design_checks[0]["ok"] and gt_prereg_path:
                    extract_from_human_prereg(design_checks[0]["path"], gt_prereg_path, out_dir)
            except Exception:
                pass

            try:
                if interpret_checks[0]["ok"] and gt_report_path:
                    extract_from_human_report(interpret_checks[0]["path"], gt_report_path, out_dir)
            except Exception:
                pass

        try:
            if execute_check["ok"]:
                if os.path.isdir(capsule_inputs_dir):
                    need = ["original_paper.pdf", "replication_data", "initial_details.txt"]
                    for name in need:
                        src = os.path.join(capsule_inputs_dir, name)
                        dst = os.path.join(out_dir, name)
                        if not os.path.exists(src) or os.path.exists(dst):
                            continue
                        try:
                            os.symlink(src, dst)
                        except Exception:
                            if os.path.isfile(src):
                                try:
                                    shutil.copy2(src, dst)
                                except Exception:
                                    pass

                run_evaluate_execute(out_dir)
        except Exception:
            pass

        # -----------------------------
        # Summarize if llm_eval stage files exist (under out_dir)
        # -----------------------------
        llm_eval_dir = os.path.join(out_dir, "llm_eval")
        expected_eval_files = [
            os.path.join(llm_eval_dir, "extract_llm_eval.json"),
            os.path.join(llm_eval_dir, "design_llm_eval.json"),
            os.path.join(llm_eval_dir, "execute_llm_eval.json"),
            os.path.join(llm_eval_dir, "interpret_llm_eval.json"),
        ]

        if os.path.isdir(llm_eval_dir) and all(os.path.exists(p) for p in expected_eval_files):
            eval_summary = self.summarize_eval_scores(out_dir)
        else:
            eval_summary = {
                "note": "llm_eval outputs not complete yet; returning presence/type checks only",
                "checks": {
                    "extract": extract_checks,
                    "design": design_checks,
                    "execute": execute_check,
                    "interpret": interpret_checks,
                },
            }

        if gt_note:
            eval_summary["gt_note"] = gt_note
        else:
            eval_summary["gt"] = {
                "gt_cache_dir": gt_dir,
                "gt_subdir": gt_subdir,
                "repo_slug": repo_slug,
                "ref": capsule_ref,
            }

        return {
            "task_id": task_id,
            "capsule_id": capsule_id,
            "capsule_path": capsule_inputs_dir,
            "output_dir": out_dir,
            "eval_summary": eval_summary,
        }

    # HAL API
    def evaluate_output(self, agent_output: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """
        Evaluate tasks.

        - Calls the team's _evaluate_task_placeholder() unchanged (so validator logic stays aligned).
        - Then augments each result with:
            - eval_summary.stage_scores
            - overall_score
            - correct
        so get_metrics() works.
        """
        results: Dict[str, Any] = {}

        # If agent_output is empty, still evaluate all configured tasks
        if isinstance(agent_output, dict) and len(agent_output) > 0:
            task_ids = [str(tid) for tid in agent_output.keys()]
        else:
            task_ids = list(self.benchmark.keys())

        for task_id in task_ids:
            solution = agent_output.get(task_id) if isinstance(agent_output, dict) else None
            parsed_ptr = self._parse_agent_obj(solution)

            # docker_runner copies container /workspace/. -> temp_dir -> results/<benchmark>/<run_id>/<task_id>/
            host_task_root = None
            if isinstance(agent_output, dict) and task_id in agent_output and isinstance(agent_output[task_id], dict):
                ptr = agent_output[task_id]
                # pick any pointer path that exists
                for v in ptr.values():
                    if isinstance(v, str) and v.strip() and os.path.isabs(v) and os.path.exists(v):
                        host_task_root = os.path.dirname(os.path.dirname(v)) 
                        break

            if host_task_root and os.path.isdir(host_task_root):
                self.TARGET_ROOT = host_task_root

            # Host-side docker eval: point TARGET_ROOT at the copied container workspace for this task
            host_task_root = os.path.join(os.getcwd(), "results", self.benchmark_name, run_id, task_id)
            if os.path.isdir(host_task_root):
                self.TARGET_ROOT = host_task_root

            ev = self._evaluate_task(task_id, parsed_ptr) or {}
            summary = (ev.get("eval_summary") or {})

            # 1) derive stage_complete (0/1) from presence/type checks if available
            stage_complete = {"extract": 0.0, "design": 0.0, "execute": 0.0, "interpret": 0.0}
            checks = summary.get("checks")
            if isinstance(checks, dict):
                try:
                    ex = checks.get("extract", [])
                    de = checks.get("design", [])
                    eu = checks.get("execute", {})
                    it = checks.get("interpret", [])

                    if isinstance(ex, list) and len(ex) > 0:
                        stage_complete["extract"] = 1.0 if all(c.get("ok") for c in ex) else 0.0
                    if isinstance(de, list) and len(de) > 0:
                        stage_complete["design"] = 1.0 if all(c.get("ok") for c in de) else 0.0
                    if isinstance(eu, dict) and ("ok" in eu):
                        stage_complete["execute"] = 1.0 if bool(eu.get("ok")) else 0.0
                    if isinstance(it, list) and len(it) > 0:
                        stage_complete["interpret"] = 1.0 if all(c.get("ok") for c in it) else 0.0
                except Exception:
                    pass

            # 2) if summarize_eval_scores() produced per-stage avg scores, prefer those
            stage_scores = dict(stage_complete)
            if isinstance(summary.get("stage_scores"), dict):
                # already provided by team; trust it
                stage_scores = {
                    k: float(summary["stage_scores"].get(k) or 0.0)
                    for k in stage_scores.keys()
                }
            else:
                # try to map your summarizer output into stage_scores
                try:
                    for st in ("extract", "design", "interpret"):
                        if isinstance(summary.get(st), dict) and summary[st].get("avg_score") is not None:
                            stage_scores[st] = float(summary[st]["avg_score"])
                    exec_parts = []
                    for k in ("execute_design", "execute_execute"):
                        if isinstance(summary.get(k), dict) and summary[k].get("avg_score") is not None:
                            exec_parts.append(float(summary[k]["avg_score"]))
                    if exec_parts:
                        stage_scores["execute"] = sum(exec_parts) / len(exec_parts)
                except Exception:
                    pass

                # inject for get_metrics()
                summary["stage_scores"] = stage_scores
                ev["eval_summary"] = summary

            # 3) add overall_score + correct expected by get_metrics()
            ev["overall_score"] = float(sum(stage_scores.values()) / 4.0)
            ev["correct"] = all(v == 1.0 for v in stage_complete.values())

            results[task_id] = ev

        return results

    # HAL API
    def get_metrics(self, eval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate:
          - pass rate (all 4 stages ok)
          - average overall_score
          - per-stage average score
        """
        task_ids = list(eval_results.keys())
        total = len(task_ids)

        passed = [tid for tid in task_ids if eval_results[tid].get("correct", False)]
        failed = [tid for tid in task_ids if not eval_results[tid].get("correct", False)]

        avg_overall = 0.0
        stage_sums = {"extract": 0.0, "design": 0.0, "execute": 0.0, "interpret": 0.0}

        for tid in task_ids:
            ev = eval_results.get(tid, {}) or {}
            avg_overall += float(ev.get("overall_score") or 0.0)
            summary = (ev.get("eval_summary") or {})
            stage_scores = (summary.get("stage_scores") or {})
            for k in stage_sums.keys():
                stage_sums[k] += float(stage_scores.get(k) or 0.0)

        avg_overall = (avg_overall / total) if total > 0 else 0.0
        stage_avgs = {k: (v / total if total > 0 else 0.0) for k, v in stage_sums.items()}
        pass_rate = (len(passed) / total) if total > 0 else 0.0

        return {
            "pass_rate_all_stages": pass_rate,
            "avg_overall_score": avg_overall,
            "avg_stage_scores": stage_avgs,
            "n_tasks": total,
            "n_passed": len(passed),
            "n_failed": len(failed),
            "passed_tasks": passed,
            "failed_tasks": failed,
            "eval_mode": REPLICATORBENCH_EVAL_MODE,
            "execute_ground_truth_path": self._gt_path,
        }


class ReplicatorBenchEasy(ReplicatorBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "replicatorbench_easy"
        super().__init__(agent_dir, config)


class ReplicatorBenchHard(ReplicatorBenchmark):
    def __init__(self, agent_dir: str, config: Dict[str, Any]):
        self.benchmark_name = "replicatorbench_hard"
        super().__init__(agent_dir, config)

    def _filter_files_dict(self, files_dict: Dict[str, str], task: Dict[str, Any]) -> Dict[str, str]:
        """
        Hard tier: do not provide original paper code.
        """
        filtered: Dict[str, str] = {}
        for target_path, source_path in files_dict.items():
            p = target_path.replace("\\", "/")
            if "/original_code/" in p or p.endswith("/original_code"):
                continue
            filtered[target_path] = source_path
        return filtered
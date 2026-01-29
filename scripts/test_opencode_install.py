#!/usr/bin/env python3
"""
OpenCode Agent – Installation & Smoke-Test Script
==================================================

Reusable diagnostic script that verifies the OpenCode CLI installs correctly
on a clean Linux environment and (optionally) executes a sample prompt.

Designed to run inside a Docker container or any Debian/Ubuntu-based system.
Output is both human-readable (coloured terminal) and machine-parseable (--json).

Phases
------
1. system-deps   – apt-get install git curl procps
2. install-cli   – run the official opencode installer
3. verify-binary – locate binary, print version/help
4. write-config  – write .opencode/config.json with full permissions
5. sample-query  – (optional, needs API key) run a trivial prompt

Examples
--------
    # Install-only (no API key required):
    python test_opencode_install.py

    # Full end-to-end with a sample query:
    ANTHROPIC_API_KEY=sk-... python test_opencode_install.py \\
        --run-query --model anthropic/claude-sonnet-4-20250514

    # Custom prompt, verbose, JSON report:
    python test_opencode_install.py --run-query -v --json \\
        --prompt 'Write "hello" to /workspace/test.txt'

    # Inside Docker via the companion shell script:
    ./test_opencode_install.sh --run-query
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Constants ────────────────────────────────────────────────

OPENCODE_INSTALL_CMD = "curl -fsSL https://opencode.ai/install | bash"

KNOWN_BINARY_LOCATIONS = [
    "~/.local/bin/opencode",
    "~/.opencode/bin/opencode",
    "~/.local/share/opencode/bin/opencode",
    "/usr/local/bin/opencode",
    "/usr/bin/opencode",
    "/opt/homebrew/bin/opencode",
]

OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {
        "*": "allow",
        "external_directory": "allow",
        "doom_loop": "allow",
        "edit": "allow",
        "bash": "allow",
        "webfetch": "allow",
        "websearch": "allow",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "task": "allow",
        "skill": "allow",
    },
}

DEFAULT_SMOKE_PROMPT = (
    'Write the JSON object {"smoke_test": "ok", "value": 42} '
    "to /workspace/smoke_answer.json. "
    "Write ONLY the JSON, no markdown fences."
)


# ── Logging ──────────────────────────────────────────────────

LOG_FMT = "%(asctime)s │ %(levelname)-5s │ %(message)s"
LOG_DATE = "%H:%M:%S"

# ANSI helpers (disabled when not a tty)
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def setup_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FMT, datefmt=LOG_DATE, level=level, stream=sys.stdout)
    return logging.getLogger("opencode-test")


# ── Data Structures ──────────────────────────────────────────


@dataclass
class StepLog:
    """One sub-step inside a phase."""
    action: str
    ok: bool
    output: str = ""
    duration_s: float = 0.0


@dataclass
class PhaseResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    duration_s: float = 0.0
    details: str = ""
    error: str = ""
    steps: List[StepLog] = field(default_factory=list)


@dataclass
class TestReport:
    phases: List[PhaseResult] = field(default_factory=list)
    overall: str = "pass"
    opencode_binary: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: PhaseResult) -> None:
        self.phases.append(result)
        if result.status == "fail":
            self.overall = "fail"


# ── Helpers ──────────────────────────────────────────────────


def run_cmd(
    cmd,
    log: logging.Logger,
    *,
    timeout: int = 300,
    check: bool = False,
    shell: bool = False,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with full logging."""
    display = cmd if isinstance(cmd, str) else " ".join(cmd)
    log.debug(f"  $ {_dim(display)}")

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
            shell=shell,
            env=env,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        log.error(f"  ⏱  Command timed out after {timeout}s: {display}")
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"  ✗  Command failed (rc={e.returncode}): {display}")
        if e.stdout:
            for line in e.stdout.strip().splitlines()[:30]:
                log.debug(f"    stdout: {line}")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[:30]:
                log.debug(f"    stderr: {line}")
        raise

    elapsed = time.monotonic() - t0

    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[:40]:
            log.debug(f"    stdout: {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[:40]:
            log.debug(f"    stderr: {line}")

    log.debug(f"    exit={result.returncode}  elapsed={elapsed:.1f}s")
    return result


def find_opencode_binary(log: logging.Logger) -> Optional[str]:
    """Search for the opencode binary in PATH and known locations."""
    # 1. PATH
    found = shutil.which("opencode")
    if found:
        log.info(f"  Found on PATH: {found}")
        return found

    # 2. Known locations
    for loc in KNOWN_BINARY_LOCATIONS:
        expanded = os.path.expanduser(loc)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            log.info(f"  Found at known location: {expanded}")
            return expanded

    # 3. Broad search
    search_roots = [os.path.expanduser("~"), "/usr/local", "/usr", "/opt"]
    for root in search_roots:
        for sub in ("bin", "local/bin", "local/share/opencode/bin"):
            candidate = os.path.join(root, sub, "opencode")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                log.info(f"  Found via search: {candidate}")
                return candidate

    return None


def collect_environment() -> Dict[str, Any]:
    """Collect environment info for the report."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "arch": platform.machine(),
        "user": os.environ.get("USER", os.environ.get("LOGNAME", "unknown")),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "cwd": os.getcwd(),
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "has_anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "path": os.environ.get("PATH", ""),
    }


# ── Phase Implementations ────────────────────────────────────


def phase_system_deps(log: logging.Logger) -> PhaseResult:
    """Phase 1: Install system dependencies (git, curl, procps)."""
    log.info(_bold("Phase 1: System Dependencies"))
    t0 = time.monotonic()
    steps: List[StepLog] = []

    # Check what's already available
    for binary in ("git", "curl", "ps"):
        path = shutil.which(binary)
        if path:
            log.info(f"  ✓ {binary} already available: {path}")
            steps.append(StepLog(f"check-{binary}", True, path))
        else:
            log.info(f"  ✗ {binary} not found, will install")
            steps.append(StepLog(f"check-{binary}", False, "not found"))

    # Clean stale apt lists (common in GPU images)
    try:
        run_cmd(
            "rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list 2>/dev/null || true",
            log, shell=True, timeout=10,
        )
    except Exception:
        pass

    # apt-get update
    try:
        log.info("  Running apt-get update ...")
        st = time.monotonic()
        r = run_cmd(["apt-get", "update"], log, timeout=120)
        steps.append(StepLog("apt-get-update", r.returncode == 0, "", time.monotonic() - st))
        if r.returncode != 0:
            return PhaseResult("system-deps", "fail", time.monotonic() - t0,
                               error=f"apt-get update failed: {r.stderr[:500]}", steps=steps)
    except Exception as e:
        return PhaseResult("system-deps", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    # apt-get install
    try:
        log.info("  Running apt-get install git curl procps ...")
        st = time.monotonic()
        r = run_cmd(["apt-get", "install", "-y", "git", "curl", "procps"], log, timeout=180)
        steps.append(StepLog("apt-get-install", r.returncode == 0, "", time.monotonic() - st))
        if r.returncode != 0:
            return PhaseResult("system-deps", "fail", time.monotonic() - t0,
                               error=f"apt-get install failed: {r.stderr[:500]}", steps=steps)
    except Exception as e:
        return PhaseResult("system-deps", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    # Verify installed
    all_ok = True
    for binary in ("git", "curl", "ps"):
        path = shutil.which(binary)
        if path:
            log.info(f"  ✓ {binary} → {path}")
            steps.append(StepLog(f"verify-{binary}", True, path))
        else:
            log.error(f"  ✗ {binary} still not found after install!")
            steps.append(StepLog(f"verify-{binary}", False))
            all_ok = False

    elapsed = time.monotonic() - t0
    status = "pass" if all_ok else "fail"
    log.info(f"  Phase 1: {_green('PASS') if all_ok else _red('FAIL')}  ({elapsed:.1f}s)")
    return PhaseResult("system-deps", status, elapsed, steps=steps)


def phase_install_cli(log: logging.Logger) -> PhaseResult:
    """Phase 2: Install OpenCode CLI via official installer."""
    log.info(_bold("Phase 2: Install OpenCode CLI"))
    t0 = time.monotonic()
    steps: List[StepLog] = []

    # Check if already installed
    existing = find_opencode_binary(log)
    if existing:
        log.info(f"  OpenCode already installed: {existing}")
        steps.append(StepLog("pre-check", True, existing))
        elapsed = time.monotonic() - t0
        log.info(f"  Phase 2: {_green('PASS (already installed)')}  ({elapsed:.1f}s)")
        return PhaseResult("install-cli", "pass", elapsed, details=existing, steps=steps)

    # Run installer
    log.info(f"  Running: {OPENCODE_INSTALL_CMD}")
    try:
        st = time.monotonic()
        r = run_cmd(OPENCODE_INSTALL_CMD, log, shell=True, timeout=300)
        install_elapsed = time.monotonic() - st
        steps.append(StepLog("run-installer", r.returncode == 0,
                             r.stdout[-2000:] if r.stdout else "", install_elapsed))

        if r.returncode != 0:
            return PhaseResult("install-cli", "fail", time.monotonic() - t0,
                               error=f"Installer exited {r.returncode}\nstderr: {r.stderr[:1000]}",
                               steps=steps)
    except subprocess.TimeoutExpired:
        return PhaseResult("install-cli", "fail", time.monotonic() - t0,
                           error="Installer timed out (300s)", steps=steps)
    except Exception as e:
        return PhaseResult("install-cli", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    # Locate binary after install
    binary = find_opencode_binary(log)
    if not binary:
        # Extra diagnostics
        log.error("  Binary not found after install. Searching filesystem ...")
        try:
            r = run_cmd(["find", "/", "-name", "opencode", "-type", "f"],
                        log, timeout=30)
            if r.stdout.strip():
                log.info(f"  find results:\n{r.stdout.strip()}")
                steps.append(StepLog("find-binary", True, r.stdout.strip()))
            else:
                steps.append(StepLog("find-binary", False, "nothing found"))
        except Exception:
            pass

        return PhaseResult("install-cli", "fail", time.monotonic() - t0,
                           error="opencode binary not found after install completed",
                           steps=steps)

    steps.append(StepLog("locate-binary", True, binary))
    elapsed = time.monotonic() - t0
    log.info(f"  Phase 2: {_green('PASS')}  ({elapsed:.1f}s) → {binary}")
    return PhaseResult("install-cli", "pass", elapsed, details=binary, steps=steps)


def phase_verify_binary(log: logging.Logger, binary: str) -> PhaseResult:
    """Phase 3: Verify binary responds to basic commands."""
    log.info(_bold("Phase 3: Verify Binary"))
    t0 = time.monotonic()
    steps: List[StepLog] = []

    if not binary:
        log.warning("  Skipped (no binary path)")
        return PhaseResult("verify-binary", "skip", 0, error="no binary")

    # File info
    try:
        stat = os.stat(binary)
        log.info(f"  Binary: {binary}")
        log.info(f"  Size:   {stat.st_size:,} bytes")
        log.info(f"  Mode:   {oct(stat.st_mode)}")
        steps.append(StepLog("stat", True, f"size={stat.st_size} mode={oct(stat.st_mode)}"))
    except Exception as e:
        steps.append(StepLog("stat", False, str(e)))

    # file(1) type check
    try:
        r = run_cmd(["file", binary], log, timeout=10)
        file_type = r.stdout.strip()
        log.info(f"  Type:   {file_type}")
        steps.append(StepLog("file-type", True, file_type))
    except Exception:
        steps.append(StepLog("file-type", False, "file command not available"))

    # Try --version
    version_output = ""
    for flag in ["--version", "version", "-v"]:
        try:
            r = run_cmd([binary, flag], log, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                version_output = r.stdout.strip().splitlines()[0]
                log.info(f"  Version ({flag}): {version_output}")
                steps.append(StepLog("version", True, version_output))
                break
        except Exception:
            continue

    if not version_output:
        log.info("  Version: could not determine (not fatal)")
        steps.append(StepLog("version", True, "undetermined"))

    # Try --help or help
    help_ok = False
    for flag in ["--help", "help", "-h"]:
        try:
            r = run_cmd([binary, flag], log, timeout=15)
            combined = (r.stdout + r.stderr).lower()
            if "usage" in combined or "run" in combined or "opencode" in combined:
                help_ok = True
                log.info(f"  Help ({flag}): responds with usage info")
                steps.append(StepLog("help", True, f"{flag} works"))
                break
        except Exception:
            continue

    if not help_ok:
        log.warning("  Help: no recognisable usage output (not fatal)")
        steps.append(StepLog("help", True, "no usage output"))

    elapsed = time.monotonic() - t0
    log.info(f"  Phase 3: {_green('PASS')}  ({elapsed:.1f}s)")
    return PhaseResult("verify-binary", "pass", elapsed, details=version_output, steps=steps)


def phase_write_config(log: logging.Logger, workspace: str) -> PhaseResult:
    """Phase 4: Init git repo, write project + global config."""
    log.info(_bold("Phase 4: Write Config & Init Workspace"))
    t0 = time.monotonic()
    steps: List[StepLog] = []

    # ── 4a. Initialize git repo (opencode requires it) ──
    try:
        log.info("  Initializing git repo in workspace ...")
        r = run_cmd(["git", "init", workspace], log, timeout=10)
        steps.append(StepLog("git-init", r.returncode == 0, workspace))
        if r.returncode == 0:
            # Create an initial commit so HEAD exists
            run_cmd(["git", "-C", workspace, "config", "user.email", "test@test.com"], log, timeout=5)
            run_cmd(["git", "-C", workspace, "config", "user.name", "Test"], log, timeout=5)
            run_cmd(["git", "-C", workspace, "commit", "--allow-empty", "-m", "init"], log, timeout=10)
            log.info("  ✓ Git repo initialized with initial commit")
            steps.append(StepLog("git-commit", True, "initial commit"))
    except Exception as e:
        log.warning(f"  Git init failed (non-fatal): {e}")
        steps.append(StepLog("git-init", False, str(e)))

    # ── 4b. Write project-level config (.opencode/config.json) ──
    config_dir = os.path.join(workspace, ".opencode")
    config_path = os.path.join(config_dir, "config.json")

    try:
        os.makedirs(config_dir, exist_ok=True)
        steps.append(StepLog("mkdir-project", True, config_dir))

        with open(config_path, "w") as f:
            json.dump(OPENCODE_CONFIG, f, indent=2)
        steps.append(StepLog("write-project-config", True, config_path))

        # Verify it reads back correctly
        with open(config_path, "r") as f:
            loaded = json.load(f)
        assert loaded == OPENCODE_CONFIG, "config round-trip mismatch"
        steps.append(StepLog("verify-project-config", True, f"{len(loaded['permission'])} permission keys"))

        log.info(f"  Project config written: {config_path}")
        log.info(f"  Permissions: {len(loaded['permission'])} keys, all 'allow'")
    except Exception as e:
        log.error(f"  Failed writing project config: {e}")
        return PhaseResult("write-config", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    # ── 4c. Write global config (~/.config/opencode/config.json) ──
    global_config_dir = os.path.expanduser("~/.config/opencode")
    global_config_path = os.path.join(global_config_dir, "config.json")

    try:
        os.makedirs(global_config_dir, exist_ok=True)
        global_config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "anthropic": {},
                "openai": {},
            },
        }
        with open(global_config_path, "w") as f:
            json.dump(global_config, f, indent=2)
        steps.append(StepLog("write-global-config", True, global_config_path))
        log.info(f"  Global config written: {global_config_path}")
    except Exception as e:
        log.warning(f"  Global config write failed (non-fatal): {e}")
        steps.append(StepLog("write-global-config", False, str(e)))

    elapsed = time.monotonic() - t0
    log.info(f"  Phase 4: {_green('PASS')}  ({elapsed:.1f}s)")
    return PhaseResult("write-config", "pass", elapsed, details=config_path, steps=steps)


def phase_sample_query(
    log: logging.Logger,
    binary: str,
    model: str,
    workspace: str,
    prompt: Optional[str] = None,
) -> PhaseResult:
    """Phase 5: Run a sample prompt and verify the output file."""
    log.info(_bold("Phase 5: Sample Query"))
    t0 = time.monotonic()
    steps: List[StepLog] = []

    if not binary:
        return PhaseResult("sample-query", "skip", 0, error="no binary")

    # Check for API key
    has_key = bool(
        os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not has_key:
        log.warning("  No API key found (OPENAI_API_KEY / ANTHROPIC_API_KEY)")
        log.warning("  Skipping sample query – set an API key to enable this phase")
        return PhaseResult("sample-query", "skip", 0,
                           error="no API key in environment", steps=steps)

    prompt = prompt or DEFAULT_SMOKE_PROMPT
    answer_file = os.path.join(workspace, "smoke_answer.json")

    # Clean up previous run
    if os.path.exists(answer_file):
        os.remove(answer_file)

    env = os.environ.copy()
    env["OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS"] = "120000"

    cmd = [
        binary,
        "run",
        "--model", model,
        "--print-logs",
        prompt,
    ]

    log.info(f"  Model:  {model}")
    log.info(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
    log.info(f"  Answer: {answer_file}")
    log.info(f"  Command: {' '.join(cmd)}")
    log.info("  Running opencode (output goes to opencode_exec.log) ...")

    exec_log_path = os.path.join(workspace, "opencode_exec.log")

    try:
        st = time.monotonic()
        with open(exec_log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                timeout=300,
                check=False,
                cwd=workspace,
            )
        query_elapsed = time.monotonic() - st

        # Read and display the log
        exec_output = ""
        if os.path.exists(exec_log_path):
            with open(exec_log_path, "r") as f:
                exec_output = f.read()
            if exec_output.strip():
                log.info(f"  opencode output ({len(exec_output)} chars):")
                for line in exec_output.strip().splitlines()[:50]:
                    log.debug(f"    | {line}")
                if len(exec_output.strip().splitlines()) > 50:
                    log.debug(f"    ... ({len(exec_output.strip().splitlines()) - 50} more lines)")

        steps.append(StepLog("run-query", result.returncode == 0,
                             f"rc={result.returncode}", query_elapsed))

        log.info(f"  CLI exited with code {result.returncode} in {query_elapsed:.1f}s")

        if result.returncode != 0:
            log.error(f"  Output (last 1000 chars): {exec_output[-1000:]}")
            steps.append(StepLog("cli-exit", False, exec_output[-1000:]))
    except subprocess.TimeoutExpired:
        # Read partial output for debugging
        if os.path.exists(exec_log_path):
            with open(exec_log_path, "r") as f:
                partial = f.read()
            log.error(f"  Timed out. Partial output ({len(partial)} chars):")
            for line in partial.strip().splitlines()[-30:]:
                log.error(f"    | {line}")
            steps.append(StepLog("timeout-output", False, partial[-2000:]))
        return PhaseResult("sample-query", "fail", time.monotonic() - t0,
                           error="query timed out (300s)", steps=steps)
    except Exception as e:
        return PhaseResult("sample-query", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    # Check answer file
    if not os.path.exists(answer_file):
        log.error(f"  ✗ Answer file not created: {answer_file}")
        # List workspace for debugging
        try:
            contents = os.listdir(workspace)
            log.info(f"  Workspace contents: {contents}")
            steps.append(StepLog("list-workspace", True, str(contents)))
        except Exception:
            pass
        return PhaseResult("sample-query", "fail", time.monotonic() - t0,
                           error=f"answer file not created: {answer_file}", steps=steps)

    # Read and validate
    try:
        with open(answer_file, "r") as f:
            raw = f.read()
        log.info(f"  Answer file content: {raw.strip()[:200]}")
        answer = json.loads(raw)
        steps.append(StepLog("read-answer", True, json.dumps(answer)))

        if answer.get("smoke_test") == "ok":
            log.info("  ✓ Smoke test value verified: smoke_test=ok")
            steps.append(StepLog("verify-answer", True, "smoke_test=ok"))
        else:
            log.warning(f"  Answer has unexpected content: {answer}")
            steps.append(StepLog("verify-answer", True, f"unexpected: {answer}"))
    except json.JSONDecodeError as e:
        log.error(f"  ✗ Answer is not valid JSON: {e}")
        log.error(f"  Raw content: {raw[:300]}")
        steps.append(StepLog("read-answer", False, f"invalid JSON: {e}"))
        return PhaseResult("sample-query", "fail", time.monotonic() - t0,
                           error=f"invalid JSON: {e}", steps=steps)
    except Exception as e:
        return PhaseResult("sample-query", "fail", time.monotonic() - t0,
                           error=str(e), steps=steps)

    elapsed = time.monotonic() - t0
    log.info(f"  Phase 5: {_green('PASS')}  ({elapsed:.1f}s)")
    return PhaseResult("sample-query", "pass", elapsed,
                       details=json.dumps(answer), steps=steps)


# ── Summary ──────────────────────────────────────────────────


def print_summary(report: TestReport) -> None:
    """Print a human-readable summary table."""
    print()
    print(_bold("═" * 60))
    print(_bold("  OpenCode Install Test – Summary"))
    print(_bold("═" * 60))

    for phase in report.phases:
        if phase.status == "pass":
            icon = _green("PASS")
        elif phase.status == "fail":
            icon = _red("FAIL")
        else:
            icon = _yellow("SKIP")

        line = f"  {icon}  {phase.name:<20s}  {phase.duration_s:6.1f}s"
        if phase.details:
            line += f"  {_dim(phase.details[:40])}"
        print(line)
        if phase.error:
            print(f"         {_red('Error: ' + phase.error[:70])}")

    print(_bold("─" * 60))
    overall = _green("ALL PASSED") if report.overall == "pass" else _red("FAILED")
    binary_info = report.opencode_binary or "not found"
    print(f"  Overall: {overall}")
    print(f"  Binary:  {binary_info}")
    total = sum(p.duration_s for p in report.phases)
    print(f"  Total:   {total:.1f}s")
    print(_bold("═" * 60))
    print()


# ── Entry Point ──────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test OpenCode CLI installation and basic functionality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        examples:
          %(prog)s                          # install-only test
          %(prog)s --run-query              # full test (needs API key)
          %(prog)s --run-query --json       # JSON report
          %(prog)s -v                       # verbose logging
        """),
    )
    parser.add_argument(
        "--run-query", action="store_true",
        help="Run phase 5: execute a sample prompt (requires API key)",
    )
    parser.add_argument(
        "--model", default="anthropic/claude-sonnet-4-20250514",
        help="Model name for sample query (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Custom prompt for sample query",
    )
    parser.add_argument(
        "--workspace", default="/workspace",
        help="Working directory (default: /workspace)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Print machine-readable JSON report at the end",
    )
    parser.add_argument(
        "--skip-deps", action="store_true",
        help="Skip phase 1 (system deps) – useful if already installed",
    )

    args = parser.parse_args()
    log = setup_logging(args.verbose)

    # Ensure workspace exists
    os.makedirs(args.workspace, exist_ok=True)

    report = TestReport(environment=collect_environment())

    log.info(_bold("OpenCode Agent – Installation & Smoke Test"))
    log.info(f"  Workspace: {args.workspace}")
    log.info(f"  Platform:  {platform.platform()}")
    log.info(f"  Python:    {sys.version.split()[0]}")
    log.info("")

    # ── Phase 1 ──
    if args.skip_deps:
        report.add(PhaseResult("system-deps", "skip", 0, details="--skip-deps"))
    else:
        report.add(phase_system_deps(log))
        if report.overall == "fail":
            log.error("Phase 1 failed – cannot continue")
            print_summary(report)
            if args.json_output:
                print(json.dumps(asdict(report), indent=2, default=str))
            return 1

    print()

    # ── Phase 2 ──
    result2 = phase_install_cli(log)
    report.add(result2)
    binary_path = result2.details if result2.status == "pass" else ""
    report.opencode_binary = binary_path

    if result2.status == "fail":
        log.error("Phase 2 failed – cannot continue")
        print_summary(report)
        if args.json_output:
            print(json.dumps(asdict(report), indent=2, default=str))
        return 1

    print()

    # ── Phase 3 ──
    report.add(phase_verify_binary(log, binary_path))
    print()

    # ── Phase 4 ──
    report.add(phase_write_config(log, args.workspace))
    print()

    # ── Phase 5 ──
    if args.run_query:
        report.add(phase_sample_query(log, binary_path, args.model,
                                       args.workspace, args.prompt))
    else:
        log.info(_bold("Phase 5: Sample Query"))
        log.info("  Skipped (use --run-query to enable)")
        report.add(PhaseResult("sample-query", "skip", 0, details="use --run-query"))

    # ── Report ──
    print_summary(report)
    if args.json_output:
        print(json.dumps(asdict(report), indent=2, default=str))

    return 0 if report.overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

from types import SimpleNamespace
import pytest

import sys
import types

sys.modules.setdefault(
    "weave",
    types.SimpleNamespace(
        init=lambda x: None,
        finish=lambda: None,
        op=lambda *a, **k: (lambda func: func),
    ),
)

inspect_ai_mod = types.ModuleType("inspect_ai")
inspect_ai_mod.eval = lambda *a, **k: None
inspect_ai_mod.TaskInfo = object
inspect_ai_mod.solver = types.SimpleNamespace(Solver=object)
sys.modules.setdefault("inspect_ai", inspect_ai_mod)
sys.modules.setdefault(
    "inspect_ai.solver",
    types.SimpleNamespace(solver=lambda *a, **k: None, Solver=object),
)

sys.modules.setdefault(
    "inspect_ai.log",
    types.SimpleNamespace(EvalLog=dict, write_eval_log=lambda *a, **k: None),
)
sys.modules.setdefault("inspect_ai.model", types.SimpleNamespace(get_model=lambda *a, **k: None))

eval_mod = types.ModuleType("inspect_ai._eval")
eval_mod.loader = types.SimpleNamespace(load_tasks=lambda *a, **k: [])
eval_mod.loader_utils = types.SimpleNamespace(task_from_str=lambda *a, **k: None)
sys.modules.setdefault("inspect_ai._eval", eval_mod)
sys.modules.setdefault("inspect_ai._eval.loader", eval_mod.loader)

from hal.generalist_cli import main as hal_test

class DummyRunner:
    def __init__(self, *args, **kwargs):
        self.benchmark = SimpleNamespace(benchmark_name=kwargs.get("benchmark_name"), get_run_dir=lambda run_id: ".")
    async def run(self, agent_name: str, upload: bool = False):
        return {"accuracy": 1.0}


def test_hal_test_cli_invokes_runner(monkeypatch):
    dummy_mod = types.ModuleType("hal.agent_runner")
    dummy_mod.AgentRunner = DummyRunner
    monkeypatch.setitem(sys.modules, "hal.agent_runner", dummy_mod)
    monkeypatch.setattr("hal.utils.logging_utils.setup_logging", lambda *a, **k: None)
    result = hal_test.main(
        [
            "--benchmark",
            "gaia",
            "-A",
            "model_name=dummy",
            "-A",
            "budget=0.01",
            "--max_tasks",
            "1",
        ],
        standalone_mode=False,
    )
    assert result is None

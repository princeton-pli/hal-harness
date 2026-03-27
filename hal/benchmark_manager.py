import importlib
from typing import Any

# Map benchmark_name -> "module:ClassName"
# Add a benchmark here; no other file needs to change.
_BENCHMARK_REGISTRY: dict[str, str] = {
    "gaia":                          "hal.benchmarks.gaia:GaiaBenchmark",
    "scicode":                       "hal.benchmarks.scicode:SciCodeBenchmark",
    "scicode_easy":                  "hal.benchmarks.scicode:SciCodeBenchmark",
    "scicode_hard":                  "hal.benchmarks.scicode:SciCodeBenchmark",
    "usaco":                         "hal.benchmarks.usaco:USACOBenchmark",
    "mlagentbench":                  "hal.benchmarks.mlagentbench:MLAgentBenchBenchmark",
    "swebench_verified":             "hal.benchmarks.swebench:SWEBenchBenchmark",
    "swebench_verified_mini":        "hal.benchmarks.swebench:SWEBenchBenchmark",
    "appworld_test_normal":          "hal.benchmarks.appworld:AppWorldBenchmark",
    "appworld_test_challenge":       "hal.benchmarks.appworld:AppWorldBenchmark",
    "taubench_retail":               "hal.benchmarks.taubench:TauBenchBenchmark",
    "taubench_airline":              "hal.benchmarks.taubench:TauBenchBenchmark",
    "corebench_easy":                "hal.benchmarks.corebench:CoreBenchEasy",
    "corebench_medium":              "hal.benchmarks.corebench:CoreBenchMedium",
    "corebench_hard":                "hal.benchmarks.corebench:CoreBenchHard",
    "scienceagentbench":             "hal.benchmarks.scienceagentbench:ScienceAgentBench",
    "assistantbench":                "hal.benchmarks.assistantbench:AssistantBenchBenchmark",
    "colbench_backend_programming":  "hal.benchmarks.colbench:ColBenchBenchmark",
    "colbench_frontend_design":      "hal.benchmarks.colbench:ColBenchBenchmark",
}


def _load_class(import_path: str) -> type:
    """Lazy import — only loads the module when first requested."""
    module_path, class_name = import_path.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


class BenchmarkManager:
    @property
    def benchmarks(self) -> list[str]:
        return sorted(_BENCHMARK_REGISTRY)

    def list_benchmarks(self) -> list[str]:
        return self.benchmarks

    def get_benchmark(
        self, benchmark_name: str, agent_dir: str, config: dict[str, Any]
    ) -> "BaseBenchmark":  # noqa: F821
        if benchmark_name not in _BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown benchmark: {benchmark_name!r}. "
                f"Available: {self.benchmarks}"
            )
        cls = _load_class(_BENCHMARK_REGISTRY[benchmark_name])
        return cls(benchmark_name, agent_dir, config)

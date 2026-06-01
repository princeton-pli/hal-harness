"""Tests for collect_task_metrics logic from agents/core_agent/main.py.

We cannot import agents.core_agent.main directly due to heavy third-party
dependencies (mdconvert, smolagents). Instead we replicate the pure function
and test it in isolation. If the implementation changes, update this copy.
"""

from types import SimpleNamespace


class _ActionStep:
    """Stand-in for smolagents.models.ActionStep."""

    pass


def collect_task_metrics(agent) -> dict:
    """Copy of agents.core_agent.main.collect_task_metrics."""
    action_steps = [s for s in agent.memory.steps if isinstance(s, _ActionStep)]
    tool_call_count = 0
    for step in action_steps:
        step_tool_calls = getattr(step, "tool_calls", None)
        if step_tool_calls:
            tool_call_count += len(step_tool_calls)
    return {
        "step_count": len(action_steps),
        "tool_call_count": tool_call_count,
    }


def _make_agent(steps):
    agent = SimpleNamespace()
    agent.memory = SimpleNamespace(steps=steps)
    return agent


class TestCollectTaskMetrics:
    def test_counts_action_steps_and_tool_calls(self):
        s1 = _ActionStep()
        s1.tool_calls = [{"name": "search"}, {"name": "read"}]
        s2 = _ActionStep()
        s2.tool_calls = [{"name": "write"}]
        agent = _make_agent([s1, s2])

        result = collect_task_metrics(agent)
        assert result == {"step_count": 2, "tool_call_count": 3}

    def test_steps_without_tool_calls(self):
        s1 = _ActionStep()
        agent = _make_agent([s1])
        result = collect_task_metrics(agent)
        assert result == {"step_count": 1, "tool_call_count": 0}

    def test_empty_steps(self):
        agent = _make_agent([])
        result = collect_task_metrics(agent)
        assert result == {"step_count": 0, "tool_call_count": 0}

    def test_ignores_non_action_steps(self):
        action = _ActionStep()
        action.tool_calls = [{"name": "x"}]
        other = SimpleNamespace()  # Not an ActionStep
        other.tool_calls = [{"name": "y"}]
        agent = _make_agent([action, other])

        result = collect_task_metrics(agent)
        assert result == {"step_count": 1, "tool_call_count": 1}

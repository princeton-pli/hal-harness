"""Test reasoning_effort kwarg construction for the example agent.

The agent builds request_kwargs and conditionally adds a `reasoning` key.
We replicate that logic here to test it without importing tau_bench.
"""


def _build_request_kwargs(model_name: str, **kwargs) -> dict:
    """Mirrors the request_kwargs construction in main.run()."""
    request_kwargs = {
        "model": model_name,
        "input": [{"role": "user", "content": "test"}],
        "max_output_tokens": 2000,
        "temperature": 1,
    }
    if "reasoning_effort" in kwargs:
        request_kwargs["reasoning"] = {"effort": kwargs["reasoning_effort"]}
    return request_kwargs


def test_reasoning_effort_passed_when_present():
    kwargs = _build_request_kwargs("o3", reasoning_effort="medium")
    assert kwargs["reasoning"] == {"effort": "medium"}


def test_reasoning_effort_not_passed_when_absent():
    kwargs = _build_request_kwargs("o3")
    assert "reasoning" not in kwargs

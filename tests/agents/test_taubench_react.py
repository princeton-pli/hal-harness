from agents.taubench_react.react import _inject_reasoning_kwargs


def test_injects_reasoning_for_matching_model():
    kwargs = {"model": "responses/o3"}
    _inject_reasoning_kwargs(
        kwargs, model_name="responses/o3", reasoning_effort="medium"
    )
    assert kwargs["reasoning"] == {"effort": "medium"}
    assert kwargs["temperature"] == 1.0


def test_no_injection_for_non_matching_model():
    kwargs = {"model": "gpt-4o"}
    _inject_reasoning_kwargs(
        kwargs, model_name="responses/o3", reasoning_effort="medium"
    )
    assert "reasoning" not in kwargs
    assert "temperature" not in kwargs


def test_removes_reasoning_effort_key():
    kwargs = {"model": "responses/o3", "reasoning_effort": "high"}
    _inject_reasoning_kwargs(kwargs, model_name="responses/o3", reasoning_effort="high")
    assert "reasoning_effort" not in kwargs
    assert kwargs["reasoning"] == {"effort": "high"}

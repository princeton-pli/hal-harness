from unittest.mock import MagicMock, patch


def _make_env_input(task_id="task_0"):
    return {
        task_id: {
            "env": "retail",
            "user_strategy": "default",
            "user_model": "gpt-4o",
            "task_split": "test",
            "user_provider": "openai",
            "task_index": 0,
        }
    }


def _stub_env():
    env = MagicMock()
    env.reset.return_value.observation = "Do something"
    env.reward = 1.0
    env.actions = []
    env.task.model_dump.return_value = {}
    env.step.return_value = None
    return env


@patch("agents.taubench_example_agent.main.get_env")
@patch("agents.taubench_example_agent.main.OpenAI")
def test_reasoning_effort_passed_when_present(mock_openai_cls, mock_get_env):
    mock_get_env.return_value = _stub_env()
    mock_client = MagicMock()
    mock_client.responses.create.return_value = MagicMock(output_text="hello")
    mock_openai_cls.return_value = mock_client

    from agents.taubench_example_agent.main import run

    run(_make_env_input(), model_name="o3", reasoning_effort="medium")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "medium"}


@patch("agents.taubench_example_agent.main.get_env")
@patch("agents.taubench_example_agent.main.OpenAI")
def test_reasoning_effort_not_passed_when_absent(mock_openai_cls, mock_get_env):
    mock_get_env.return_value = _stub_env()
    mock_client = MagicMock()
    mock_client.responses.create.return_value = MagicMock(output_text="hello")
    mock_openai_cls.return_value = mock_client

    from agents.taubench_example_agent.main import run

    run(_make_env_input(), model_name="o3")

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert "reasoning" not in call_kwargs

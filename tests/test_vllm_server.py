from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.vllm_server import (
    build_vllm_server_command,
    build_vllm_server_commands,
    check_openai_endpoint,
    normalize_openai_base_url,
    start_vllm_server,
    start_vllm_servers,
    wait_for_vllm_server,
)


def _config() -> MMAIConfig:
    return MMAIConfig(
        preset_name="test",
        debug_mode=False,
        trial={
            "model_name": "trial-model",
            "reasoning_parser": "gemma4",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        patient={
            "model_name": "patient-model",
            "reasoning_parser": "gemma4",
            "chat_template_kwargs": {"enable_thinking": True},
        },
        local={
            "trial": {
                "max_model_len": 10000,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.8,
            },
            "patient": {
                "max_model_len": 120000,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9,
            },
        },
        remote={
            "enabled": True,
            "server_urls": [
                "http://localhost:8000/v1",
                "http://127.0.0.1:8001/v1",
            ],
        },
        embedding={},
        model_metadata_cache_dir=None,
        raw={},
    )


def test_build_vllm_server_command_uses_task_model_and_local_runtime():
    """Build a patient server command from task model and local runtime config."""
    command = build_vllm_server_command(config=_config(), task="patient")

    assert command.model_name == "patient-model"
    assert command.base_url == "http://localhost:8000/v1"
    assert command.command == [
        "vllm",
        "serve",
        "patient-model",
        "--served-model-name",
        "patient-model",
        "--host",
        "localhost",
        "--port",
        "8000",
        "--max-model-len",
        "120000",
        "--tensor-parallel-size",
        "2",
        "--gpu-memory-utilization",
        "0.9",
        "--reasoning-parser",
        "gemma4",
        "--default-chat-template-kwargs",
        '{"enable_thinking": true}',
    ]


def test_build_vllm_server_command_selects_server_url_and_allows_extra_args():
    """Select the requested server URL and append caller-provided vLLM args."""
    command = build_vllm_server_command(
        config=_config(),
        task="trial",
        server_index=1,
        extra_args=["--dtype", "bfloat16"],
    )

    assert command.host == "127.0.0.1"
    assert command.port == 8001
    assert command.command[-2:] == ["--dtype", "bfloat16"]
    assert "--max-model-len" in command.command
    assert command.command[command.command.index("--max-model-len") + 1] == "10000"


def test_build_vllm_server_command_uses_served_model_alias():
    """Allow vLLM to load one model ID and expose a separate endpoint name."""
    config = _config()
    config.remote["served_model_name"] = "endpoint-model"

    command = build_vllm_server_command(config=config, task="patient")

    assert command.model_name == "patient-model"
    assert command.served_model_name == "endpoint-model"
    assert command.command[command.command.index("--served-model-name") + 1] == (
        "endpoint-model"
    )


def test_normalize_openai_base_url_adds_scheme_and_v1():
    """Normalize user-entered OpenAI-compatible base URLs."""
    assert normalize_openai_base_url("localhost:8000") == "http://localhost:8000/v1"
    assert normalize_openai_base_url("http://host:9000/v1") == "http://host:9000/v1"


def test_start_vllm_server_invokes_subprocess(monkeypatch, capsys):
    """Start one server by invoking subprocess with the resolved command."""
    calls = []

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.subprocess.Popen", fake_popen)

    process = start_vllm_server(
        config=_config(),
        task="patient",
        stdout=-1,
        stderr=-1,
        wait_until_ready=False,
    )

    assert isinstance(process, FakeProcess)
    assert "vLLM server URL: http://localhost:8000/v1" in capsys.readouterr().out
    assert calls[0][0][:3] == ["vllm", "serve", "patient-model"]
    assert calls[0][1]["text"] is True
    assert calls[0][1]["stdout"] == -1
    assert calls[0][1]["stderr"] == -1


def test_start_vllm_server_can_suppress_url_print(monkeypatch, capsys):
    """Allow callers to suppress the printed remote server URL."""

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        return FakeProcess()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.subprocess.Popen", fake_popen)

    start_vllm_server(
        config=_config(),
        task="patient",
        print_url=False,
        wait_until_ready=False,
    )

    assert capsys.readouterr().out == ""


def test_plural_helpers_use_all_configured_server_urls(monkeypatch):
    """Build and start one vLLM server for each configured remote URL."""
    commands = build_vllm_server_commands(config=_config(), task="trial")

    assert [command.port for command in commands] == [8000, 8001]

    calls = []

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.subprocess.Popen", fake_popen)

    processes = start_vllm_servers(
        config=_config(),
        task="trial",
        wait_until_ready=False,
    )

    assert len(processes) == 2
    assert [call[0][call[0].index("--port") + 1] for call in calls] == [
        "8000",
        "8001",
    ]


def test_wait_for_vllm_server_polls_models_endpoint(monkeypatch):
    """Poll the OpenAI-compatible models endpoint until the server is ready."""
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.headers, timeout))
        return FakeResponse()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.request.urlopen", fake_urlopen)

    wait_for_vllm_server(
        "http://localhost:8000/v1",
        timeout=1,
        poll_interval=0.1,
        api_key="test-key",
    )

    assert calls == [
        (
            "http://localhost:8000/v1/models",
            {"Authorization": "Bearer test-key"},
            0.1,
        )
    ]


def test_check_openai_endpoint_returns_statuses(monkeypatch):
    """Check one or more OpenAI-compatible /models endpoints."""
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.headers, timeout))
        return FakeResponse()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.request.urlopen", fake_urlopen)

    result = check_openai_endpoint(
        "localhost:8000, http://127.0.0.1:8001/v1",
        api_key="key",
        timeout=3,
    )

    assert result == [
        ("http://localhost:8000/v1", 200, "HTTP 200"),
        ("http://127.0.0.1:8001/v1", 200, "HTTP 200"),
    ]
    assert calls[0] == (
        "http://localhost:8000/v1/models",
        {"Authorization": "Bearer key"},
        3,
    )

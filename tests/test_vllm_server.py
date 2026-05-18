from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.vllm_server import (
    build_vllm_server_command,
    build_vllm_server_commands,
    start_vllm_server,
    start_vllm_servers,
)


def _config() -> MMAIConfig:
    return MMAIConfig(
        preset_name="test",
        debug_mode=False,
        trial={"model_name": "trial-model"},
        patient={"model_name": "patient-model"},
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
    ]


def test_build_vllm_server_command_selects_server_url_and_allows_extra_args():
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


def test_start_vllm_server_invokes_subprocess(monkeypatch, capsys):
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
    )

    assert isinstance(process, FakeProcess)
    assert "vLLM server URL: http://localhost:8000/v1" in capsys.readouterr().out
    assert calls[0][0][:3] == ["vllm", "serve", "patient-model"]
    assert calls[0][1]["text"] is True
    assert calls[0][1]["stdout"] == -1
    assert calls[0][1]["stderr"] == -1


def test_start_vllm_server_can_suppress_url_print(monkeypatch, capsys):
    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        return FakeProcess()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.subprocess.Popen", fake_popen)

    start_vllm_server(
        config=_config(),
        task="patient",
        print_url=False,
    )

    assert capsys.readouterr().out == ""


def test_plural_helpers_use_all_configured_server_urls(monkeypatch):
    commands = build_vllm_server_commands(config=_config(), task="trial")

    assert [command.port for command in commands] == [8000, 8001]

    calls = []

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.subprocess.Popen", fake_popen)

    processes = start_vllm_servers(config=_config(), task="trial")

    assert len(processes) == 2
    assert [call[0][call[0].index("--port") + 1] for call in calls] == [
        "8000",
        "8001",
    ]

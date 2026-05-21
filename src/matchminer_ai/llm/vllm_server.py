"""Helpers for starting OpenAI-compatible local vLLM servers."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Sequence
from urllib import error, request
from urllib.parse import urlparse

from matchminer_ai.config import MMAIConfig, load_default_preset


@dataclass(frozen=True)
class VLLMServerCommand:
    """Resolved command and connection details for one vLLM server."""

    command: list[str]
    base_url: str
    host: str
    port: int
    model_name: str
    task: str


def _resolve_task_config(
    config: MMAIConfig, task: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if task not in {"trial", "patient"}:
        raise ValueError("task must be 'trial' or 'patient'.")

    llm_config = dict(getattr(config, task))
    local_config = dict(getattr(config, "local", {}).get(task, {}))
    missing = [
        key
        for key in (
            "model_name",
            "max_model_len",
            "tensor_parallel_size",
            "gpu_memory_utilization",
        )
        if key not in {**llm_config, **local_config}
    ]
    if missing:
        raise ValueError(
            f"Config for {task!r} is missing required vLLM server keys: "
            f"{', '.join(missing)}"
        )
    merged_config = {**llm_config, **local_config}
    return merged_config, local_config


def _remote_server_urls(config: MMAIConfig) -> list[str]:
    remote_config = dict(getattr(config, "remote", {}))
    raw_urls = remote_config.get("server_urls", ["http://localhost:8000/v1"])
    if isinstance(raw_urls, str):
        urls = [url.strip() for url in raw_urls.split(",")]
    else:
        urls = [str(url).strip() for url in raw_urls]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("config.remote.server_urls must contain at least one URL.")
    return urls


def _remote_server_url(config: MMAIConfig, server_index: int) -> str:
    urls = _remote_server_urls(config)
    if server_index < 0 or server_index >= len(urls):
        raise ValueError(
            f"server_index {server_index} is out of range for "
            f"{len(urls)} server URL(s)."
        )
    return urls[server_index]


def _host_and_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return host, port


def build_vllm_server_command(
    *,
    config: MMAIConfig | None = None,
    task: str = "patient",
    server_index: int = 0,
    extra_args: Sequence[str] | None = None,
) -> VLLMServerCommand:
    """
    Build a ``vllm serve`` command from MatchMiner-AI configuration.

    The served model name is set to the configured model name so remote requests
    using ``trial.model_name`` or ``patient.model_name`` match the running server.
    The helper maps only the known local engine fields needed by the default
    config. Additional server-only CLI flags, such as speculative decoding
    options, should be supplied explicitly with ``extra_args``.
    """
    resolved_config = config or load_default_preset()
    llm_config, local_config = _resolve_task_config(resolved_config, task)
    base_url = _remote_server_url(resolved_config, server_index)
    host, port = _host_and_port(base_url)
    model_name = str(llm_config["model_name"])

    command = [
        "vllm",
        "serve",
        model_name,
        "--served-model-name",
        model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(local_config["max_model_len"]),
        "--tensor-parallel-size",
        str(local_config["tensor_parallel_size"]),
        "--gpu-memory-utilization",
        str(local_config["gpu_memory_utilization"]),
    ]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)

    return VLLMServerCommand(
        command=command,
        base_url=base_url,
        host=host,
        port=port,
        model_name=model_name,
        task=task,
    )


def build_vllm_server_commands(
    *,
    config: MMAIConfig | None = None,
    task: str = "patient",
    extra_args: Sequence[str] | None = None,
) -> list[VLLMServerCommand]:
    """Build one ``vllm serve`` command for each configured remote server URL."""
    resolved_config = config or load_default_preset()
    return [
        build_vllm_server_command(
            config=resolved_config,
            task=task,
            server_index=server_index,
            extra_args=extra_args,
        )
        for server_index, _url in enumerate(_remote_server_urls(resolved_config))
    ]


def wait_for_vllm_server(
    base_url: str,
    *,
    process: subprocess.Popen[str] | None = None,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
    api_key: str | None = None,
) -> None:
    """
    Wait until a vLLM OpenAI-compatible server responds to ``/models``.

    ``base_url`` should be the OpenAI-compatible base URL, usually ending in
    ``/v1``. Raises ``TimeoutError`` if the server does not become ready before
    ``timeout`` seconds.
    """
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.monotonic() + timeout
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "vLLM server process exited before becoming ready "
                f"with code {process.returncode}."
            )

        req = request.Request(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=min(poll_interval, 10.0)) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, error.URLError, error.HTTPError) as exc:
            last_error = exc

        time.sleep(poll_interval)

    message = f"Timed out waiting for vLLM server at {models_url}."
    if last_error is not None:
        message += f" Last error: {last_error}"
    raise TimeoutError(message)


def start_vllm_server(
    *,
    config: MMAIConfig | None = None,
    task: str = "patient",
    server_index: int = 0,
    extra_args: Sequence[str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    print_url: bool = True,
    wait_until_ready: bool = True,
    ready_timeout: float = 600.0,
    ready_poll_interval: float = 5.0,
) -> subprocess.Popen[str]:
    """
    Start one local OpenAI-compatible vLLM server from config.

    Returns the ``subprocess.Popen`` handle so callers can monitor or terminate
    the server process. Additional vLLM server CLI flags are not read from the
    config; pass them explicitly with ``extra_args``. By default, prints the
    remote base URL that should be used in ``config.remote["server_urls"]`` and
    waits until the server responds to ``/v1/models``.
    """
    command = build_vllm_server_command(
        config=config,
        task=task,
        server_index=server_index,
        extra_args=extra_args,
    )
    if print_url:
        print(f"vLLM server URL: {command.base_url}")
    process = subprocess.Popen(
        command.command,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )
    if wait_until_ready:
        wait_for_vllm_server(
            command.base_url,
            process=process,
            timeout=ready_timeout,
            poll_interval=ready_poll_interval,
        )
    return process


def start_vllm_servers(
    *,
    config: MMAIConfig | None = None,
    task: str = "patient",
    extra_args: Sequence[str] | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    print_url: bool = True,
    wait_until_ready: bool = True,
    ready_timeout: float = 600.0,
    ready_poll_interval: float = 5.0,
) -> list[subprocess.Popen[str]]:
    """Start one local vLLM server for each URL in ``config.remote.server_urls``."""
    resolved_config = config or load_default_preset()
    return [
        start_vllm_server(
            config=resolved_config,
            task=task,
            server_index=server_index,
            extra_args=extra_args,
            stdout=stdout,
            stderr=stderr,
            print_url=print_url,
            wait_until_ready=wait_until_ready,
            ready_timeout=ready_timeout,
            ready_poll_interval=ready_poll_interval,
        )
        for server_index, _url in enumerate(_remote_server_urls(resolved_config))
    ]


__all__ = [
    "VLLMServerCommand",
    "build_vllm_server_command",
    "build_vllm_server_commands",
    "start_vllm_server",
    "start_vllm_servers",
    "wait_for_vllm_server",
]

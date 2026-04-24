import asyncio
import sys
from types import SimpleNamespace
from typing import Any, ClassVar, Callable, Awaitable

from mmai.llm.backends import RemoteBackend
from mmai.llm.prompt_rendering import Prompt


class FakeCompletions:
    """Minimal async completions namespace matching the OpenAI client shape."""

    def __init__(self, client):
        self.client = client

    async def create(self, **kwargs):
        return await self.client.create(**kwargs)


class FakeAsyncOpenAI:
    """Fake AsyncOpenAI client that records calls and returns configurable output."""

    clients: ClassVar[list["FakeAsyncOpenAI"]] = []
    behavior: ClassVar[
        Callable[["FakeAsyncOpenAI", dict[str, Any]], Awaitable[Any]] | None
    ] = None

    def __init__(self, *, base_url, api_key, timeout):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.completions = FakeCompletions(self)
        self.calls: list[dict[str, Any]] = []
        FakeAsyncOpenAI.clients.append(self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if FakeAsyncOpenAI.behavior is not None:
            return await FakeAsyncOpenAI.behavior(self, kwargs)
        return _response(f"{self.base_url}:{kwargs['prompt']}")


def _response(text, finish_reason="stop"):
    """Build a minimal OpenAI-style completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text, finish_reason=finish_reason)]
    )


def _prompts(*contents):
    """Build indexed Prompt objects for backend tests."""
    return [
        Prompt(row_idx=row_idx, prompt_text=content, max_tokens=10)
        for row_idx, content in enumerate(contents)
    ]


def _llm_config(**overrides):
    """Return a complete remote LLM config with optional overrides."""
    config = {
        "model_name": "model",
        "server_urls": ["http://server-a/v1"],
        "request_timeout": 123,
        "max_concurrent_requests": 1,
        "max_retries": 3,
        "batch_size": 1000,
        "retry_backoff_base": 0.0,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 10,
            "repetition_penalty": 1.0,
        },
    }
    config.update(overrides)
    return config


def _install_fakes(monkeypatch):
    """Install fake OpenAI and metadata dependencies for remote backend tests."""
    FakeAsyncOpenAI.clients = []
    FakeAsyncOpenAI.behavior = None
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    monkeypatch.setattr(
        "mmai.llm.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_name": model_name,
            "model_sha": "sha",
        },
    )


def test_remote_backend_single_server_preserves_order(monkeypatch):
    """A single remote server returns outputs aligned to input prompt order."""
    _install_fakes(monkeypatch)

    texts, metadata, finish_reasons = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2"),
        llm_config=_llm_config(),
    )

    assert texts == [
        "http://server-a/v1:p0",
        "http://server-a/v1:p1",
        "http://server-a/v1:p2",
    ]
    assert metadata["model_sha"] == "sha"
    assert finish_reasons == ["stop", "stop", "stop"]
    assert FakeAsyncOpenAI.clients[0].timeout == 183


def test_remote_backend_multiple_servers_distributes_and_preserves_order(monkeypatch):
    """Multiple remote servers receive round-robin prompts while output order holds."""
    _install_fakes(monkeypatch)

    texts, _, _ = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2", "p3"),
        llm_config=_llm_config(
            server_urls=["http://server-a/v1", "http://server-b/v1"]
        ),
    )

    assert texts == [
        "http://server-a/v1:p0",
        "http://server-b/v1:p1",
        "http://server-a/v1:p2",
        "http://server-b/v1:p3",
    ]
    calls_by_server = {
        client.base_url: [call["prompt"] for call in client.calls]
        for client in FakeAsyncOpenAI.clients
    }
    assert calls_by_server["http://server-a/v1"] == ["p0", "p2"]
    assert calls_by_server["http://server-b/v1"] == ["p1", "p3"]


def test_remote_backend_uses_env_var_api_key(monkeypatch):
    """Remote backend should source API keys from the environment only."""
    _install_fakes(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(api_key="config-key"),
    )

    assert FakeAsyncOpenAI.clients[0].api_key == "env-key"


def test_remote_backend_allows_multiple_in_flight_and_respects_limit(monkeypatch):
    """Remote backend allows concurrent requests but respects configured limits."""
    _install_fakes(monkeypatch)
    active = 0
    max_active = 0

    async def behavior(client, kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _response(kwargs["prompt"])

    FakeAsyncOpenAI.behavior = behavior

    texts, _, _ = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2", "p3", "p4"),
        llm_config=_llm_config(max_concurrent_requests=2),
    )

    assert texts == ["p0", "p1", "p2", "p3", "p4"]
    assert max_active == 2


def test_remote_backend_retries_transient_failures(monkeypatch):
    """Transient completion failures are retried before returning successful output."""
    _install_fakes(monkeypatch)
    attempts = 0

    async def behavior(client, kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return _response("recovered", finish_reason="length")

    FakeAsyncOpenAI.behavior = behavior

    texts, _, finish_reasons = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(max_retries=2),
    )

    assert attempts == 2
    assert texts == ["recovered"]
    assert finish_reasons == ["length"]

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, ClassVar

from matchminer_ai.llm.backends import RemoteBackend
from matchminer_ai.llm.prompt_rendering import Prompt


class FakeChatCompletions:
    """Minimal async chat completions namespace matching the OpenAI client shape."""

    def __init__(self, client):
        self.client = client

    async def create(self, **kwargs):
        return await self.client.create(**kwargs)


class FakeChat:
    """Minimal async chat namespace."""

    def __init__(self, client):
        self.completions = FakeChatCompletions(client)


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
        self.chat = FakeChat(self)
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        FakeAsyncOpenAI.clients.append(self)

    async def aclose(self):
        self.closed = True

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if FakeAsyncOpenAI.behavior is not None:
            return await FakeAsyncOpenAI.behavior(self, kwargs)
        return _response(f"{self.base_url}:{kwargs['messages'][-1]['content']}")


def _response(text, finish_reason="stop", reasoning="", reasoning_content=None):
    """Build a minimal OpenAI-style chat completion response."""
    message_kwargs = {"content": text, "reasoning": reasoning}
    if reasoning_content is not None:
        message_kwargs["reasoning_content"] = reasoning_content
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(**message_kwargs),
                finish_reason=finish_reason,
            )
        ]
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
        "matchminer_ai.llm.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_name": model_name,
            "model_sha": "sha",
        },
    )


def test_remote_backend_single_server_preserves_order(monkeypatch):
    """A single remote server returns outputs aligned to input prompt order."""
    _install_fakes(monkeypatch)

    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2"),
        llm_config=_llm_config(),
    )

    assert result.final_outputs == [
        "http://server-a/v1:p0",
        "http://server-a/v1:p1",
        "http://server-a/v1:p2",
    ]
    assert result.model_metadata["model_sha"] == "sha"
    assert result.finish_reasons == ["stop", "stop", "stop"]
    assert FakeAsyncOpenAI.clients[0].timeout == 183
    assert FakeAsyncOpenAI.clients[0].closed is True


def test_remote_backend_multiple_servers_distributes_and_preserves_order(monkeypatch):
    """Multiple remote servers receive round-robin prompts while output order holds."""
    _install_fakes(monkeypatch)

    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2", "p3"),
        llm_config=_llm_config(
            server_urls=["http://server-a/v1", "http://server-b/v1"]
        ),
    )

    assert result.final_outputs == [
        "http://server-a/v1:p0",
        "http://server-b/v1:p1",
        "http://server-a/v1:p2",
        "http://server-b/v1:p3",
    ]
    calls_by_server = {
        client.base_url: [call["messages"][-1]["content"] for call in client.calls]
        for client in FakeAsyncOpenAI.clients
    }
    assert calls_by_server["http://server-a/v1"] == ["p0", "p2"]
    assert calls_by_server["http://server-b/v1"] == ["p1", "p3"]
    assert all(client.closed for client in FakeAsyncOpenAI.clients)


def test_remote_backend_captures_structured_reasoning(monkeypatch):
    """Remote chat responses expose final content separately from reasoning."""
    _install_fakes(monkeypatch)

    async def behavior(client, kwargs):
        return _response("final answer", reasoning="thinking text")

    FakeAsyncOpenAI.behavior = behavior
    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(),
    )

    assert result.final_outputs == ["final answer"]
    assert result.reasoning_outputs == ["thinking text"]
    assert result.raw_outputs == []
    assert result.finish_reasons == ["stop"]


def test_remote_backend_passes_chat_template_kwargs(monkeypatch):
    """Remote chat requests pass Gemma 4 thinking kwargs through vLLM extra_body."""
    _install_fakes(monkeypatch)

    RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(
            chat_template_kwargs={"enable_thinking": True},
        ),
    )

    assert FakeAsyncOpenAI.clients[0].calls[0]["extra_body"][
        "chat_template_kwargs"
    ] == {"enable_thinking": True}


def test_remote_backend_forwards_sampling_params(monkeypatch):
    """Remote requests forward config sampling params to request args/extra_body."""
    _install_fakes(monkeypatch)

    RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.95,
                "presence_penalty": 1.5,
                "max_tokens": 99,
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.1,
                "skip_special_tokens": False,
            },
        ),
    )

    call = FakeAsyncOpenAI.clients[0].calls[0]
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.95
    assert call["presence_penalty"] == 1.5
    assert call["max_tokens"] == 10
    assert call["extra_body"] == {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "skip_special_tokens": False,
    }


def test_remote_backend_accepts_legacy_reasoning_content(monkeypatch):
    """Older vLLM servers used message.reasoning_content."""
    _install_fakes(monkeypatch)

    async def behavior(client, kwargs):
        return _response("final answer", reasoning="", reasoning_content="old thinking")

    FakeAsyncOpenAI.behavior = behavior
    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(),
    )

    assert result.final_outputs == ["final answer"]
    assert result.reasoning_outputs == ["old thinking"]


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
        return _response(kwargs["messages"][-1]["content"])

    FakeAsyncOpenAI.behavior = behavior

    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0", "p1", "p2", "p3", "p4"),
        llm_config=_llm_config(max_concurrent_requests=2),
    )

    assert result.final_outputs == ["p0", "p1", "p2", "p3", "p4"]
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

    result = RemoteBackend().generate_llm_outputs(
        prompt_list=_prompts("p0"),
        llm_config=_llm_config(max_retries=2),
    )

    assert attempts == 2
    assert result.final_outputs == ["recovered"]
    assert result.finish_reasons == ["length"]

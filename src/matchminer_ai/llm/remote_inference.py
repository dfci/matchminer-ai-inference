"""Remote OpenAI-compatible inference orchestration."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, TypeVar, cast
from urllib.parse import urlparse

from matchminer_ai.llm.prompt_rendering import Prompt


logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class ModelResult:
    """Remote inference result tied to the original prompt row index."""

    row_idx: int
    reasoning: str
    summary: str
    finish_reason: str = "stop"


def normalize_remote_server_urls(llm_config: Dict[str, Any]) -> list[str]:
    """Return configured remote server URLs after validation and normalization."""
    if "server_urls" not in llm_config:
        raise ValueError("Remote backend requires llm_config['server_urls'].")
    raw_server_urls = llm_config["server_urls"]

    if isinstance(raw_server_urls, str):
        server_urls = [url.strip() for url in raw_server_urls.split(",")]
    else:
        server_urls = [str(url).strip() for url in raw_server_urls]

    server_urls = [normalize_openai_base_url(url) for url in server_urls if url]
    if not server_urls:
        raise ValueError("Remote backend requires at least one server URL.")
    return server_urls


def normalize_openai_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL, adding scheme and /v1 if needed."""
    value = (base_url or "").strip() or "http://localhost:8000/v1"
    if "://" not in value:
        value = f"http://{value}"
    value = value.rstrip("/")
    parsed = urlparse(value)
    if parsed.path in {"", "/"}:
        value = f"{value}/v1"
    return value


def remote_request_model_name(llm_config: Dict[str, Any]) -> str:
    """Return the model name to send to the OpenAI-compatible endpoint."""
    return str(llm_config.get("served_model_name") or llm_config["model_name"])


def _run_sync(awaitable_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run an async task from sync code, including notebook event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable_factory())

    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[T] = executor.submit(lambda: asyncio.run(awaitable_factory()))
        return future.result()


def connect_to_remote_servers(
    server_urls: list[str],
    request_timeout: float = 600.0,
    api_key: str = "not-needed",
) -> list[tuple[Any, int]]:
    """
    Create AsyncOpenAI clients for externally managed vLLM servers.

    Parameters
    ----------
    server_urls
        OpenAI-compatible base URLs, usually ending in ``/v1``.
    request_timeout
        Per-request timeout in seconds. The OpenAI client receives this value
        plus a small transport buffer.
    api_key
        API key passed to the OpenAI client.

    Returns
    -------
    list[tuple[Any, int]]
        ``(client, port)`` tuples. The port is used only for logging.
    """
    from openai import AsyncOpenAI

    server_urls = [u.strip() for u in server_urls if u.strip()]
    logger.info("Using %d external server(s): %s", len(server_urls), server_urls)
    server_clients: list[tuple[Any, int]] = []
    for url in server_urls:
        client = AsyncOpenAI(
            base_url=url,
            api_key=api_key or "not-needed",
            timeout=request_timeout + 60,
        )
        server_clients.append((client, urlparse(url).port or 0))

    logger.info("All %d external server(s) connected.", len(server_urls))
    return server_clients


async def single_inference_request(
    client: Any,
    row_idx: int,
    messages: list[dict[str, str]],
    model: str,
    request_params: dict[str, Any],
    extra_body_params: dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
    max_retries: int = 6,
    base_timeout: float = 600.0,
    retry_backoff_base: float = 1.0,
) -> ModelResult:
    """
    Send one chat-completion request and retry transient failures.

    The returned ``ModelResult.row_idx`` is copied from the input so callers can
    restore original ordering after concurrent execution. vLLM reasoning
    parsers expose the final text as ``message.content`` and the reasoning
    trace as ``message.reasoning`` (``reasoning_content`` on older servers).
    """
    for attempt in range(max_retries):
        try:
            extra = dict(extra_body_params)
            if chat_template_kwargs:
                extra["chat_template_kwargs"] = dict(chat_template_kwargs)
            request_kwargs = {
                "model": model,
                "messages": messages,
                **request_params,
            }
            if extra:
                request_kwargs["extra_body"] = extra
            response = await asyncio.wait_for(
                client.chat.completions.create(**request_kwargs),
                timeout=base_timeout,
            )
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            content = cast(str, getattr(message, "content", "") or "")
            reasoning = cast(
                str,
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
                or "",
            )
            finish_reason = cast(str, getattr(choice, "finish_reason", None) or "stop")
            return ModelResult(
                row_idx=row_idx,
                reasoning=reasoning,
                summary=content,
                finish_reason=finish_reason,
            )

        except asyncio.TimeoutError:
            wait_time = min(retry_backoff_base * (2**attempt), 30.0)
            if attempt < max_retries - 1:
                logger.info(
                    "  Row %d: timeout (attempt %d/%d), retrying in %.1fs.",
                    row_idx,
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.info("  Row %d: all retries exhausted (timeout)", row_idx)
                raise

        except Exception as exc:
            wait_time = min(retry_backoff_base * (2**attempt), 30.0)
            if attempt < max_retries - 1:
                logger.info(
                    "  Row %d: error '%s' (attempt %d/%d), retrying in %.1fs.",
                    row_idx,
                    exc,
                    attempt + 1,
                    max_retries,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.info("  Row %d: all retries exhausted", row_idx)
                raise

    raise RuntimeError(f"Unexpected retry loop exit for row {row_idx}.")


async def run_inference_batch(
    client: Any,
    prompts: list[Prompt],
    model: str,
    request_params: dict[str, Any],
    extra_body_params: dict[str, Any],
    chat_template_kwargs: dict[str, Any] | None = None,
    max_concurrent: int = 16,
    batch_size: int = 1000,
    max_retries: int = 6,
    base_timeout: float = 600.0,
    port: int = 0,
    retry_backoff_base: float = 1.0,
) -> list[ModelResult]:
    """
    Send prompts to one remote server with bounded concurrency.

    Prompts are processed in chunks of ``batch_size``. Within each chunk, at
    most ``max_concurrent`` requests are in flight for this server.
    """
    total = len(prompts)
    all_results: list[ModelResult] = []

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = prompts[batch_start:batch_end]
        logger.info(
            "  Processing remote server %s batch %d-%d of %d.",
            port or "unknown",
            batch_start + 1,
            batch_end,
            total,
        )

        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        async def bounded_request(prompt: Prompt) -> ModelResult:
            async with semaphore:
                messages = prompt.messages or [
                    {"role": "user", "content": prompt.prompt_text}
                ]
                return await single_inference_request(
                    client=client,
                    row_idx=prompt.row_idx,
                    messages=messages,
                    model=model,
                    request_params={
                        **request_params,
                        "max_tokens": prompt.max_tokens,
                    },
                    extra_body_params=extra_body_params,
                    chat_template_kwargs=chat_template_kwargs,
                    max_retries=max_retries,
                    base_timeout=base_timeout,
                    retry_backoff_base=retry_backoff_base,
                )

        batch_results = await asyncio.gather(
            *[bounded_request(prompt) for prompt in batch_prompts]
        )
        all_results.extend(batch_results)

    return all_results


async def generate_remote_llm_outputs_async(
    *,
    prompts: list[Prompt],
    llm_config: Dict[str, Any],
    server_urls: list[str],
    api_key: str,
) -> tuple[list[str], list[str], list[str]]:
    """
    Run remote generation across one or more servers.

    Prompts are distributed round-robin across ``server_urls``. Each server runs
    its assigned prompts via ``run_inference_batch``. Returned text and finish
    reason lists are restored to input ``Prompt.row_idx`` order.
    """
    if not prompts:
        return [], [], []

    model_name = remote_request_model_name(llm_config)
    sampling_params = dict(llm_config["sampling_params"])
    required_remote_keys = [
        "server_urls",
        "max_concurrent_requests",
        "request_timeout",
        "max_retries",
        "batch_size",
    ]
    missing = [key for key in required_remote_keys if key not in llm_config]
    if missing:
        raise ValueError(
            "Remote backend requires llm_config keys: "
            f"{', '.join(required_remote_keys)}. Missing: {', '.join(missing)}"
        )

    max_concurrent_requests = max(1, int(llm_config["max_concurrent_requests"]))
    request_timeout = float(llm_config["request_timeout"])
    max_retries = max(1, int(llm_config["max_retries"]))
    retry_backoff_base = float(llm_config.get("retry_backoff_base", 1.0))
    batch_size = max(1, int(llm_config["batch_size"]))
    send_vllm_extra_body = bool(llm_config.get("send_vllm_extra_body", True))
    chat_template_kwargs = (
        llm_config.get("chat_template_kwargs") if send_vllm_extra_body else None
    )
    request_param_keys = {
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
    }
    request_params = {
        key: value
        for key, value in sampling_params.items()
        if key in request_param_keys
    }
    extra_body_params = (
        {
            key: value
            for key, value in sampling_params.items()
            if key not in request_param_keys and key != "max_tokens"
        }
        if send_vllm_extra_body
        else {}
    )
    server_clients = connect_to_remote_servers(
        server_urls=server_urls,
        request_timeout=request_timeout,
        api_key=api_key,
    )

    try:
        server_prompt_groups: list[list[Prompt]] = [[] for _ in server_clients]
        for i, prompt in enumerate(prompts):
            server_prompt_groups[i % len(server_clients)].append(prompt)

        tasks = []
        for server_idx, (client, port) in enumerate(server_clients):
            prompt_group = server_prompt_groups[server_idx]
            if prompt_group:
                tasks.append(
                    run_inference_batch(
                        client=client,
                        prompts=prompt_group,
                        model=model_name,
                        request_params=request_params,
                        extra_body_params=extra_body_params,
                        chat_template_kwargs=chat_template_kwargs,
                        max_concurrent=max_concurrent_requests,
                        batch_size=batch_size,
                        max_retries=max_retries,
                        base_timeout=request_timeout,
                        port=port,
                        retry_backoff_base=retry_backoff_base,
                    )
                )

        all_batch_results = await asyncio.gather(*tasks)
        results = [
            result for batch_results in all_batch_results for result in batch_results
        ]
        texts: list[str] = [""] * len(prompts)
        reasonings: list[str] = [""] * len(prompts)
        finish_reasons: list[str] = [""] * len(prompts)
        for result in results:
            texts[result.row_idx] = result.summary
            reasonings[result.row_idx] = result.reasoning
            finish_reasons[result.row_idx] = result.finish_reason
        return texts, reasonings, finish_reasons
    finally:
        close_tasks = [
            client.aclose() for client, _ in server_clients if hasattr(client, "aclose")
        ]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)


def generate_remote_llm_outputs(
    *,
    prompts: list[Prompt],
    llm_config: Dict[str, Any],
    server_urls: list[str],
    api_key: str,
) -> tuple[list[str], list[str], list[str]]:
    """Synchronous wrapper around ``generate_remote_llm_outputs_async``."""
    return _run_sync(
        lambda: generate_remote_llm_outputs_async(
            prompts=prompts,
            llm_config=llm_config,
            server_urls=server_urls,
            api_key=api_key,
        )
    )


__all__ = [
    "ModelResult",
    "Prompt",
    "connect_to_remote_servers",
    "generate_remote_llm_outputs",
    "generate_remote_llm_outputs_async",
    "normalize_openai_base_url",
    "normalize_remote_server_urls",
    "remote_request_model_name",
    "run_inference_batch",
    "single_inference_request",
]

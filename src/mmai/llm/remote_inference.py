"""Remote OpenAI-compatible inference orchestration."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, TypeVar, cast
from urllib.parse import urlparse

from mmai.llm.prompt_rendering import Prompt


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

    server_urls = [url for url in server_urls if url]
    if not server_urls:
        raise ValueError("Remote backend requires at least one server URL.")
    return server_urls


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
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    reasoning_marker: str = "assistantfinal",
    max_retries: int = 6,
    base_timeout: float = 600.0,
    retry_backoff_base: float = 1.0,
) -> ModelResult:
    """
    Send one completion request and retry transient failures.

    The returned ``ModelResult.row_idx`` is copied from the input so callers can
    restore original ordering after concurrent execution. This function does
    not split reasoning from summary; downstream package postprocessing expects
    the raw model text.
    """
    for attempt in range(max_retries):
        try:
            extra: dict[str, float | int] = {
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0.0:
                extra["min_p"] = min_p
            response = await asyncio.wait_for(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    extra_body=extra,
                ),
                timeout=base_timeout,
            )
            choice = response.choices[0]
            raw_text = cast(str, getattr(choice, "text", "") or "")
            finish_reason = cast(str, getattr(choice, "finish_reason", None) or "stop")
            return ModelResult(
                row_idx=row_idx,
                reasoning="",
                summary=raw_text,
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
    temperature: float,
    top_k: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    reasoning_marker: str = "assistantfinal",
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

        async def bounded_request(
            row_idx: int, prompt: str, prompt_max_tokens: int
        ) -> ModelResult:
            async with semaphore:
                return await single_inference_request(
                    client=client,
                    row_idx=row_idx,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=prompt_max_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    reasoning_marker=reasoning_marker,
                    max_retries=max_retries,
                    base_timeout=base_timeout,
                    retry_backoff_base=retry_backoff_base,
                )

        batch_results = await asyncio.gather(
            *[
                bounded_request(p.row_idx, p.prompt_text, p.max_tokens)
                for p in batch_prompts
            ]
        )
        all_results.extend(batch_results)

    return all_results


async def generate_remote_llm_outputs_async(
    *,
    prompts: list[Prompt],
    llm_config: Dict[str, Any],
    server_urls: list[str],
    api_key: str,
) -> tuple[list[str], list[str]]:
    """
    Run remote generation across one or more servers.

    Prompts are distributed round-robin across ``server_urls``. Each server runs
    its assigned prompts via ``run_inference_batch``. Returned text and finish
    reason lists are restored to input ``Prompt.row_idx`` order.
    """
    if not prompts:
        return [], []

    model_name = str(llm_config["model_name"])
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
    server_clients = connect_to_remote_servers(
        server_urls=server_urls,
        request_timeout=request_timeout,
        api_key=api_key,
    )

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
                    temperature=float(sampling_params["temperature"]),
                    top_k=int(sampling_params["top_k"]),
                    top_p=float(sampling_params.get("top_p", 1.0)),
                    presence_penalty=float(
                        sampling_params.get("presence_penalty", 0.0)
                    ),
                    min_p=float(sampling_params.get("min_p", 0.0)),
                    repetition_penalty=float(sampling_params["repetition_penalty"]),
                    reasoning_marker=str(
                        llm_config.get("reasoning_marker", "assistantfinal")
                    ),
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
    finish_reasons: list[str] = [""] * len(prompts)
    for result in results:
        texts[result.row_idx] = result.summary
        finish_reasons[result.row_idx] = result.finish_reason
    return texts, finish_reasons


def generate_remote_llm_outputs(
    *,
    prompts: list[Prompt],
    llm_config: Dict[str, Any],
    server_urls: list[str],
    api_key: str,
) -> tuple[list[str], list[str]]:
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
    "normalize_remote_server_urls",
    "run_inference_batch",
    "single_inference_request",
]

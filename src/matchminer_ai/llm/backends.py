"""Inference backend implementations and backend registry."""

from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, cast

from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.llm.reasoning import parse_reasoning_output
from matchminer_ai.llm.reasoning import resolve_reasoning_parser
from matchminer_ai.llm.remote_inference import generate_remote_llm_outputs
from matchminer_ai.llm.remote_inference import normalize_remote_server_urls
from matchminer_ai.llm.remote_inference import remote_request_model_name

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


@dataclass(frozen=True)
class LLMGenerationResult:
    """Structured output from one backend generation call."""

    final_outputs: list[str]
    model_metadata: Dict[str, Any]
    finish_reasons: list[str]
    reasoning_outputs: list[str]
    raw_outputs: list[str]


def _default_metadata_cache_dir() -> str:
    """Return the default on-disk cache directory for model metadata."""
    return os.path.join(
        os.path.expanduser("~"), ".cache", "matchminer_ai", "model_metadata"
    )


def create_model_metadata(model_name: str) -> Dict[str, Any]:
    """Fetch immutable model metadata from Hugging Face Hub."""
    from huggingface_hub import model_info

    metadata = model_info(model_name)
    return {
        "model_name": model_name,
        "model_sha": metadata.sha,
        "created_at": metadata.created_at.isoformat(),
        "last_modified": metadata.last_modified.isoformat(),
    }


def get_model_metadata(
    model_name: str,
    *,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Load model metadata from cache, fetching it if needed.

    Metadata is stored by model name so summarization and embedding outputs can
    include model provenance without repeated Hugging Face Hub calls.
    """
    cache_dir = cache_dir or _default_metadata_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}.json")

    model_dict: Dict[str, Any]
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as handle:
            model_dict = json.load(handle)
        if not isinstance(model_dict, dict):
            raise ValueError(f"Cached metadata for {model_name} is not a mapping.")
    else:
        model_dict = create_model_metadata(model_name)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(model_dict, handle)

    return model_dict


def get_endpoint_model_metadata(
    model_name: str,
    *,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Return model metadata for a remote endpoint model name.

    OpenAI-compatible endpoints can expose names that are not Hugging Face model
    IDs. In that case, return synthetic provenance metadata instead of failing
    the run before the endpoint is contacted.
    """
    try:
        return get_model_metadata(model_name, cache_dir=cache_dir)
    except Exception as exc:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "model_name": model_name,
            "model_sha": "openai-compatible-endpoint",
            "created_at": now,
            "last_modified": now,
            "metadata_error": str(exc),
        }


@lru_cache(maxsize=2)
def _get_local_llm(
    model_name: str,
    llm_kwargs_json: str,
):
    """Load and cache a local vLLM model instance from config kwargs."""
    from vllm import LLM

    llm_kwargs = json.loads(llm_kwargs_json)
    return LLM(model=model_name, **llm_kwargs)


def clear_local_llm_cache() -> None:
    """Release cached local vLLM engine handles and clear Python/GPU caches."""
    _get_local_llm.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class LocalBackend:
    """Local vLLM-backed implementation."""

    def generate_llm_outputs(
        self,
        *,
        prompt_list: list[Prompt],
        llm_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate LLM outputs with a local vLLM model.

        Parameters
        ----------
        prompt_list
            Rendered prompts to send to vLLM. Output order follows this list.
        llm_config
            Model and sampling configuration. Local-engine fields from
            ``config.local.<task>`` are passed through to ``vllm.LLM`` as
            keyword arguments, and ``sampling_params`` is passed through to
            ``vllm.SamplingParams`` as keyword arguments.

        Returns
        -------
        LLMGenerationResult
            Final outputs, reasoning traces, raw local outputs, finish reasons,
            and model metadata.
        """
        from vllm import SamplingParams

        model_name = llm_config["model_name"]
        llm_kwargs = {
            key: value
            for key, value in llm_config.items()
            if key
            not in {
                "model_name",
                "sampling_params",
                "prompt_file",
                "prompt_files",
                "reasoning_parser",
                "chat_template_kwargs",
                "boilerplate_marker",
                "chunk_size",
                "chunk_overlap",
                "prompt_margin_tokens",
                "text_token_threshold",
                "prompt_build_workers",
                "model_metadata_cache_dir",
                "vllm_server_args",
            }
        }
        sampling_params = dict(llm_config["sampling_params"])
        parser_name = resolve_reasoning_parser(
            str(model_name),
            str(llm_config.get("reasoning_parser", "auto")),
        )

        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        llm = _get_local_llm(
            model_name,
            json.dumps(llm_kwargs, sort_keys=True),
        )
        prompts = [prompt.prompt_text for prompt in prompt_list]
        responses = llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(**sampling_params),
        )
        raw_texts = [response.outputs[0].text for response in responses]
        tokenizer = llm.get_tokenizer()
        parsed_outputs = [
            parse_reasoning_output(
                text,
                parser_name=parser_name,
                tokenizer=tokenizer,
            )
            for text in raw_texts
        ]
        reasonings = [reasoning for reasoning, _content in parsed_outputs]
        texts = [content for _reasoning, content in parsed_outputs]
        finish_reasons = [
            cast(str, response.outputs[0].finish_reason) for response in responses
        ]
        return LLMGenerationResult(
            final_outputs=texts,
            model_metadata=model_metadata,
            finish_reasons=finish_reasons,
            reasoning_outputs=reasonings,
            raw_outputs=raw_texts,
        )

    def truncate_texts(
        self,
        texts: list[str],
        *,
        patient_config: Dict[str, Any],
    ) -> list[str]:
        """Truncate long texts using the model tokenizer."""
        from transformers import AutoTokenizer

        model_name = patient_config["model_name"]
        text_token_threshold = int(patient_config["text_token_threshold"])
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        truncated: list[str] = []
        for text in texts:
            text_tokens = tokenizer(text, add_special_tokens=False).input_ids
            if len(text_tokens) > text_token_threshold:
                first_part = text_tokens[: text_token_threshold // 2]
                last_part = text_tokens[-text_token_threshold // 2 :]
                text = (
                    tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)
                )
            truncated.append(text)
        return truncated


@dataclass
class RemoteBackend:
    """OpenAI-compatible remote vLLM HTTP backend."""

    def generate_llm_outputs(
        self,
        *,
        prompt_list: list[Prompt],
        llm_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> LLMGenerationResult:
        """
        Generate LLM outputs through one or more remote vLLM servers.

        Remote execution uses OpenAI-compatible chat completions, distributes
        prompts across configured server URLs, and restores outputs to
        ``Prompt.row_idx`` order.
        """
        model_name = remote_request_model_name(llm_config)
        api_key = str(os.environ.get("OPENAI_API_KEY", "not-needed")).strip()
        server_urls = normalize_remote_server_urls(llm_config)
        model_metadata = get_endpoint_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        texts, reasonings, finish_reasons = generate_remote_llm_outputs(
            prompts=prompt_list,
            llm_config=llm_config,
            server_urls=server_urls,
            api_key=api_key,
        )
        return LLMGenerationResult(
            final_outputs=texts,
            model_metadata=model_metadata,
            finish_reasons=finish_reasons,
            reasoning_outputs=reasonings,
            raw_outputs=[],
        )


def get_backend(name: str) -> LocalBackend | RemoteBackend:
    """Return a backend by name."""
    if name == "local":
        return LocalBackend()
    if name == "remote":
        return RemoteBackend()
    raise ValueError(f"Unsupported backend: {name}")


def remote_enabled(config: "MMAIConfig") -> bool:
    """Return whether remote LLM inference is enabled for summarization."""
    return bool(getattr(config, "remote", {}).get("enabled", False))


def build_summarization_runtime_config(
    task_name: str,
    llm_config: Dict[str, Any],
    *,
    config: "MMAIConfig",
) -> Dict[str, Any]:
    """Merge task LLM settings with mode-specific runtime settings."""
    runtime_config = dict(llm_config)
    local_config = dict(getattr(config, "local", {}).get(task_name, {}))
    runtime_config.update(local_config)
    if remote_enabled(config):
        remote_config = dict(getattr(config, "remote", {}))
        remote_config.pop("enabled", None)
        runtime_config.update(remote_config)
    return runtime_config


def get_summarization_backend(config: "MMAIConfig") -> LocalBackend | RemoteBackend:
    """Return the backend used for trial/patient LLM summarization."""
    if remote_enabled(config):
        return RemoteBackend()
    return LocalBackend()

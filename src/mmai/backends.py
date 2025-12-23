"""Backend registry stubs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _default_metadata_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "mmai", "model_metadata")


def create_model_metadata(model_name: str) -> Dict[str, Any]:
    from huggingface_hub import model_info

    metadata = model_info(model_name)
    return {
        "model_sha": metadata.sha,
        "created_at": metadata.created_at.isoformat(),
        "last_modified": metadata.last_modified.isoformat(),
    }


def get_model_metadata(
    model_name: str,
    *,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
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


@dataclass
class LocalBackend:
    """Local vLLM-backed implementation."""

    def generate_from_messages(
        self,
        *,
        messages_list: list[list[dict[str, str]]],
        model_name: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        sampling_params: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[str], Dict[str, Any]]:
        from vllm import LLM, SamplingParams

        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            for messages in messages_list
        ]
        responses = llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                max_tokens=sampling_params["max_tokens"],
                repetition_penalty=sampling_params["repetition_penalty"],
            ),
        )
        return [response.outputs[0].text for response in responses], model_metadata

    def get_llm(
        self,
        *,
        model_name: str,
        max_model_len: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        sampling_params: Dict[str, Any] | None = None,
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        from vllm import LLM

        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        return (
            LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            ),
            model_metadata,
        )


def get_backend(name: str) -> LocalBackend:
    """Return a backend by name."""
    if name == "local":
        return LocalBackend()
    raise ValueError(f"Unsupported backend: {name}")

"""Backend registry stubs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple


def _default_metadata_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "mmai", "model_metadata")


def add_run_specific_info(
    model_dict: Dict[str, Any], sampling_params: Dict[str, Any] | None
) -> Dict[str, Any]:
    if sampling_params:
        model_dict["trial_model_sampling_param_temperature"] = sampling_params.get(
            "temperature"
        )
        model_dict["trial_model_sampling_param_top_k"] = sampling_params.get("top_k")
        model_dict["trial_model_sampling_param_max_tokens"] = sampling_params.get(
            "max_tokens"
        )
        model_dict["trial_model_sampling_param_repetition_penalty"] = (
            sampling_params.get("repetition_penalty")
        )
    return model_dict


def create_model_metadata(model_name: str) -> Dict[str, Any]:
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
    sampling_params: Dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    cache_dir = cache_dir or _default_metadata_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as handle:
            model_dict = json.load(handle)
    else:
        model_dict = create_model_metadata(model_name)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(model_dict, handle)

    return add_run_specific_info(model_dict, sampling_params)


@dataclass
class LocalBackend:
    """Local vLLM-backed implementation."""

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
            sampling_params=sampling_params,
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

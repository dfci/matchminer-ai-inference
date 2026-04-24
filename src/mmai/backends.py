"""Inference backend implementations and backend registry."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any, Dict, Tuple, cast

from mmai.prompt_rendering import Prompt
from mmai.remote_inference import generate_remote_llm_outputs
from mmai.remote_inference import normalize_remote_server_urls

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def _default_metadata_cache_dir() -> str:
    """Return the default on-disk cache directory for model metadata."""
    return os.path.join(os.path.expanduser("~"), ".cache", "mmai", "model_metadata")


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


@lru_cache(maxsize=4)
def _get_embedding_model(model_path: str, device: str, prompt: str):
    """Load and cache a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    model.prompts["query"] = prompt
    return model


@lru_cache(maxsize=2)
def _get_local_llm(
    model_name: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    """Load and cache a local vLLM model instance."""
    from vllm import LLM

    return LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _load_prompt_text(filename: str) -> str:
    """Load a prompt text asset bundled with the package."""
    prompt_path = resources.files("mmai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _resolve_embedding_runtime(
    embedding_config: Dict[str, Any],
) -> tuple[str, str, str]:
    """Resolve embedding model path, device, and query prompt text."""
    model_path = str(embedding_config.get("model_path", "")).strip()
    device = str(embedding_config.get("device", "cpu")).strip() or "cpu"
    prompt_filename = str(embedding_config.get("prompt_file", "")).strip()
    query_prompt = _load_prompt_text(prompt_filename).strip()
    return model_path, device, query_prompt


@dataclass
class LocalBackend:
    """Local vLLM-backed implementation."""

    def generate_llm_outputs(
        self,
        *,
        prompt_list: list[Prompt],
        llm_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[str], Dict[str, Any], list[str]]:
        """
        Generate LLM outputs with a local vLLM model.

        Parameters
        ----------
        prompt_list
            Rendered prompts to send to vLLM. Output order follows this list.
        llm_config
            Model and sampling configuration.

        Returns
        -------
        tuple
            ``(texts, model_metadata, finish_reasons)``.
        """
        from vllm import SamplingParams

        model_name = llm_config["model_name"]
        max_model_len = llm_config["max_model_len"]
        tensor_parallel_size = llm_config["tensor_parallel_size"]
        gpu_memory_utilization = llm_config["gpu_memory_utilization"]
        sampling_params = dict(llm_config["sampling_params"])

        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        llm = _get_local_llm(
            model_name,
            int(tensor_parallel_size),
            int(max_model_len),
            float(gpu_memory_utilization),
        )
        prompts = [prompt.prompt_text for prompt in prompt_list]
        responses = llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                temperature=sampling_params["temperature"],
                top_k=sampling_params["top_k"],
                max_tokens=sampling_params["max_tokens"],
                repetition_penalty=sampling_params["repetition_penalty"],
            ),
        )
        texts = [response.outputs[0].text for response in responses]
        finish_reasons = [
            cast(str, response.outputs[0].finish_reason) for response in responses
        ]
        return texts, model_metadata, finish_reasons

    def run_checker(
        self,
        prompts: list[str],
        *,
        checker_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
        """Run a text-classification checker model on prompts."""
        from transformers import AutoTokenizer, pipeline

        model_name = checker_config["model_name"]
        device = checker_config["device"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_metadata_cache_dir,
            trust_remote_code=True,
        )
        checker_pipeline = pipeline(
            "text-classification",
            model_name,
            tokenizer=tokenizer,
            truncation=True,
            padding="max_length",
            max_length=4096,
            device=device,
        )
        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        outputs = cast(
            list[dict[str, Any]],
            checker_pipeline(prompts),
        )
        return outputs, model_metadata

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

    def generate_embeddings(
        self,
        texts: list[str],
        *,
        embedding_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[list[float]], Dict[str, Any]]:
        """Generate sentence-transformer embeddings and model metadata."""
        model_path, device, query_prompt = _resolve_embedding_runtime(embedding_config)
        model = _get_embedding_model(model_path, device, query_prompt)
        model_metadata = get_model_metadata(
            model_path,
            cache_dir=model_metadata_cache_dir,
        )
        embeddings = model.encode(texts, prompt="query")
        embedding_list = (
            embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        )
        return cast(list[list[float]], embedding_list), model_metadata

    def count_embedding_tokens(
        self,
        texts: list[str],
        *,
        embedding_config: Dict[str, Any],
    ) -> list[int]:
        """Count embedding-model input tokens after applying the query prompt."""
        model_path, device, query_prompt = _resolve_embedding_runtime(embedding_config)
        model = _get_embedding_model(model_path, device, query_prompt)
        prepared = [f"{query_prompt} {text}".strip() for text in texts]
        encoded = model.tokenizer(prepared, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        return [len(input_ids) for input_ids in encoded]


@dataclass
class RemoteBackend:
    """OpenAI-compatible remote vLLM HTTP backend."""

    def generate_llm_outputs(
        self,
        *,
        prompt_list: list[Prompt],
        llm_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[str], Dict[str, Any], list[str]]:
        """
        Generate LLM outputs through one or more remote vLLM servers.

        Remote execution uses OpenAI-compatible completions, distributes prompts
        across configured server URLs, and restores outputs to ``Prompt.row_idx``
        order.
        """
        model_name = str(llm_config["model_name"])
        api_key = str(os.environ.get("OPENAI_API_KEY", "not-needed")).strip()
        server_urls = normalize_remote_server_urls(llm_config)
        model_metadata = get_model_metadata(
            model_name,
            cache_dir=model_metadata_cache_dir,
        )
        texts, finish_reasons = generate_remote_llm_outputs(
            prompts=prompt_list,
            llm_config=llm_config,
            server_urls=server_urls,
            api_key=api_key,
        )

        return texts, model_metadata, finish_reasons

    def truncate_texts(
        self,
        texts: list[str],
        *,
        patient_config: Dict[str, Any],
    ) -> list[str]:
        raise NotImplementedError("Remote backend is not implemented yet.")

    def generate_embeddings(
        self,
        texts: list[str],
        *,
        embedding_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[list[float]], Dict[str, Any]]:
        raise NotImplementedError("Remote backend is not implemented yet.")

    def count_embedding_tokens(
        self,
        texts: list[str],
        *,
        embedding_config: Dict[str, Any],
    ) -> list[int]:
        """Count embedding-model input tokens after applying the query prompt."""
        model_path, device, query_prompt = _resolve_embedding_runtime(embedding_config)
        model = _get_embedding_model(model_path, device, query_prompt)
        prepared = [f"{query_prompt} {text}".strip() for text in texts]
        encoded = model.tokenizer(prepared, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        return [len(input_ids) for input_ids in encoded]

    def run_checker(
        self,
        prompts: list[str],
        *,
        checker_config: Dict[str, Any],
        model_metadata_cache_dir: str | None = None,
    ) -> Tuple[list[dict[str, Any]], Dict[str, Any]]:
        raise NotImplementedError("Remote backend is not implemented yet.")


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

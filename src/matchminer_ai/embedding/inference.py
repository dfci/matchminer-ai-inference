"""Embedding-model inference helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Dict, cast

from matchminer_ai.llm.backends import get_model_metadata


def _load_prompt_text(filename: str) -> str:
    """Load a prompt text asset bundled with the package."""
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _resolve_embedding_runtime(
    embedding_config: Dict[str, Any],
) -> tuple[str, str, str, int | None]:
    """Resolve embedding model path, device, query prompt text, and length."""
    model_path = str(embedding_config.get("model_path", "")).strip()
    device = str(embedding_config.get("device", "cpu")).strip() or "cpu"
    prompt_filename = str(embedding_config.get("prompt_file", "")).strip()
    max_seq_length_value = embedding_config.get("max_seq_length")
    max_seq_length = (
        None if max_seq_length_value is None else int(max_seq_length_value)
    )
    query_prompt = _load_prompt_text(prompt_filename).strip()
    return model_path, device, query_prompt, max_seq_length


@lru_cache(maxsize=4)
def _get_embedding_model(
    model_path: str,
    device: str,
    prompt: str,
    max_seq_length: int | None,
):
    """Load and cache a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    model.prompts["query"] = prompt
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


@lru_cache(maxsize=4)
def _get_embedding_tokenizer(model_path: str):
    """Load and cache the tokenizer for embedding token counts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def generate_embeddings(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[list[float]], Dict[str, Any]]:
    """Generate sentence-transformer embeddings and model metadata."""
    model_path, device, query_prompt, max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    model = _get_embedding_model(model_path, device, query_prompt, max_seq_length)
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
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
) -> list[int]:
    """Count embedding-model input tokens after applying the query prompt."""
    model_path, _device, query_prompt, _max_seq_length = _resolve_embedding_runtime(
        embedding_config
    )
    tokenizer = _get_embedding_tokenizer(model_path)
    prepared = [f"{query_prompt} {text}".strip() for text in texts]
    encoded = tokenizer(prepared, add_special_tokens=True, truncation=False)
    input_ids = (
        encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    )
    return [len(ids) for ids in input_ids]


__all__ = [
    "count_embedding_tokens",
    "generate_embeddings",
]

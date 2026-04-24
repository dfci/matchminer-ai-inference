"""Embedding-model inference helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Dict, cast

from mmai.llm.backends import get_model_metadata


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


@lru_cache(maxsize=4)
def _get_embedding_model(model_path: str, device: str, prompt: str):
    """Load and cache a SentenceTransformer embedding model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=device)
    model.prompts["query"] = prompt
    return model


def generate_embeddings(
    texts: list[str],
    *,
    embedding_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[list[float]], Dict[str, Any]]:
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


__all__ = [
    "count_embedding_tokens",
    "generate_embeddings",
]

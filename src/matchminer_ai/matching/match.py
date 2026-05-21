"""Candidate match generation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _stack_embeddings(values: pd.Series) -> torch.Tensor:
    vectors = []
    for value in values:
        if isinstance(value, np.ndarray):
            vectors.append(value.astype(float))
            continue
        if isinstance(value, list):
            vectors.append(np.asarray(value, dtype=float))
            continue
        raise ValueError("Embeddings must be list or numpy array values.")
    if not vectors:
        return torch.empty((0, 0), dtype=torch.float32)
    dim = max(vec.size for vec in vectors)
    if any(vec.size != dim for vec in vectors):
        raise ValueError("Embedding vectors must share the same dimension.")
    return torch.tensor(np.vstack(vectors).astype(float), dtype=torch.float32)


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    return F.normalize(matrix, p=2, dim=1)


def _resolve_id_column(df: pd.DataFrame, label: str) -> str:
    candidates = [col for col in ("patient_id", "space_trial_id") if col in df.columns]
    if not candidates:
        raise ValueError(f"{label} must include either patient_id or space_trial_id")
    if len(candidates) > 1:
        raise ValueError(
            f"{label} must include only one identifier column; found {candidates}"
        )
    id_col = candidates[0]
    if df[id_col].duplicated().any():
        raise ValueError(f"{label} must have unique {id_col} values")
    return id_col


def generate_candidate_matches(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    *,
    k: int | None = 20,
) -> pd.DataFrame:
    """
    Generate top-k candidate matches by ranking corpus items for each query item.

    Parameters
    ----------
    query_df : pd.DataFrame
        Query-side DataFrame with embeddings.
        Must contain:
            - patient_id or space_trial_id: str
            - embedding : array-like
    corpus_df : pd.DataFrame
        Corpus-side DataFrame with embeddings.
        Must contain:
            - patient_id or space_trial_id: str
            - embedding : array-like
    k : int | None, default 20
        Number of top matches to return per query entity. If None, return all
        corpus items per query (sorted by similarity).

    Returns
    -------
    pd.DataFrame
        Ranked top-k match pairs including:
            - patient_id or space_trial_id (from query_df)
            - patient_id or space_trial_id (from corpus_df)
            - similarity_score : float
            - rank : int (1 = highest similarity per query)
    """
    # Validate required schema for IDs and embeddings.
    query_id_col = _resolve_id_column(query_df, "query_df")
    corpus_id_col = _resolve_id_column(corpus_df, "corpus_df")
    if "embedding" not in query_df.columns:
        raise ValueError("query_df must include embedding")
    if "embedding" not in corpus_df.columns:
        raise ValueError("corpus_df must include embedding")
    if k is not None and k <= 0:
        raise ValueError("k must be a positive integer or None")

    # Parse embeddings and ensure consistent vector dimensionality.
    query_embeddings = _stack_embeddings(query_df["embedding"])
    corpus_embeddings = _stack_embeddings(corpus_df["embedding"])
    if query_embeddings.numel() == 0 or corpus_embeddings.numel() == 0:
        return pd.DataFrame(
            columns=[query_id_col, corpus_id_col, "similarity_score", "rank"]
        )

    # Normalize embeddings and compute cosine similarity matrix.
    query_norm = _normalize_rows(query_embeddings)
    corpus_norm = _normalize_rows(corpus_embeddings)
    similarity = query_norm @ corpus_norm.T

    # Extract top-k corpus matches per query.
    query_ids = query_df[query_id_col].astype(str).tolist()
    corpus_ids = corpus_df[corpus_id_col].astype(str).tolist()

    records: list[dict[str, object]] = []
    num_corpus = similarity.shape[1]
    top_k = num_corpus if k is None else min(k, num_corpus)
    for row_idx, query_id in enumerate(query_ids):
        scores = similarity[row_idx]
        if top_k == 0:
            continue
        if top_k == num_corpus:
            ranked_idx = torch.argsort(scores, descending=True)
        else:
            ranked_idx = torch.topk(scores, k=top_k, largest=True).indices
        for rank, corpus_idx in enumerate(ranked_idx, start=1):
            corpus_idx_int = int(corpus_idx)
            records.append(
                {
                    query_id_col: query_id,
                    corpus_id_col: corpus_ids[corpus_idx_int],
                    "similarity_score": float(scores[corpus_idx_int]),
                    "rank": int(rank),
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    return result

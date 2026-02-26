"""Embedding regression tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from mmai.embedding import embed_for_matching
from mmai.patients import summarize_from_relevant_sentences
from mmai.trials import summarize_trials

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


REGRESSION_DATA_DIR = (
    Path(__file__).resolve().parent / "data" / "embedding_regression" / "mmai-synthetic"
)


def _load_patient_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "patient_input.csv")


def _load_patient_output_gold() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "patient_output.csv")


def _load_trial_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "trial_input.csv")


def _load_trial_output_gold() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "trial_output.csv")


def _generate_patient_package_embeddings(
    patient_input: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> pd.DataFrame:
    """Run patient summarize+embed pipeline from post-tagger input."""
    patient_input = patient_input[["patient_id", "patient_long_text"]].copy()
    patient_summaries, _ = cast(
        tuple[pd.DataFrame, dict],
        summarize_from_relevant_sentences(
            patient_input,
            config=config,
            return_qc=False,
        ),
    )
    patient_embedded = embed_for_matching(
        patient_summaries,
        entity_type="patient",
        config=config,
    )
    return patient_embedded[["patient_id", "embedding"]].copy()


def _generate_trial_package_embeddings(
    trial_input: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> pd.DataFrame:
    """Run trial summarize+embed pipeline from raw trial-level input."""
    trial_input = trial_input[
        ["trial_id", "trial_title", "brief_summary", "eligibility_criteria"]
    ].copy()
    trial_spaces = cast(
        pd.DataFrame,
        summarize_trials(
            trial_input,
            config=config,
            return_qc=False,
        ),
    )
    trial_embedded = embed_for_matching(
        trial_spaces,
        entity_type="trial",
        config=config,
    )
    return trial_embedded[["space_trial_id", "trial_id", "embedding"]].copy()


def _parse_embedding_vector(value: object) -> np.ndarray:
    """Parse embedding values from arrays/lists or serialized bracket strings."""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=float)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not text:
        return np.asarray([], dtype=float)
    return np.fromstring(text, sep=" ", dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _compare_embedding_frames(
    package_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    id_col: str,
    package_embedding_col: str = "embedding",
    gold_embedding_col: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Compare package and gold embeddings by id and return per-id cosine scores.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        (scores_df, missing_in_package, missing_in_gold)
    """
    package = package_df.copy()
    gold = gold_df.copy()

    package[id_col] = package[id_col].astype(str)
    gold[id_col] = gold[id_col].astype(str)

    package_ids = set(package[id_col])
    gold_ids = set(gold[id_col])
    missing_in_package = sorted(gold_ids - package_ids)
    missing_in_gold = sorted(package_ids - gold_ids)

    merged = package[[id_col, package_embedding_col]].merge(
        gold[[id_col, gold_embedding_col]],
        on=id_col,
        how="inner",
    )
    merged["cosine_similarity"] = merged.apply(
        lambda row: _cosine_similarity(
            _parse_embedding_vector(row[package_embedding_col]),
            _parse_embedding_vector(row[gold_embedding_col]),
        ),
        axis=1,
    )
    return merged[[id_col, "cosine_similarity"]], missing_in_package, missing_in_gold


def _compare_patient_package_vs_gold(
    patient_package_embeddings: pd.DataFrame,
    patient_gold: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    return _compare_embedding_frames(
        patient_package_embeddings,
        patient_gold,
        id_col="patient_id",
        package_embedding_col="embedding",
        gold_embedding_col="patient_embedding",
    )


def _compare_trial_package_vs_gold(
    trial_package_embeddings: pd.DataFrame,
    trial_gold: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    return _compare_embedding_frames(
        trial_package_embeddings,
        trial_gold,
        id_col="space_trial_id",
        package_embedding_col="embedding",
        gold_embedding_col="trial_embedding",
    )

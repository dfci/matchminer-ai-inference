"""Embedding regression tests.

Note: run patient and trial tests separately to avoid vLLM hangs:
- pytest -m resource_heavy tests/test_embedding_regression.py -k patient
- pytest -m resource_heavy tests/test_embedding_regression.py -k trial

To see progress logs while running, add -s (disable capture):
- pytest -m resource_heavy tests/test_embedding_regression.py -k patient -s
- pytest -m resource_heavy tests/test_embedding_regression.py -k trial -s
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
import torch

from mmai.embedding import embed_for_matching
from mmai.patients import summarize_patients
from mmai.trials import summarize_trials

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


REGRESSION_DATA_DIR = (
    Path(__file__).resolve().parent / "data" / "embedding_regression" / "mmai-synthetic"
)


def _load_patient_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "patient_input.csv")


def _load_patient_output_gold() -> pd.DataFrame:
    return pd.read_parquet(REGRESSION_DATA_DIR / "patient_output.parquet")


def _load_trial_input() -> pd.DataFrame:
    return pd.read_csv(REGRESSION_DATA_DIR / "trial_input.csv")


def _load_trial_output_gold() -> pd.DataFrame:
    return pd.read_parquet(REGRESSION_DATA_DIR / "trial_output.parquet")


def _generate_patient_package_embeddings(
    patient_input: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> pd.DataFrame:
    """Run patient summarize+embed pipeline from raw note-level input."""
    patient_notes = patient_input[["patient_id", "note_text", "note_date"]].copy()
    patient_summaries = cast(
        pd.DataFrame,
        summarize_patients(
            patient_notes,
            config=config,
            return_qc=False,
        ),
    )
    patient_embedded = cast(
        pd.DataFrame,
        embed_for_matching(
            patient_summaries,
            entity_type="patient",
            config=config,
        ),
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
    trial_embedded = cast(
        pd.DataFrame,
        embed_for_matching(
            trial_spaces,
            entity_type="trial",
            config=config,
        ),
    )
    return trial_embedded[["space_trial_id", "embedding"]].copy()


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
    a_tensor = torch.tensor(a, dtype=torch.float32)
    b_tensor = torch.tensor(b, dtype=torch.float32)
    if a_tensor.numel() == 0 or b_tensor.numel() == 0:
        return 0.0
    a_norm = torch.nn.functional.normalize(a_tensor, p=2, dim=0)
    b_norm = torch.nn.functional.normalize(b_tensor, p=2, dim=0)
    return float(torch.dot(a_norm, b_norm))


def _pairwise_cosine_similarity(
    lhs_vectors: list[np.ndarray], rhs_vectors: list[np.ndarray]
) -> np.ndarray:
    """Compute pairwise cosine similarity using the same torch path as matching."""
    if not lhs_vectors or not rhs_vectors:
        return np.zeros((len(lhs_vectors), len(rhs_vectors)), dtype=float)
    dim = lhs_vectors[0].size
    if any(vec.size != dim for vec in lhs_vectors + rhs_vectors):
        raise ValueError("Embedding vectors must share the same dimension.")
    lhs_tensor = torch.tensor(np.vstack(lhs_vectors).astype(float), dtype=torch.float32)
    rhs_tensor = torch.tensor(np.vstack(rhs_vectors).astype(float), dtype=torch.float32)
    lhs_norm = torch.nn.functional.normalize(lhs_tensor, p=2, dim=1)
    rhs_norm = torch.nn.functional.normalize(rhs_tensor, p=2, dim=1)
    return (lhs_norm @ rhs_norm.T).cpu().numpy()


def _compare_embedding_frames(
    package_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    id_col: str,
    package_embedding_col: str = "embedding",
    gold_embedding_col: str,
) -> pd.DataFrame:
    """
    Compare package and gold embeddings by id and return per-id cosine scores.

    Returns
    -------
    pd.DataFrame
        Per-id cosine similarity table.
    """
    package = package_df.copy()
    gold = gold_df.copy()

    package[id_col] = package[id_col].astype(str)
    gold[id_col] = gold[id_col].astype(str)

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
    return merged[[id_col, "cosine_similarity"]]


def _compare_patient_package_vs_gold(
    patient_package_embeddings: pd.DataFrame,
    patient_gold: pd.DataFrame,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    """
    Compare trial embeddings by trial_id using bidirectional best-match cosine.

    For each trial:
    - pkg_to_gold: mean best gold match for each package space
    - gold_to_pkg: mean best package match for each gold space
    - trial_score: average of the two means
    """
    package = trial_package_embeddings.copy()
    gold = trial_gold.copy()

    if "trial_id" not in package.columns and "space_trial_id" in package.columns:
        package["trial_id"] = (
            package["space_trial_id"].astype(str).str.split(r"[-_]").str[0]
        )
    if "trial_id" not in gold.columns and "space_trial_id" in gold.columns:
        gold["trial_id"] = gold["space_trial_id"].astype(str).str.split(r"[-_]").str[0]

    if "trial_id" not in package.columns or "trial_id" not in gold.columns:
        raise ValueError("trial_id is required for trial embedding comparison.")

    package["trial_id"] = package["trial_id"].astype(str)
    gold["trial_id"] = gold["trial_id"].astype(str)

    package_trials = set(package["trial_id"])
    gold_trials = set(gold["trial_id"])

    rows: list[dict[str, object]] = []
    # For each shared trial_id, compare the set of package spaces to the set of gold spaces.
    common_trials = sorted(package_trials & gold_trials)
    for trial_id in common_trials:
        pkg_vectors = [
            _parse_embedding_vector(value)
            for value in package.loc[package["trial_id"] == trial_id, "embedding"]
        ]
        gold_vectors = [
            _parse_embedding_vector(value)
            for value in gold.loc[gold["trial_id"] == trial_id, "trial_embedding"]
        ]
        if not pkg_vectors or not gold_vectors:
            rows.append(
                {
                    "trial_id": trial_id,
                    "n_pkg_spaces": len(pkg_vectors),
                    "n_gold_spaces": len(gold_vectors),
                    "pkg_to_gold": 0.0,
                    "gold_to_pkg": 0.0,
                    "trial_score": 0.0,
                }
            )
            continue

        # Build pairwise cosine matrix: each row is a package space and each column is a gold space.
        sim_matrix = _pairwise_cosine_similarity(pkg_vectors, gold_vectors)
        # Direction 1: for each package space, keep its best-matching gold space, then average.
        pkg_to_gold = float(sim_matrix.max(axis=1).mean())
        # Direction 2: for each gold space, keep its best-matching package space, then average.
        gold_to_pkg = float(sim_matrix.max(axis=0).mean())
        rows.append(
            {
                "trial_id": trial_id,
                "n_pkg_spaces": len(pkg_vectors),
                "n_gold_spaces": len(gold_vectors),
                "pkg_to_gold": pkg_to_gold,
                "gold_to_pkg": gold_to_pkg,
                "trial_score": (pkg_to_gold + gold_to_pkg) / 2.0,
            }
        )

    return pd.DataFrame(rows)


@pytest.mark.resource_heavy
def test_patient_embedding_regression_mmai_synthetic():
    """Simple patient embedding regression check against gold outputs."""
    print("Loading patient regression inputs...")
    patient_input = _load_patient_input()
    print(f"Loaded {len(patient_input)} patient inputs.")
    patient_gold = _load_patient_output_gold()
    print(f"Loaded {len(patient_gold)} patient gold embeddings.")
    print("Generating patient package embeddings...")
    package_embeddings = _generate_patient_package_embeddings(patient_input)
    print(f"Generated {len(package_embeddings)} patient embeddings.")
    print("Comparing patient embeddings to gold...")
    scores = _compare_patient_package_vs_gold(package_embeddings, patient_gold)
    assert not scores.empty, "No patient embeddings were comparable to gold output."
    print(f"Compared {len(scores)} patient embeddings.")
    print(scores)

    mean_score = float(scores["cosine_similarity"].mean())
    if mean_score < 0.8:
        worst = scores.nsmallest(5, "cosine_similarity").to_dict("records")
        raise AssertionError(
            "Patient embedding regression drift detected: "
            f"mean cosine_similarity={mean_score:.6f} (< 0.8). "
            f"min={float(scores['cosine_similarity'].min()):.6f}. "
            f"worst_5={worst}"
        )


@pytest.mark.resource_heavy
def test_trial_embedding_regression_mmai_synthetic():
    """Simple trial embedding regression check against gold outputs."""
    print("Loading trial regression inputs...")
    trial_input = _load_trial_input()
    print(f"Loaded {len(trial_input)} trial inputs.")
    trial_gold = _load_trial_output_gold()
    print(f"Loaded {len(trial_gold)} trial gold embeddings.")
    print("Generating trial package embeddings...")
    package_embeddings = _generate_trial_package_embeddings(trial_input)
    print(f"Generated {len(package_embeddings)} trial embeddings.")
    print("Comparing trial embeddings to gold...")
    scores = _compare_trial_package_vs_gold(package_embeddings, trial_gold)
    assert not scores.empty, "No trial embeddings were comparable to gold output."
    print(f"Compared {len(scores)} trial embeddings.")
    print(scores)

    mean_score = float(scores["trial_score"].mean())
    if mean_score < 0.8:
        worst = scores.nsmallest(5, "trial_score").to_dict("records")
        raise AssertionError(
            "Trial embedding regression drift detected: "
            f"mean trial_score={mean_score:.6f} (< 0.8). "
            f"min={float(scores['trial_score'].min()):.6f}. "
            f"worst_5={worst}"
        )

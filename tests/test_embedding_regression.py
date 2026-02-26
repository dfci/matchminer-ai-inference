"""Embedding regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from mmai.patients import summarize_from_relevant_sentences
from mmai.trials import summarize_trials


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


def _summarize_patient_input_for_regression() -> pd.DataFrame:
    patient_input = _load_patient_input()[["patient_id", "patient_long_text"]].copy()
    summaries, _ = cast(
        tuple[pd.DataFrame, dict],
        summarize_from_relevant_sentences(patient_input, return_qc=False),
    )
    return summaries


def _summarize_trial_input_for_regression() -> pd.DataFrame:
    trial_input = _load_trial_input()[
        ["trial_id", "trial_title", "brief_summary", "eligibility_criteria"]
    ].copy()
    return cast(pd.DataFrame, summarize_trials(trial_input, return_qc=False))

"""Patient postprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def parse_boilerplate(
    df: pd.DataFrame, reasoning_marker: str, boilerplate_marker: str
) -> pd.DataFrame:
    """Split LLM output into summary and boilerplate portions."""
    df = df.copy()
    df["cleaned_patient_summary"] = (
        df["original_patient_summary"]
        .str.split(reasoning_marker, n=1)
        .apply(lambda parts: parts[-1])
        .str.strip()
    )

    split_df = df["cleaned_patient_summary"].str.split(
        boilerplate_marker, n=1, expand=True
    )
    df["cancer_history_summary"] = split_df[0].str.strip()
    df["general_exclusion_criteria_evidence"] = split_df[1].str.strip()
    df["general_exclusion_criteria_evidence"] = df[
        "general_exclusion_criteria_evidence"
    ].fillna(df["cancer_history_summary"])
    return df


def clean_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with empty or obviously invalid patient summaries."""
    df = df.copy()
    df = df.dropna(subset=["cancer_history_summary"])
    df = df[
        ~df.cancer_history_summary.str.contains(
            r"No cancer|no cancer|No primary|No evidence of malignancy",
            case=False,
            na=False,
        )
    ]
    df = df[~df.cancer_history_summary.str.startswith("No information")]
    return df


def postprocess_patient_summaries(df: pd.DataFrame, config: MMAIConfig) -> pd.DataFrame:
    """Postprocess patient summaries into clean outputs."""
    patient_config = dict(config.patient)
    reasoning_marker = patient_config["reasoning_marker"]
    boilerplate_marker = patient_config["boilerplate_marker"]
    df = parse_boilerplate(df, reasoning_marker, boilerplate_marker)
    df = clean_bad_data(df)
    if not config.debug_mode:
        df = df.drop(
            columns=[
                "patient_long_text",
                "original_patient_summary",
                "cleaned_patient_summary",
            ],
            errors="ignore",
        )
    return df

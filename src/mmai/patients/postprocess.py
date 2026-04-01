"""Patient postprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from mmai._qc.patients import build_qc_artifact

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def parse_boilerplate(
    df: pd.DataFrame, reasoning_marker: str, boilerplate_marker: str
) -> pd.DataFrame:
    """Split final patient summary output into summary and boilerplate portions."""
    df = df.copy()
    df["cleaned_patient_summary"] = (
        df["original_patient_summary"]
        .str.split(reasoning_marker, n=1)
        .apply(lambda parts: parts[-1])
        .str.strip()
    )

    split_df = df["cleaned_patient_summary"].str.split(
        boilerplate_marker, n=1, expand=True, regex=True
    )
    df["cancer_history_summary"] = split_df[0].str.strip()
    df["general_exclusion_criteria_evidence"] = split_df[1].str.strip()
    df["general_exclusion_criteria_evidence"] = df[
        "general_exclusion_criteria_evidence"
    ].fillna(df["cancer_history_summary"])
    return df


def clean_bad_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove rows with empty/non-informative summaries and return QC artifact data."""
    source = df.copy()
    source = source.dropna(subset=["cancer_history_summary"])
    source = source[
        ~source.cancer_history_summary.str.contains(
            r"No cancer|no cancer|No primary|No evidence of malignancy",
            case=False,
            na=False,
        )
    ]
    cleaned = source[~source.cancer_history_summary.str.startswith("No information")]
    # record information for QC report
    source_ids = source.get("patient_id", pd.Series(dtype=object)).astype(str)
    cleaned_ids = cleaned.get("patient_id", pd.Series(dtype=object)).astype(str)
    dropped_ids = sorted(set(source_ids) - set(cleaned_ids))
    artifact = build_qc_artifact(
        metric="patients_dropped_noninformative_summary",
        ids=dropped_ids,
        denominator=int(source_ids.nunique()),
    )
    return cleaned, artifact


def postprocess_patient_summaries(
    df: pd.DataFrame,
    config: MMAIConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Postprocess final serial patient summaries into clean outputs."""
    patient_config = dict(config.patient)
    reasoning_marker = patient_config["reasoning_marker"]
    boilerplate_marker = patient_config["boilerplate_marker"]
    parsed = parse_boilerplate(df, reasoning_marker, boilerplate_marker)
    cleaned, qc_artifact = clean_bad_data(parsed)
    if not config.debug_mode:
        cleaned = cleaned.drop(
            columns=[
                "original_patient_summary",
                "cleaned_patient_summary",
            ],
            errors="ignore",
        )
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned, qc_artifact

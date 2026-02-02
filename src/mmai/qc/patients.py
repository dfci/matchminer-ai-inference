"""QC helpers for patient summarization outputs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_KEYWORDS = [
    "Age",
    "Sex",
    "Cancer type",
    "Histology",
    "Current extent",
    "Biomarkers",
    "Treatment history",
]


def _as_list(value: Iterable[str]) -> list[str]:
    return list(value)


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def patient_qc_report(
    patient_summaries: pd.DataFrame,
    *,
    patient_inputs: pd.DataFrame,
    tagged_patients: pd.DataFrame | None = None,
    expected_keywords: list[str] | None = None,
    max_summary_length: int = 4000,
) -> pd.DataFrame:
    """
    Build a QC report for patient summarization outputs.

    Parameters
    ----------
    patient_summaries : pd.DataFrame
        Output from summarize_patients, one row per patient.
        Required columns: patient_id, cancer_history_summary,
        general_exclusion_criteria_evidence.
    patient_inputs : pd.DataFrame
        Original patient notes table with patient_id column. Used to detect
        patients with zero summaries.
    tagged_patients : pd.DataFrame, optional
        Output from extract_relevant_sentences, used to identify
        patients with no tagged notes (empty patient_long_text).
    expected_keywords : list[str], optional
        Keywords expected in each cancer_history_summary.
    max_summary_length : int
        Maximum allowed length (characters) before flagging a summary.

    Returns
    -------
    pd.DataFrame
        A QC report with metric name, value, and optional ids.
    """
    expected_keywords = expected_keywords or _as_list(DEFAULT_KEYWORDS)

    summaries = patient_summaries.copy()
    required = [
        "patient_id",
        "cancer_history_summary",
        "general_exclusion_criteria_evidence",
    ]
    missing = [col for col in required if col not in summaries.columns]
    if missing:
        raise ValueError(f"patient_summaries is missing columns: {', '.join(missing)}")

    summaries["cancer_history_summary"] = _normalize_series(
        summaries["cancer_history_summary"]
    )
    summaries["general_exclusion_criteria_evidence"] = _normalize_series(
        summaries["general_exclusion_criteria_evidence"]
    )

    metrics: list[dict[str, object]] = []
    if "patient_id" not in patient_inputs.columns:
        raise ValueError("patient_inputs must include patient_id")
    total_patients = int(patient_inputs["patient_id"].nunique())

    # Patients with no tagged notes.
    no_tagged_ids: set[str] | None = None
    if tagged_patients is not None:
        if "patient_id" not in tagged_patients.columns:
            raise ValueError("tagged_patients must include patient_id")
        if "patient_long_text" not in tagged_patients.columns:
            raise ValueError("tagged_patients must include patient_long_text")
        tagged = tagged_patients.copy()
        tagged["patient_long_text"] = _normalize_series(tagged["patient_long_text"])
        no_tagged_ids = set(
            tagged.loc[tagged["patient_long_text"].str.strip() == "", "patient_id"]
        )
    metrics.append(
        {
            "metric": "patients_with_no_tagged_notes",
            "value": len(no_tagged_ids) if no_tagged_ids is not None else None,
            "percent": (
                (len(no_tagged_ids) / total_patients * 100)
                if total_patients and no_tagged_ids is not None
                else None
            ),
            "ids": sorted(no_tagged_ids) if no_tagged_ids is not None else None,
        }
    )

    # Patients missing summaries (blank or NA).
    missing_summary_ids = set(
        summaries.loc[
            summaries["cancer_history_summary"].str.strip() == "", "patient_id"
        ]
    )
    input_ids = set(patient_inputs["patient_id"].astype(str))
    output_ids = set(summaries["patient_id"].astype(str))
    missing_summary_ids.update(input_ids - output_ids)
    metrics.append(
        {
            "metric": "patients_missing_summaries",
            "value": len(missing_summary_ids),
            "percent": (len(missing_summary_ids) / total_patients * 100)
            if total_patients
            else 0.0,
            "ids": sorted(missing_summary_ids),
        }
    )

    # Summary equals boilerplate exclusions.
    same_text = summaries.loc[
        summaries["cancer_history_summary"].str.strip()
        == summaries["general_exclusion_criteria_evidence"].str.strip(),
        "patient_id",
    ]
    metrics.append(
        {
            "metric": "patients_summary_equals_boilerplate",
            "value": int(same_text.nunique()),
            "percent": (int(same_text.nunique()) / total_patients * 100)
            if total_patients
            else 0.0,
            "ids": sorted(same_text.astype(str).unique().tolist()),
        }
    )

    # Missing expected keywords (per keyword).
    for keyword in expected_keywords:
        missing_patients = summaries.loc[
            ~summaries["cancer_history_summary"].str.contains(keyword, regex=False),
            "patient_id",
        ]
        metrics.append(
            {
                "metric": f"patients_missing_keyword:{keyword}",
                "value": int(missing_patients.nunique()),
                "percent": (int(missing_patients.nunique()) / total_patients * 100)
                if total_patients
                else 0.0,
                "ids": sorted(missing_patients.astype(str).unique().tolist()),
            }
        )

    # Excessive length summaries.
    excessive = summaries[
        summaries["cancer_history_summary"].str.len() > max_summary_length
    ]
    metrics.append(
        {
            "metric": "patient_summaries_excessive_length",
            "value": len(excessive),
            "percent": (len(excessive) / total_patients * 100)
            if total_patients
            else 0.0,
            "ids": sorted(excessive["patient_id"].astype(str).tolist()),
        }
    )

    return pd.DataFrame(metrics)

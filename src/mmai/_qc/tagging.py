"""Internal QC helpers for patient tagging outputs."""

from __future__ import annotations

import pandas as pd


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def tagger_qc_report(
    tagged_notes: pd.DataFrame,
    *,
    patient_note_source: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a QC report for patient tagging outputs.

    Parameters
    ----------
    tagged_notes : pd.DataFrame
        Output from extract_relevant_sentences, one row per patient.
        Required columns: patient_id, patient_long_text.
    patient_note_source : pd.DataFrame
        Original patient notes table with patient_id column. Used to compute totals.

    Returns
    -------
    pd.DataFrame
        A QC report with metric name, value, and optional ids.
    """
    if "patient_id" not in patient_note_source.columns:
        raise ValueError("patient_note_source must include patient_id")
    if "patient_id" not in tagged_notes.columns:
        raise ValueError("tagged_notes must include patient_id")
    if "patient_long_text" not in tagged_notes.columns:
        raise ValueError("tagged_notes must include patient_long_text")

    tagged = tagged_notes.copy()
    tagged["patient_long_text"] = _normalize_series(tagged["patient_long_text"])
    no_tagged_ids = set(
        tagged.loc[tagged["patient_long_text"].str.strip() == "", "patient_id"]
    )

    total_patients = int(patient_note_source["patient_id"].nunique())
    metrics = [
        {
            "metric": "patients_with_no_tagged_notes",
            "value": len(no_tagged_ids),
            "percent": (len(no_tagged_ids) / total_patients * 100)
            if total_patients
            else 0.0,
            "ids": sorted(no_tagged_ids),
        }
    ]
    return pd.DataFrame(metrics)

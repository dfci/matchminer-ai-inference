"""Shared QC helpers used by patient and trial QC reports."""

from __future__ import annotations

import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize missing values and enforce string dtype for text metrics."""
    return series.fillna("").astype(str)


def build_qc_artifact(
    *,
    metric: str,
    ids: list[str],
    denominator: int,
    numerator: int | None = None,
) -> dict[str, object]:
    """Build a standardized QC artifact dictionary."""
    normalized_ids = sorted({str(item) for item in ids})
    value = int(numerator) if numerator is not None else len(normalized_ids)
    return {
        "metric": metric,
        "numerator": value,
        "denominator": int(denominator),
        "ids": normalized_ids,
    }


def qc_artifact_to_report_row(artifact: dict[str, object]) -> dict[str, object]:
    """Convert a standardized QC artifact into a report row."""
    metric_obj = artifact.get("metric", "")
    metric = str(metric_obj) if metric_obj is not None else ""
    ids_obj = artifact.get("ids", [])
    ids = sorted(str(item) for item in ids_obj) if isinstance(ids_obj, list) else []
    numerator_obj = artifact.get("numerator", len(ids))
    numerator = (
        int(numerator_obj) if isinstance(numerator_obj, (int, float)) else len(ids)
    )
    denominator_obj = artifact.get("denominator", 0)
    denominator = (
        int(denominator_obj) if isinstance(denominator_obj, (int, float)) else 0
    )
    return {
        "metric": metric,
        "value": numerator,
        "denominator": denominator,
        "percent": (numerator / denominator * 100) if denominator else 0.0,
        "ids": ids,
    }

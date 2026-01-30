"""QC helpers for trial summarization outputs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_KEYWORDS = [
    "Age",
    "Sex",
    "Cancer type allowed",
    "Histology allowed",
    "Cancer burden allowed",
    "Prior treatment required",
    "Prior treatment excluded",
    "Biomarkers required",
    "Biomarkers excluded",
]


def _as_list(value: Iterable[str]) -> list[str]:
    return list(value)


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def trial_qc_report(
    trial_spaces: pd.DataFrame,
    *,
    trial_inputs: pd.DataFrame | None = None,
    expected_keywords: list[str] | None = None,
    max_space_length: int = 2000,
) -> pd.DataFrame:
    """
    Build a QC report for trial summarization outputs.

    Parameters
    ----------
    trial_spaces : pd.DataFrame
        Output from summarize_trials, one row per trial space.
        Required columns: trial_id, clinical_space_number,
        clinical_space_summary, general_exclusion_criteria.
    trial_inputs : pd.DataFrame, optional
        Original trial input table with trial_id column. If provided,
        used to detect trials with zero spaces.
    expected_keywords : list[str], optional
        Keywords expected in each clinical_space_summary.
    max_space_length : int
        Maximum allowed length (characters) before flagging a space.

    Returns
    -------
    pd.DataFrame
        A QC report with metric name, value, and optional ids.
    """
    expected_keywords = expected_keywords or _as_list(DEFAULT_KEYWORDS)

    spaces = trial_spaces.copy()
    required = [
        "trial_id",
        "clinical_space_number",
        "clinical_space_summary",
        "general_exclusion_criteria",
    ]
    missing = [col for col in required if col not in spaces.columns]
    if missing:
        raise ValueError(f"trial_spaces is missing columns: {', '.join(missing)}")

    spaces["clinical_space_summary"] = _normalize_series(
        spaces["clinical_space_summary"]
    )
    spaces["general_exclusion_criteria"] = _normalize_series(
        spaces["general_exclusion_criteria"]
    )

    metrics: list[dict[str, object]] = []

    # Trials missing summaries (no rows) or blank summaries.
    missing_summary_ids: set[str] = set()
    blank_summary_ids = set(
        spaces.loc[spaces["clinical_space_summary"].str.strip() == "", "trial_id"]
    )
    missing_summary_ids.update(blank_summary_ids)
    if trial_inputs is not None and "trial_id" in trial_inputs.columns:
        input_ids = set(trial_inputs["trial_id"].astype(str))
        output_ids = set(spaces["trial_id"].astype(str))
        missing_summary_ids.update(input_ids - output_ids)
    metrics.append(
        {
            "metric": "trials_missing_summaries",
            "value": len(missing_summary_ids),
            "ids": sorted(missing_summary_ids),
        }
    )

    # Spaces per trial.
    spaces_per_trial = spaces.groupby("trial_id").size()
    metrics.extend(
        [
            {
                "metric": "spaces_per_trial_min",
                "value": int(spaces_per_trial.min()) if len(spaces_per_trial) else 0,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_median",
                "value": float(spaces_per_trial.median())
                if len(spaces_per_trial)
                else 0.0,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_max",
                "value": int(spaces_per_trial.max()) if len(spaces_per_trial) else 0,
                "ids": [],
            },
        ]
    )

    # Non-distinct spaces: duplicate space numbers (esp "1") or duplicate text.
    dup_number_ids = set(
        spaces.loc[
            spaces.duplicated(subset=["trial_id", "clinical_space_number"], keep=False),
            "trial_id",
        ]
    )
    dup_text_ids = set(
        spaces.loc[
            spaces.duplicated(
                subset=["trial_id", "clinical_space_summary"], keep=False
            ),
            "trial_id",
        ]
    )
    non_distinct_ids = sorted(set(dup_number_ids) | set(dup_text_ids))
    metrics.append(
        {
            "metric": "trials_with_non_distinct_spaces",
            "value": len(non_distinct_ids),
            "ids": non_distinct_ids,
        }
    )

    # Missing expected keywords.
    keyword_missing_ids: list[str] = []
    for _, row in spaces.iterrows():
        summary = row["clinical_space_summary"]
        missing_keywords = [
            keyword for keyword in expected_keywords if keyword not in summary
        ]
        if missing_keywords:
            keyword_missing_ids.append(
                row["space_trial_id"] if "space_trial_id" in row else row["trial_id"]
            )
    metrics.append(
        {
            "metric": "spaces_missing_expected_keywords",
            "value": len(keyword_missing_ids),
            "ids": sorted(set(keyword_missing_ids)),
        }
    )

    # Missing boilerplate exclusions.
    boilerplate_missing = spaces[
        spaces["general_exclusion_criteria"].str.strip().isin(["", "None", "none"])
    ]
    metrics.append(
        {
            "metric": "trials_missing_boilerplate_exclusions",
            "value": boilerplate_missing["trial_id"].nunique(),
            "ids": sorted(boilerplate_missing["trial_id"].unique().tolist()),
        }
    )

    # Excessive length spaces.
    excessive = spaces[spaces["clinical_space_summary"].str.len() > max_space_length]
    metrics.append(
        {
            "metric": "spaces_excessive_length",
            "value": len(excessive),
            "ids": sorted(
                excessive.get("space_trial_id", excessive["trial_id"])
                .astype(str)
                .tolist()
            ),
        }
    )

    return pd.DataFrame(metrics)

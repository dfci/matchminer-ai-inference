"""Internal QC helpers for trial summarization outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

from mmai.backends import get_backend

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


DEFAULT_KEYWORDS = [
    "Age",
    "Sex",
    "Cancer type",
    "Histology",
    "Cancer burden",
    "Prior treatment required",
    "Prior treatment excluded",
    "Biomarkers required",
    "Biomarkers excluded",
]


def _as_list(value: Iterable[str]) -> list[str]:
    return list(value)


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _ensure_space_trial_id(spaces: pd.DataFrame) -> pd.DataFrame:
    if "space_trial_id" in spaces.columns:
        return spaces
    if {"trial_id", "clinical_space_number"}.issubset(spaces.columns):
        spaces = spaces.copy()
        spaces["space_trial_id"] = (
            spaces["trial_id"].astype(str)
            + "-"
            + spaces["clinical_space_number"].astype(str)
        )
    return spaces


def trial_qc_report(
    trial_spaces: pd.DataFrame,
    *,
    trial_source: pd.DataFrame,
    unfiltered_spaces: pd.DataFrame,
    finish_reasons: pd.Series | list[str] | None,
    config: MMAIConfig | None = None,
    max_embedding_input_tokens: int = 2500,
    expected_keywords: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a QC report for trial summarization outputs.

    Parameters
    ----------
    trial_spaces : pd.DataFrame
        Output from summarize_trials, one row per trial space.
        Required columns: trial_id, clinical_space_number,
        clinical_space_summary, general_exclusion_criteria.
    trial_source : pd.DataFrame
        Original trial input table with trial_id column. Used to detect
        trials with zero spaces.
    unfiltered_spaces : pd.DataFrame
        Trial-space table prior to keyword filtering, used for keyword checks.
    finish_reasons : pd.Series | list[str] | None
        Finish reasons from trial-level generation. Expected to be indexed
        by trial_id so truncated-response IDs map to trials.
    config : MMAIConfig | None, optional
        Config used to resolve backend and embedding settings when token counts
        are computed inside this QC function.
    max_embedding_input_tokens : int
        Maximum token length accepted by the embedding model.
    expected_keywords : list[str], optional
        Keywords expected in each clinical_space_summary.

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
    if "trial_id" not in trial_source.columns:
        raise ValueError("trial_source must include trial_id")

    spaces["clinical_space_summary"] = _normalize_series(
        spaces["clinical_space_summary"]
    )
    spaces["general_exclusion_criteria"] = _normalize_series(
        spaces["general_exclusion_criteria"]
    )
    spaces = _ensure_space_trial_id(spaces)
    if finish_reasons is None:
        raise ValueError("finish_reasons is required for trial_qc_report")
    finish_series = _normalize_series(pd.Series(finish_reasons))
    total_generated = int(len(finish_series))

    metrics: list[dict[str, object]] = []
    total_trials = int(trial_source["trial_id"].nunique())
    total_spaces = int(len(spaces))

    # Trials missing summaries (no rows) or blank summaries.
    missing_summary_ids: set[str] = set()
    blank_summary_ids = set(
        spaces.loc[spaces["clinical_space_summary"].str.strip() == "", "trial_id"]
    )
    missing_summary_ids.update(blank_summary_ids)
    input_ids = set(trial_source["trial_id"].astype(str))
    output_ids = set(spaces["trial_id"].astype(str))
    missing_summary_ids.update(input_ids - output_ids)
    metrics.append(
        {
            "metric": "trials_missing_in_output",
            "value": len(missing_summary_ids),
            "percent": (len(missing_summary_ids) / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": sorted(missing_summary_ids),
        }
    )

    length_ids = finish_series[finish_series == "length"].index.to_series()
    metrics.append(
        {
            "metric": "trials_truncated_llm_response",
            "value": int(length_ids.nunique()),
            "percent": (int(length_ids.nunique()) / total_generated * 100)
            if total_generated
            else 0.0,
            "ids": sorted(length_ids.astype(str).unique().tolist()),
        }
    )

    if config is not None and config.embedding:
        backend = get_backend(config.backend)
        embedding_config = dict(config.embedding)
        token_series = pd.Series(
            backend.count_embedding_tokens(
                spaces["clinical_space_summary"].fillna("").astype(str).tolist(),
                embedding_config=embedding_config,
            ),
            index=spaces["space_trial_id"].astype(str).tolist(),
        )
        token_series = pd.to_numeric(token_series, errors="coerce").fillna(0)
        over_limit_ids = token_series[token_series > max_embedding_input_tokens].index
        metrics.append(
            {
                "metric": "spaces_exceed_embedding_token_limit",
                "value": int(over_limit_ids.nunique()),
                "percent": (int(over_limit_ids.nunique()) / total_spaces * 100)
                if total_spaces
                else 0.0,
                "ids": sorted(over_limit_ids.astype(str).unique().tolist()),
            }
        )

    # Spaces per trial.
    spaces_per_trial = spaces.groupby("trial_id").size()
    metrics.extend(
        [
            {
                "metric": "spaces_per_trial_min",
                "value": int(spaces_per_trial.min()) if len(spaces_per_trial) else 0,
                "percent": None,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_median",
                "value": float(spaces_per_trial.median())
                if len(spaces_per_trial)
                else 0.0,
                "percent": None,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_max",
                "value": int(spaces_per_trial.max()) if len(spaces_per_trial) else 0,
                "percent": None,
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
            "percent": (len(non_distinct_ids) / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": non_distinct_ids,
        }
    )

    # Missing expected keywords (per keyword).
    keyword_spaces = _ensure_space_trial_id(unfiltered_spaces.copy())
    if "clinical_space_summary" not in keyword_spaces.columns:
        raise ValueError("keyword_source must include clinical_space_summary")
    keyword_spaces["clinical_space_summary"] = _normalize_series(
        keyword_spaces["clinical_space_summary"]
    )
    for keyword in expected_keywords:
        missing_spaces = keyword_spaces.loc[
            ~keyword_spaces["clinical_space_summary"].str.contains(keyword, regex=False)
        ]
        ids = (
            missing_spaces["space_trial_id"]
            if "space_trial_id" in missing_spaces.columns
            else missing_spaces["trial_id"]
        )
        metrics.append(
            {
                "metric": f"spaces_dropped_missing_keyword:{keyword}",
                "value": len(missing_spaces),
                "percent": (len(missing_spaces) / total_spaces * 100)
                if total_spaces
                else 0.0,
                "ids": sorted(ids.astype(str).tolist()),
            }
        )

    # Missing boilerplate exclusions.
    boilerplate_missing = spaces[
        spaces["general_exclusion_criteria"].str.strip().isin(["", "None", "none"])
    ]
    metrics.append(
        {
            "metric": "trials_exclusion_criteria_not_extracted",
            "value": boilerplate_missing["trial_id"].nunique(),
            "percent": (boilerplate_missing["trial_id"].nunique() / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": sorted(boilerplate_missing["trial_id"].unique().tolist()),
        }
    )

    return pd.DataFrame(metrics)

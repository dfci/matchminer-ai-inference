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


def trial_qc_report(
    trial_spaces: pd.DataFrame,
    *,
    trial_source: pd.DataFrame,
    unfiltered_spaces: pd.DataFrame,
    truncated_llm_qc_artifact: dict[str, object],
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
    truncated_llm_qc_artifact : dict[str, object]
        QC artifact for trials truncated due to finish_reason='length' with
        metric, numerator, denominator, and ids.
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
        "space_trial_id",
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
            "denominator": total_trials,
            "percent": (len(missing_summary_ids) / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": sorted(missing_summary_ids),
        }
    )

    metrics.append(qc_artifact_to_report_row(truncated_llm_qc_artifact))

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
                "denominator": total_spaces,
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
                "denominator": None,
                "percent": None,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_median",
                "value": float(spaces_per_trial.median())
                if len(spaces_per_trial)
                else 0.0,
                "denominator": None,
                "percent": None,
                "ids": [],
            },
            {
                "metric": "spaces_per_trial_max",
                "value": int(spaces_per_trial.max()) if len(spaces_per_trial) else 0,
                "denominator": None,
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
            "denominator": total_trials,
            "percent": (len(non_distinct_ids) / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": non_distinct_ids,
        }
    )

    # Missing expected keywords (per keyword).
    keyword_spaces = unfiltered_spaces.copy()
    if "clinical_space_summary" not in keyword_spaces.columns:
        raise ValueError("unfiltered_spaces must include clinical_space_summary")
    if "space_trial_id" not in keyword_spaces.columns:
        raise ValueError("unfiltered_spaces must include space_trial_id")
    keyword_spaces["clinical_space_summary"] = _normalize_series(
        keyword_spaces["clinical_space_summary"]
    )
    for keyword in expected_keywords:
        missing_spaces = keyword_spaces.loc[
            ~keyword_spaces["clinical_space_summary"].str.contains(keyword, regex=False)
        ]
        metrics.append(
            {
                "metric": f"spaces_dropped_missing_keyword:{keyword}",
                "value": len(missing_spaces),
                "denominator": total_spaces,
                "percent": (len(missing_spaces) / total_spaces * 100)
                if total_spaces
                else 0.0,
                "ids": sorted(missing_spaces["space_trial_id"].astype(str).tolist()),
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
            "denominator": total_trials,
            "percent": (boilerplate_missing["trial_id"].nunique() / total_trials * 100)
            if total_trials
            else 0.0,
            "ids": sorted(boilerplate_missing["trial_id"].unique().tolist()),
        }
    )

    return pd.DataFrame(metrics)

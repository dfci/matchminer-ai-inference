"""Patient postprocessing helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


logger = logging.getLogger(__name__)


def _split_boilerplate_section(text: str, boilerplate_marker: str) -> tuple[str, str]:
    """Split generated text at the line containing the boilerplate marker."""
    lines = text.splitlines()
    split_idx = next(
        (idx for idx, line in enumerate(lines) if boilerplate_marker in line),
        -1,
    )
    if split_idx == -1:
        cleaned = text.strip()
        return cleaned, cleaned or "None"

    main_part = "\n".join(lines[:split_idx]).strip()
    boilerplate_part = "\n".join(lines[split_idx + 1 :]).strip() or "None"
    return main_part, boilerplate_part


def parse_boilerplate(df: pd.DataFrame, boilerplate_marker: str) -> pd.DataFrame:
    """Split final patient summary output into summary and boilerplate portions."""
    df = df.copy()
    summary_source = df["patient_answer_text"].fillna("").astype(str)
    cleaned_summary = summary_source.str.strip()
    split_parts = cleaned_summary.apply(
        lambda text: _split_boilerplate_section(str(text), boilerplate_marker)
    )
    df["cancer_history_summary"] = split_parts.apply(lambda parts: parts[0])
    df["general_exclusion_criteria_evidence"] = split_parts.apply(
        lambda parts: parts[1]
    )
    return df


def clean_bad_data(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Remove empty or non-informative summaries and return their patient IDs."""
    cleaned = df.copy()
    initial_patient_ids = set(cleaned["patient_id"].astype(str))
    cleaned["cancer_history_summary"] = cleaned["cancer_history_summary"].fillna("")

    cleaned = cleaned[cleaned["cancer_history_summary"] != ""]
    cleaned = cleaned[
        ~cleaned["cancer_history_summary"].str.startswith("No information")
    ]

    retained_patient_ids = set(cleaned["patient_id"].astype(str))
    removed_patient_ids = initial_patient_ids - retained_patient_ids
    logger.info(
        "Filtered %d patient summary row(s) as non-informative.",
        len(removed_patient_ids),
    )
    return cleaned, removed_patient_ids


def postprocess_patient_summaries(
    df: pd.DataFrame,
    config: MMAIConfig,
) -> tuple[pd.DataFrame, set[str]]:
    """Parse patient summaries, remove non-informative rows, and return their IDs."""
    patient_config = dict(config.patient)
    boilerplate_marker = patient_config["boilerplate_marker"]
    parsed = parse_boilerplate(df, boilerplate_marker)
    cleaned, removed_patient_ids = clean_bad_data(parsed)
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned, removed_patient_ids

"""Patient postprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


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


def postprocess_patient_summaries(
    df: pd.DataFrame,
    config: MMAIConfig,
) -> pd.DataFrame:
    """Postprocess final serial patient summaries into clean outputs."""
    patient_config = dict(config.patient)
    boilerplate_marker = patient_config["boilerplate_marker"]
    cleaned = parse_boilerplate(df, boilerplate_marker).copy()
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned

"""Patient postprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from matchminer_ai._qc.patients import build_qc_artifact

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
    boilerplate_marker = patient_config["boilerplate_marker"]
    parsed = parse_boilerplate(df, boilerplate_marker)
    cleaned, qc_artifact = clean_bad_data(parsed)
    cleaned = cleaned.copy()
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned, qc_artifact

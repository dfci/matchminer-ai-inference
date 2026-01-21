"""Patient tagging helpers."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import pandas as pd

from mmai.backends import get_backend
from mmai.config import MMAIConfig, load_default_preset

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def split_sentences_and_dedup_notes(
    note_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Split reports into sentence chunks and remove duplicates."""
    chunk_frames: list[pd.DataFrame] = []
    for i in range(0, note_rows.shape[0]):
        rpt_text: str = re.sub("\n|\r", " ", note_rows.iloc[i]["note_text"].strip())
        rpt_text = re.sub(r"\s+", " ", rpt_text)
        rpt_text = re.sub("\\. ", "<excerpt break>", rpt_text)
        chunks = pd.Series(rpt_text.split("<excerpt break>")).str.strip()
        chunks = chunks[chunks != ""]

        if len(chunks):
            chunk_frame = pd.DataFrame(
                {
                    "note_date": note_rows.iloc[i]["note_date"],
                    "note_type": note_rows.iloc[i]["note_type"],
                    "excerpt": chunks,
                }
            )
            chunk_frames.append(chunk_frame)

    if len(chunk_frames):
        return pd.concat(chunk_frames, axis=0).drop_duplicates(
            subset=["excerpt"], keep="first"
        )
    return pd.DataFrame(columns=["note_date", "note_type", "excerpt"])


def _format_relevant_text(
    patient_id: str,
    excerpts_frame: pd.DataFrame,
) -> pd.Series:
    excerpts_frame = (
        excerpts_frame.groupby(["note_date", "note_type"])["excerpt"]
        .agg(". ".join)
        .reset_index()
    )
    excerpts_frame = excerpts_frame[excerpts_frame["excerpt"].fillna("") != ""]
    excerpts_frame["date_text"] = (
        excerpts_frame["note_date"].astype(str)
        + " "
        + excerpts_frame["note_type"]
        + " "
        + excerpts_frame["excerpt"]
    )

    return pd.Series(
        {
            "patient_id": patient_id,
            "patient_long_text": "\n".join(excerpts_frame["date_text"].tolist()),
        }
    )


def _extract_relevant_text_from_notes(
    patient_id: str,
    excerpts_frame: pd.DataFrame,
    backend: Any,
    *,
    tagger_config: dict,
) -> tuple[pd.Series, dict[str, Any]]:
    negative_tag_cutoff = float(tagger_config["negative_tag_cutoff"])
    positive_tag_cutoff = float(tagger_config["positive_tag_cutoff"])
    predictions, model_metadata = backend.tag_excerpts(
        excerpts_frame["excerpt"].tolist(),
        tagger_config=tagger_config,
    )
    predictions_frame = pd.DataFrame(predictions)
    excerpts_frame = excerpts_frame.copy()
    excerpts_frame.loc[:, ["label", "score"]] = predictions_frame[["label", "score"]]

    negative_condition = (excerpts_frame["label"] == "NEGATIVE") & (
        excerpts_frame["score"] < negative_tag_cutoff
    )
    positive_condition = (excerpts_frame["label"] == "POSITIVE") & (
        excerpts_frame["score"] > positive_tag_cutoff
    )

    excerpts_frame = excerpts_frame[negative_condition | positive_condition].copy()

    return (
        _format_relevant_text(
            patient_id=patient_id,
            excerpts_frame=excerpts_frame,
        ),
        model_metadata,
    )


def extract_relevant_text_from_patient(
    note_rows: pd.DataFrame,
    backend: Any,
    *,
    tagger_config: dict,
) -> tuple[pd.Series, dict[str, Any]]:
    """Extract relevant snippets for a single patient."""
    note_rows = note_rows.copy()
    note_rows["note_date"] = pd.to_datetime(note_rows["note_date"])
    note_rows = note_rows.sort_values(by=["note_date", "note_text"]).reset_index()

    excerpts_frame = split_sentences_and_dedup_notes(
        note_rows,
    )
    if len(excerpts_frame) > 0:
        return _extract_relevant_text_from_notes(
            patient_id=note_rows["patient_id"].iloc[0],
            excerpts_frame=excerpts_frame,
            backend=backend,
            tagger_config=tagger_config,
        )
    return (
        pd.Series(
            {
                "patient_id": note_rows["patient_id"].iloc[0],
                "patient_long_text": "",
            }
        ),
        {"model_name": tagger_config.get("model_name")},
    )


def extract_relevant_sentences(
    df: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Extract relevant sentences from longitudinal patient notes.

    Parameters
    ----------
    df : pd.DataFrame
        Note-level input. One row per note.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        note_text : str
            Full note text.
        note_type : str
            Type of note (clinical_note, pathology_report, etc.).
        note_date : str or datetime
            Date of the note.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Patient-level DataFrame (one row per patient) and metadata
        about the tagger model and configuration.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    tagger_config = dict(patient_config["tagger"])
    backend = get_backend(resolved_config.backend)
    df = df.copy()
    df = df[df["note_text"].notna()]
    df.loc[:, "note_text"] = df["note_text"].astype(str)
    df.loc[:, "note_date"] = pd.to_datetime(df["note_date"])

    grouped = df.groupby("patient_id").apply(
        extract_relevant_text_from_patient,
        backend=backend,
        tagger_config=tagger_config,
    )
    result_df = grouped.apply(lambda item: item[0]).reset_index(drop=True)
    metadata = {
        "config_snapshot": {"patient": patient_config},
        "model_metadata": grouped.iloc[0][1] if len(grouped) else {},
    }
    return result_df, metadata


__all__ = ["extract_relevant_sentences"]

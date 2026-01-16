"""Patient tagging helpers."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import pandas as pd

from mmai.backends import get_backend

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def split_sentences_and_dedup_notes(
    patient_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Split reports into sentence chunks and remove duplicates."""
    chunk_frames: list[pd.DataFrame] = []
    for i in range(0, patient_frame.shape[0]):
        rpt_text: str = re.sub("\n|\r", " ", patient_frame.iloc[i]["note_text"].strip())
        rpt_text = re.sub(r"\s+", " ", rpt_text)
        rpt_text = re.sub("\\. ", "<excerpt break>", rpt_text)
        chunks = pd.Series(rpt_text.split("<excerpt break>")).str.strip()
        chunks = chunks[chunks != ""]

        if len(chunks):
            chunk_frame = pd.DataFrame(
                {
                    "note_date": patient_frame.iloc[i]["note_date"],
                    "note_type": patient_frame.iloc[i]["note_type"],
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
    notes_frame: pd.DataFrame,
) -> pd.Series:
    notes_frame = (
        notes_frame.groupby(["note_date", "note_type"])["excerpt"]
        .agg(". ".join)
        .reset_index()
    )
    notes_frame = notes_frame[notes_frame["excerpt"].fillna("") != ""]
    notes_frame["date_text"] = (
        notes_frame["note_date"].astype(str)
        + " "
        + notes_frame["note_type"]
        + " "
        + notes_frame["excerpt"]
    )

    return pd.Series(
        {
            "patient_id": patient_id,
            "patient_long_text": "\n".join(notes_frame["date_text"].tolist()),
        }
    )


def _extract_relevant_text_from_notes(
    patient_id: str,
    notes_frame: pd.DataFrame,
    backend: Any,
    *,
    tagger_config: dict,
) -> pd.Series:
    negative_tag_cutoff = float(tagger_config["negative_tag_cutoff"])
    positive_tag_cutoff = float(tagger_config["positive_tag_cutoff"])
    predictions = backend.tag_excerpts(
        notes_frame["excerpt"].tolist(),
        tagger_config=tagger_config,
    )
    predictions_frame = pd.DataFrame(predictions)
    notes_frame = notes_frame.copy()
    notes_frame.loc[:, ["label", "score"]] = predictions_frame[["label", "score"]]

    negative_condition = (notes_frame["label"] == "NEGATIVE") & (
        notes_frame["score"] < negative_tag_cutoff
    )
    positive_condition = (notes_frame["label"] == "POSITIVE") & (
        notes_frame["score"] > positive_tag_cutoff
    )

    notes_frame = notes_frame[negative_condition | positive_condition].copy()

    return _format_relevant_text(
        patient_id=patient_id,
        notes_frame=notes_frame,
    )


def extract_relevant_text_from_patient(
    patient_frame_original: pd.DataFrame,
    backend: Any,
    *,
    tagger_config: dict,
) -> pd.Series:
    """Extract relevant snippets for a single patient."""
    patient_frame = patient_frame_original.copy()
    patient_frame["note_date"] = pd.to_datetime(patient_frame["note_date"])
    patient_frame = patient_frame.sort_values(
        by=["note_date", "note_text"]
    ).reset_index()

    notes_frame = split_sentences_and_dedup_notes(
        patient_frame,
    )
    if len(notes_frame) > 0:
        return _extract_relevant_text_from_notes(
            patient_id=patient_frame["patient_id"].iloc[0],
            notes_frame=notes_frame,
            backend=backend,
            tagger_config=tagger_config,
        )
    return pd.Series(
        {
            "patient_id": patient_frame["patient_id"].iloc[0],
            "patient_long_text": "",
        }
    )


def _validate_tagger_config(patient_config: dict) -> dict:
    if "tagger" not in patient_config:
        raise ValueError("patient config is missing required key: tagger")
    tagger_config = dict(patient_config["tagger"])
    required_keys = [
        "model_name",
        "device",
        "positive_tag_cutoff",
        "negative_tag_cutoff",
    ]
    missing = [key for key in required_keys if key not in tagger_config]
    if missing:
        raise ValueError(
            "patient tagger config is missing required keys: " + ", ".join(missing)
        )
    return tagger_config


def extract_relevant_sentences(
    df: pd.DataFrame,
    *,
    config: MMAIConfig,
) -> pd.DataFrame:
    """Extract relevant sentences from patient notes."""
    patient_config = dict(config.patient)
    tagger_config = _validate_tagger_config(patient_config)
    backend = get_backend(config.backend)
    df = df.copy()
    df = df[df["note_text"].notna()]
    df.loc[:, "note_text"] = df["note_text"].astype(str)
    df.loc[:, "note_date"] = pd.to_datetime(df["note_date"])

    result_df = (
        df.groupby("patient_id")
        .apply(
            extract_relevant_text_from_patient,
            backend=backend,
            tagger_config=tagger_config,
        )
        .reset_index(drop=True)
    )
    return result_df


__all__ = ["extract_relevant_sentences"]

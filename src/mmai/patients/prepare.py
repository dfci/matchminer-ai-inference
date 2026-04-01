"""Patient note preparation helpers for serial summarization."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DATE_HEADER_RE = re.compile(r"=== Clinical Note dated (.+?) ===")


def validate_note_inputs(
    notes: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize note-level patient inputs."""
    required_columns = ["patient_id", "note_text", "note_date"]
    missing = [column for column in required_columns if column not in notes.columns]
    if missing:
        raise ValueError(
            "patient note input must include columns "
            "'patient_id', 'note_text', and 'note_date'. Missing: "
            f"{', '.join(missing)}"
        )

    normalized = notes.copy()
    normalized = normalized[normalized["note_text"].notna()].copy()
    normalized["note_text"] = normalized["note_text"].astype(str)
    normalized["note_date"] = pd.to_datetime(normalized["note_date"])
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    return normalized


def deduplicate_patient_notes(
    notes: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Remove duplicate sentences across a patient's notes, keeping first occurrence.
    """
    seen: set[str] = set()
    deduped_notes: list[tuple[str, str]] = []
    for date_str, note_text in notes:
        sentences = _SENTENCE_SPLIT_RE.split(note_text)
        kept_sentences: list[str] = []
        for sentence in sentences:
            normalized = sentence.strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            kept_sentences.append(normalized)
        if kept_sentences:
            deduped_notes.append((date_str, " ".join(kept_sentences)))
    return deduped_notes


def concatenate_and_chunk_notes(
    notes: list[tuple[str, str]],
    tokenizer: Any,
    *,
    chunk_size: int = 10000,
    chunk_overlap: int = 500,
) -> list[tuple[str, str, str]]:
    """
    Concatenate chronologically ordered notes and chunk them by token length.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    if not notes:
        return []

    blocks = [
        f"=== Clinical Note dated {date_str} ===\n{note_text}\n"
        for date_str, note_text in notes
    ]
    full_text = "\n".join(blocks)
    all_dates = [date_str for date_str, _ in notes]
    all_tokens = tokenizer(full_text, add_special_tokens=False).input_ids

    if len(all_tokens) <= chunk_size:
        return [(full_text, all_dates[0], all_dates[-1])]

    chunks: list[tuple[str, str, str]] = []
    stride = chunk_size - chunk_overlap
    start = 0

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        chunk_tokens = all_tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        found_dates = _DATE_HEADER_RE.findall(chunk_text)
        if found_dates:
            first_date = found_dates[0]
            last_date = found_dates[-1]
        elif chunks:
            first_date = chunks[-1][2]
            last_date = first_date
        else:
            first_date = all_dates[0]
            last_date = all_dates[0]

        chunks.append((chunk_text, first_date, last_date))
        if end >= len(all_tokens):
            break
        start += stride

    return chunks


def prepare_patient_notes(
    notes: pd.DataFrame,
    tokenizer: Any,
    *,
    chunk_size: int = 10000,
    chunk_overlap: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert note-level input into patient-level and chunk-level prepared tables.
    """
    normalized = validate_note_inputs(notes)
    normalized = normalized.sort_values(["patient_id", "note_date"]).reset_index(
        drop=True
    )

    patient_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []

    for patient_id, group in normalized.groupby("patient_id", sort=False):
        patient_notes = [
            (
                row["note_date"].date().isoformat(),
                str(row["note_text"]),
            )
            for _, row in group.iterrows()
        ]
        patient_notes = deduplicate_patient_notes(patient_notes)
        last_note_date = patient_notes[-1][0] if patient_notes else ""

        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "last_note_date": last_note_date,
            }
        )

        if not patient_notes:
            continue

        patient_chunks = concatenate_and_chunk_notes(
            patient_notes,
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk_index, (chunk_text, first_date, last_date) in enumerate(
            patient_chunks
        ):
            chunk_rows.append(
                {
                    "patient_id": str(patient_id),
                    "chunk_index": chunk_index,
                    "first_date": first_date,
                    "last_date": last_date,
                    "chunk_text": chunk_text,
                }
            )

    patient_df = pd.DataFrame(
        patient_rows,
        columns=["patient_id", "last_note_date"],
    )
    chunk_df = pd.DataFrame(
        chunk_rows,
        columns=[
            "patient_id",
            "chunk_index",
            "first_date",
            "last_date",
            "chunk_text",
        ],
    )
    return patient_df, chunk_df


__all__ = [
    "concatenate_and_chunk_notes",
    "deduplicate_patient_notes",
    "prepare_patient_notes",
    "validate_note_inputs",
]

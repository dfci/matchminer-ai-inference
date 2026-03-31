"""Patient note preparation helpers for serial summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DATE_HEADER_RE = re.compile(r"=== Clinical Note dated (.+?) ===")


@dataclass(frozen=True)
class PreparedPatientChunk:
    """A single token-bounded note chunk for one patient."""

    patient_id: str
    chunk_index: int
    first_date: str
    last_date: str
    chunk_text: str


@dataclass(frozen=True)
class PreparedPatientNotes:
    """Prepared note chunks and metadata for one patient."""

    patient_id: str
    starting_summary: str | None
    last_note_date: str
    chunks: list[PreparedPatientChunk]


def validate_note_inputs(
    notes: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    note_text_col: str = "note_text",
    note_date_col: str = "note_date",
) -> pd.DataFrame:
    """Validate and normalize note-level patient inputs."""
    required_columns = [patient_id_col, note_text_col, note_date_col]
    missing = [column for column in required_columns if column not in notes.columns]
    if missing:
        raise ValueError(
            "patient note input is missing required columns: " f"{', '.join(missing)}"
        )

    normalized = notes.copy()
    normalized = normalized[normalized[note_text_col].notna()].copy()
    normalized[note_text_col] = normalized[note_text_col].astype(str)
    normalized[note_date_col] = pd.to_datetime(normalized[note_date_col])
    normalized[patient_id_col] = normalized[patient_id_col].astype(str)
    return normalized


def validate_existing_summaries(
    existing_summaries: pd.DataFrame,
    *,
    patient_id_col: str = "patient_id",
    summary_col: str = "patient_summary",
) -> pd.DataFrame:
    """Validate and normalize existing patient summary state."""
    required_columns = [patient_id_col, summary_col]
    missing = [
        column
        for column in required_columns
        if column not in existing_summaries.columns
    ]
    if missing:
        raise ValueError(
            "existing summaries input is missing required columns: "
            f"{', '.join(missing)}"
        )

    normalized = existing_summaries.copy()
    normalized[patient_id_col] = normalized[patient_id_col].astype(str)
    normalized[summary_col] = normalized[summary_col].where(
        normalized[summary_col].notna(),
        None,
    )

    normalized = normalized.drop_duplicates(subset=[patient_id_col], keep="last")
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
    existing_summaries: pd.DataFrame | None = None,
    patient_id_col: str = "patient_id",
    note_text_col: str = "note_text",
    note_date_col: str = "note_date",
    summary_col: str = "patient_summary",
    chunk_size: int = 10000,
    chunk_overlap: int = 500,
) -> list[PreparedPatientNotes]:
    """
    Convert note-level input into chronologically chunked patient summaries.
    """
    normalized = validate_note_inputs(
        notes,
        patient_id_col=patient_id_col,
        note_text_col=note_text_col,
        note_date_col=note_date_col,
    )
    normalized = normalized.sort_values([patient_id_col, note_date_col]).reset_index(
        drop=True
    )
    existing_summary_lookup: dict[str, dict[str, Any]] = {}
    if existing_summaries is not None:
        normalized_existing = validate_existing_summaries(
            existing_summaries,
            patient_id_col=patient_id_col,
            summary_col=summary_col,
        )
        existing_summary_lookup = {
            str(row[patient_id_col]): row.to_dict()
            for _, row in normalized_existing.iterrows()
        }

    prepared_patients: list[PreparedPatientNotes] = []
    for patient_id, group in normalized.groupby(patient_id_col, sort=False):
        existing_summary = existing_summary_lookup.get(str(patient_id), {})
        patient_notes = [
            (
                row[note_date_col].date().isoformat(),
                str(row[note_text_col]),
            )
            for _, row in group.iterrows()
        ]
        patient_notes = deduplicate_patient_notes(patient_notes)
        if not patient_notes:
            prepared_patients.append(
                PreparedPatientNotes(
                    patient_id=str(patient_id),
                    starting_summary=existing_summary.get(summary_col),
                    last_note_date="",
                    chunks=[],
                )
            )
            continue

        patient_chunks = concatenate_and_chunk_notes(
            patient_notes,
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        prepared_patients.append(
            PreparedPatientNotes(
                patient_id=str(patient_id),
                starting_summary=existing_summary.get(summary_col),
                last_note_date=patient_notes[-1][0],
                chunks=[
                    PreparedPatientChunk(
                        patient_id=str(patient_id),
                        chunk_index=chunk_index,
                        first_date=first_date,
                        last_date=last_date,
                        chunk_text=chunk_text,
                    )
                    for chunk_index, (chunk_text, first_date, last_date) in enumerate(
                        patient_chunks
                    )
                ],
            )
        )

    return prepared_patients


__all__ = [
    "PreparedPatientChunk",
    "PreparedPatientNotes",
    "concatenate_and_chunk_notes",
    "deduplicate_patient_notes",
    "prepare_patient_notes",
    "validate_existing_summaries",
    "validate_note_inputs",
]

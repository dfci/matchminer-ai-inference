"""Filesystem checkpoints for resumable patient summarization."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd


_PREPARED_CHUNKS_FILENAME = "prepared_chunks.parquet"
_ROUND_CHECKPOINT_PATTERN = re.compile(r"round_(\d+)\.parquet")
logger = logging.getLogger(__name__)


def prepare_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """Create a patient checkpoint directory if needed and return its path."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_prepared_chunks(checkpoint_dir: str | Path) -> pd.DataFrame | None:
    """Load prepared patient note chunks, or return ``None`` if none were saved."""
    path = Path(checkpoint_dir) / _PREPARED_CHUNKS_FILENAME
    if not path.exists():
        return None
    logger.info("Loading prepared patient data from %s.", path)
    return pd.read_parquet(path)


def save_prepared_chunks(
    checkpoint_dir: str | Path,
    prepared_chunks: pd.DataFrame,
) -> None:
    """Save prepared patient note chunks for reuse by a later retry."""
    path = prepare_checkpoint_dir(checkpoint_dir) / _PREPARED_CHUNKS_FILENAME
    prepared_chunks.to_parquet(path, index=False)
    logger.info("Saved prepared patient data to %s.", path)


def load_round_checkpoints(
    checkpoint_dir: str | Path,
) -> dict[int, pd.DataFrame]:
    """Load completed round checkpoints keyed by their zero-based round index."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return {}

    # Sorting by filename preserves round order because indexes are zero-padded.
    checkpoints: dict[int, pd.DataFrame] = {}
    for path in sorted(checkpoint_path.glob("round_*.parquet")):
        match = _ROUND_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints[int(match.group(1))] = pd.read_parquet(path)
    if checkpoints:
        logger.info(
            "Loaded %d completed patient summarization round(s) from %s.",
            len(checkpoints),
            checkpoint_path,
        )
    return checkpoints


def save_round_checkpoint(
    checkpoint_dir: str | Path,
    round_idx: int,
    round_results: pd.DataFrame,
) -> None:
    """Save all results from one completed patient summarization round."""
    checkpoint_path = prepare_checkpoint_dir(checkpoint_dir)
    path = checkpoint_path / f"round_{round_idx:04d}.parquet"
    round_results.to_parquet(path, index=False)
    logger.info(
        "Saved completed patient summarization round %d to %s.", round_idx, path
    )


__all__ = [
    "load_prepared_chunks",
    "load_round_checkpoints",
    "prepare_checkpoint_dir",
    "save_prepared_chunks",
    "save_round_checkpoint",
]

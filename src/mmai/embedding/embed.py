"""Embedding step for trial/patient matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from mmai.backends import get_backend
from mmai.config import load_default_preset

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def _resolve_text_column(entity_type: Literal["patient", "trial"]) -> str:
    if entity_type == "patient":
        return "cancer_history_summary"
    if entity_type == "trial":
        return "clinical_space_summary"
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def _resolve_id_columns(entity_type: Literal["patient", "trial"]) -> list[str]:
    if entity_type == "patient":
        return ["patient_id"]
    if entity_type == "trial":
        return ["space_trial_id"]
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def embed_for_matching(
    df: pd.DataFrame,
    *,
    entity_type: Literal["patient", "trial"],
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Convert trial or patient summaries into embedding vectors for semantic matching.

    Parameters
    ----------
    df : pd.DataFrame
        Input summaries to embed.

        For entity_type="patient"
            One row per patient.

            Expected columns
            ----------------
            cancer_history_summary : str
                Summary text to embed for each patient.

        For entity_type="trial"
            One row per clinical space.

            Expected columns
            ----------------
            clinical_space_summary : str
                Summary text to embed for each clinical space.
            space_trial_id : str
                Unique identifier for the clinical space.
    entity_type : {"patient", "trial"}
        Controls which summary column is used as the text to embed.
    config : MMAIConfig, optional
        MMAI configuration containing embedding settings
        (model_path, device, prompt_file/query_prompt). Uses default preset
        when omitted.
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the embedding plus identifier columns for matching.

        Columns
        -------
        For entity_type="patient"
            patient_id : str
            embedding : array-like
                Vector representation of the summary text in a shared semantic space.

        For entity_type="trial"
            space_trial_id : str
            embedding : array-like
                Vector representation of the summary text in a shared semantic space.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    """
    text_col = _resolve_text_column(entity_type)
    if text_col not in df.columns:
        raise ValueError(f"df is missing required column for {entity_type}: {text_col}")

    id_cols = _resolve_id_columns(entity_type)
    missing = [col for col in id_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns for {entity_type}: {missing}")

    resolved_config = config or load_default_preset()
    embedding_config = dict(getattr(resolved_config, "embedding", {}))

    output = df.copy()

    summaries = output[text_col].fillna("").astype(str).tolist()
    backend = get_backend(resolved_config.backend)
    embeddings, model_metadata = backend.generate_embeddings(
        summaries,
        embedding_config=embedding_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )
    output["embedding"] = embeddings
    result = output[id_cols + ["embedding"]].copy()
    if return_metadata:
        metadata_payload = {
            "config_snapshot": resolved_config.raw,
            "model_metadata": {
                "embedding_model": model_metadata,
            },
        }
        return result, metadata_payload
    return result

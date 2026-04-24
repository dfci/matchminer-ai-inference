"""Reasonable match check helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import resources

import pandas as pd
import torch

from mmai.backends import get_backend
from mmai.config import load_default_preset

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def _load_reasonable_match_template(filename: str) -> str:
    prompt_path = resources.files("mmai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _build_reasonable_match_prompts(
    candidate_pairs: pd.DataFrame,
    *,
    template: str,
) -> list[str]:
    patient_summaries = (
        candidate_pairs["cancer_history_summary"].fillna("").astype(str).tolist()
    )
    clinical_spaces = (
        candidate_pairs["clinical_space_summary"].fillna("").astype(str).tolist()
    )
    return [
        template.format(clinical_space, patient_summary)
        for patient_summary, clinical_space in zip(
            patient_summaries, clinical_spaces, strict=False
        )
    ]


def reasonable_match_check(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_unreasonable: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate whether each candidate patient-trial pair is a clinically reasonable match.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        DataFrame of candidate patient-trial pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        space_trial_id : str
            Trial-space identifier.
        cancer_history_summary : str
            Patient summary text.
        clinical_space_summary : str
            Trial clinical-space summary text.

    config : MMAIConfig, optional
        MMAI configuration containing reasonable match checker settings.
        Uses default preset when omitted.
    filter_unreasonable : bool, default True
        If True, only return rows where ``reasonable_match`` evaluates to True.
    return_metadata : bool, default False
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.

    Returns
    -------
    pd.DataFrame
        Derived output table containing:

        Columns
        -------
        patient_id : str
            Patient identifier.
        space_trial_id : str
            Trial-space identifier.
        reasonable_match_score : float
            Model-generated confidence score that the candidate pair is clinically
            reasonable.
        reasonable_match : bool
            Whether the candidate pair is considered clinically reasonable.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    """
    # Validate that candidate pair rows contain the text + ids needed for checker prompts.
    required = [
        "patient_id",
        "space_trial_id",
        "cancer_history_summary",
        "clinical_space_summary",
    ]
    missing = [col for col in required if col not in candidate_pairs.columns]
    if missing:
        raise ValueError(
            f"candidate_pairs is missing required columns: {', '.join(missing)}"
        )

    # Resolve run config and build checker prompts from the configured template.
    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("reasonable_match", {}))
    prompt_file = str(checker_config["prompt_file"]).strip()
    score_cutoff = float(checker_config.get("score_cutoff", 0.2))

    template = _load_reasonable_match_template(prompt_file)
    prompts = _build_reasonable_match_prompts(candidate_pairs, template=template)

    # Run the backend text-classification model over all prompts.
    backend = get_backend("local")
    predictions, model_metadata = backend.run_checker(
        prompts,
        checker_config=checker_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )

    if len(predictions) != len(candidate_pairs):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    # Convert model outputs into a compact, derived result table.
    output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
    confidence_scores = [
        float(torch.sigmoid(torch.tensor(float(prediction["score"]))).item())
        for prediction in predictions
    ]
    output["reasonable_match_score"] = confidence_scores
    output["reasonable_match"] = [score >= score_cutoff for score in confidence_scores]

    # Optionally keep only reasonable matches.
    if filter_unreasonable:
        keep_rows = output["reasonable_match"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "config_snapshot": resolved_config.raw,
            "model_metadata": {
                "reasonable_match_checker": model_metadata,
            },
        }
        return output, metadata_payload
    return output

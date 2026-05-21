"""Exclusion criteria check helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import resources

from matchminer_ai.config import config_snapshot, load_default_preset
from .inference import run_checker

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig
    import pandas as pd
else:
    import pandas as pd


def _load_exclusion_criteria_template(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _build_exclusion_criteria_prompts(
    matches: pd.DataFrame,
    *,
    template: str,
) -> list[str]:
    patient_evidence = (
        matches["general_exclusion_criteria_evidence"].fillna("").astype(str).tolist()
    )
    trial_exclusions = (
        matches["general_exclusion_criteria"].fillna("").astype(str).tolist()
    )
    return [
        template.format(evidence, criteria)
        for evidence, criteria in zip(patient_evidence, trial_exclusions, strict=False)
    ]


def exclusion_criteria_check(
    matches: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_excluded: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate whether each candidate patient-trial pair passes exclusion criteria.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame of candidate patient-trial pairs to evaluate for exclusion checks.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        trial_id : str
            Trial identifier.
        general_exclusion_criteria : str
            Trial-level exclusion criteria text.
        general_exclusion_criteria_evidence : str
            Patient-level evidence related to exclusion criteria.

    config : MMAIConfig, optional
        MMAI configuration containing exclusion checker settings.
        Uses default preset when omitted.
    filter_excluded : bool, default False
        If True, return only rows where ``exclusion_criteria_pass`` is True.
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
        trial_id : str
            Trial identifier.
        exclusion_score : float
            Model-generated confidence score associated with exclusion-check label.
        exclusion_criteria_pass : bool
            True when the patient is predicted to pass exclusion criteria for
            this trial; False otherwise.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    # Validate that rows contain the text + ids needed for exclusion checker prompts.
    required = [
        "patient_id",
        "trial_id",
        "general_exclusion_criteria",
        "general_exclusion_criteria_evidence",
    ]
    missing = [col for col in required if col not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing required columns: {', '.join(missing)}")

    # Resolve run config and build checker prompts from the configured template.
    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("exclusion_criteria", {}))
    prompt_file = str(checker_config["prompt_file"]).strip()

    template = _load_exclusion_criteria_template(prompt_file)
    prompts = _build_exclusion_criteria_prompts(matches, template=template)

    # Run the backend text-classification model over all prompts.
    predictions, model_metadata = run_checker(
        prompts,
        checker_config=checker_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )

    if len(predictions) != len(matches):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    # Convert model outputs into a compact, derived result table.
    output = matches[["patient_id", "trial_id"]].copy()
    output["exclusion_score"] = [
        float(prediction["score"]) for prediction in predictions
    ]
    output["exclusion_criteria_pass"] = [
        str(prediction["label"]).strip().upper() == "NEGATIVE"
        for prediction in predictions
    ]

    # Optionally keep only rows that passed exclusion criteria.
    if filter_excluded:
        keep_rows = output["exclusion_criteria_pass"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "exclusion_criteria_checker": model_metadata,
            },
        }
        return output, metadata_payload
    return output


__all__ = ["exclusion_criteria_check"]

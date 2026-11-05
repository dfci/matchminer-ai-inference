"""Match quality scoring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from importlib import resources

import pandas as pd
import torch

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.llm.backends import get_model_metadata
from .inference import run_checker

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _load_match_quality_template(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _validate_text_list(name: str, values: list[str]) -> list[str]:
    # Type hints aren't enforced at runtime. A bare str is iterable as
    # characters, so reject it so it can't pass the length check by accident.
    if isinstance(values, str):
        raise TypeError(f"{name} must be a list of strings, not a single str.")
    texts = list(values)
    bad = [i for i, text in enumerate(texts) if not isinstance(text, str)]
    if bad:
        raise TypeError(
            f"{name} must contain only strings; non-string values at indices {bad}."
        )
    return texts


def score_match_quality_pairs(
    *,
    patient_summaries: list[str],
    clinical_space_summaries: list[str],
    config: MMAIConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Score patient / clinical-space text pairs for match quality, in input order.

    Service-layer entrypoint: takes plain text with no identifiers or
    DataFrames. Pair ``i`` is
    ``(patient_summaries[i], clinical_space_summaries[i])``. To check one
    patient against many spaces, repeat the patient summary.

    Parameters
    ----------
    patient_summaries : list[str]
        Patient summary text, one per pair.
    clinical_space_summaries : list[str]
        Trial clinical-space summary text, one per pair. Must be the same
        length as ``patient_summaries``.
    config : MMAIConfig, optional
        MMAI configuration containing ``match_quality`` checker settings.
        Uses the default preset when omitted.

    Returns
    -------
    tuple[list[dict], dict]
        - One ``{"score": float, "pass": bool}`` per pair, in input order.
          ``score`` is the sigmoid confidence in [0, 1]; ``pass`` is
          ``score >= match_quality.score_cutoff``.
        - Model metadata (``model_name``, ``model_sha``, ...).

    Raises
    ------
    TypeError
        If either argument is a single ``str`` or contains non-string values.
    ValueError
        If the two sequences differ in length, or the checker returns a
        different number of predictions than pairs.
    """
    patients = _validate_text_list("patient_summaries", patient_summaries)
    spaces = _validate_text_list("clinical_space_summaries", clinical_space_summaries)
    if len(patients) != len(spaces):
        raise ValueError(
            "patient_summaries and clinical_space_summaries must be the same "
            f"length; got {len(patients)} and {len(spaces)}."
        )

    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("match_quality", {}))
    score_cutoff = float(checker_config["score_cutoff"])

    # Nothing to score: skip loading the checker pipeline.
    if not patients:
        model_metadata = get_model_metadata(
            checker_config["model_name"],
            cache_dir=resolved_config.model_metadata_cache_dir,
        )
        return [], model_metadata

    template = _load_match_quality_template(str(checker_config["prompt_file"]).strip())
    # The checker was trained on clinical space first, patient summary second.
    prompts = [
        template.format(space, patient)
        for patient, space in zip(patients, spaces, strict=True)
    ]

    predictions, model_metadata = run_checker(
        prompts,
        checker_config=checker_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )
    if len(predictions) != len(prompts):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    results: list[dict[str, Any]] = []
    for prediction in predictions:
        score = float(torch.sigmoid(torch.tensor(float(prediction["score"]))).item())
        results.append({"score": score, "pass": score >= score_cutoff})
    return results, model_metadata


def score_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_low_quality: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate the clinical match quality of each candidate patient-trial pair.

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
        MMAI configuration containing match quality checker settings.
        Uses default preset when omitted.
    filter_low_quality : bool, default True
        If True, only return rows where ``match_quality_pass`` evaluates to True.
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
        match_quality_score : float
            Model-generated confidence score for clinical match quality.
        match_quality_pass : bool
            Whether the match quality score meets the configured cutoff.
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

    # Resolve once so the config snapshot matches the config actually scored with.
    resolved_config = config or load_default_preset()

    # Missing text becomes "" (the service layer requires str), then score.
    results, model_metadata = score_match_quality_pairs(
        patient_summaries=(
            candidate_pairs["cancer_history_summary"].fillna("").astype(str).tolist()
        ),
        clinical_space_summaries=(
            candidate_pairs["clinical_space_summary"].fillna("").astype(str).tolist()
        ),
        config=resolved_config,
    )

    # Map pair results back onto the candidate rows, in input order.
    output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
    # Explicit dtypes so an empty input still yields float / bool columns.
    output["match_quality_score"] = pd.Series(
        [result["score"] for result in results], index=output.index, dtype="float64"
    )
    output["match_quality_pass"] = pd.Series(
        [result["pass"] for result in results], index=output.index, dtype="bool"
    )

    # Optionally keep only matches that pass the quality threshold.
    if filter_low_quality:
        keep_rows = output["match_quality_pass"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "score_match_quality": model_metadata,
            },
        }
        return output, metadata_payload
    return output

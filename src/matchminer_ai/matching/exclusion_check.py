"""Exclusion criteria check helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from importlib import resources

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.llm.backends import get_model_metadata
from .inference import run_checker
from .rerank import _validate_text_list

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig
    import pandas as pd
else:
    import pandas as pd


def _load_exclusion_criteria_template(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def exclusion_criteria_check_pairs(
    *,
    patient_exclusion_evidence: list[str],
    trial_exclusion_criteria: list[str],
    config: MMAIConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Check patient evidence / trial exclusion criteria text pairs, in input order.

    Service-layer entrypoint: takes plain text with no identifiers or
    DataFrames. Pair ``i`` is
    ``(patient_exclusion_evidence[i], trial_exclusion_criteria[i])``. To check
    one patient against many trials, repeat the patient evidence.

    Parameters
    ----------
    patient_exclusion_evidence : list[str]
        Patient-level evidence related to exclusion criteria, one per pair.
    trial_exclusion_criteria : list[str]
        Trial-level exclusion criteria text, one per pair. Must be the same
        length as ``patient_exclusion_evidence``.
    config : MMAIConfig, optional
        MMAI configuration containing ``exclusion_criteria`` checker settings.
        Uses the default preset when omitted.

    Returns
    -------
    tuple[list[dict], dict]
        - One ``{"exclusion_score": float, "exclusion_criteria_pass": bool}``
          per pair, in input order (the same names as the
          ``exclusion_criteria_check`` columns). ``exclusion_criteria_pass``
          is True when the checker predicts the patient meets none of the
          trial's exclusion criteria. ``exclusion_score`` is the checker's
          confidence in that prediction, whichever way it went.
        - Model metadata (``model_name``, ``model_sha``, ...).

    Raises
    ------
    TypeError
        If either argument is a single ``str`` or contains non-string values.
    ValueError
        If the two sequences differ in length, or the checker returns a
        different number of predictions than pairs.
    """
    patients = _validate_text_list(
        "patient_exclusion_evidence", patient_exclusion_evidence
    )
    trials = _validate_text_list("trial_exclusion_criteria", trial_exclusion_criteria)
    if len(patients) != len(trials):
        raise ValueError(
            "patient_exclusion_evidence and trial_exclusion_criteria must be the "
            f"same length; got {len(patients)} and {len(trials)}."
        )

    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.exclusion_criteria)

    # Nothing to check: skip loading the checker pipeline.
    if not patients:
        model_metadata = get_model_metadata(
            checker_config["model_name"],
            cache_dir=resolved_config.model_metadata_cache_dir,
        )
        return [], model_metadata

    template = _load_exclusion_criteria_template(
        str(checker_config["prompt_file"]).strip()
    )

    prompts = [
        template.format(patient, trial)
        for patient, trial in zip(patients, trials, strict=True)
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

    # A NEGATIVE label means no evidence the patient meets the exclusion criteria.
    results: list[dict[str, Any]] = [
        {
            "exclusion_score": float(prediction["score"]),
            "exclusion_criteria_pass": (
                str(prediction["label"]).strip().upper() == "NEGATIVE"
            ),
        }
        for prediction in predictions
    ]
    return results, model_metadata


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

    # Resolve once so the config snapshot matches the config actually checked with.
    resolved_config = config or load_default_preset()

    # Missing text becomes "" (the service layer requires str), then check.
    results, model_metadata = exclusion_criteria_check_pairs(
        patient_exclusion_evidence=(
            matches["general_exclusion_criteria_evidence"]
            .fillna("")
            .astype(str)
            .tolist()
        ),
        trial_exclusion_criteria=(
            matches["general_exclusion_criteria"].fillna("").astype(str).tolist()
        ),
        config=resolved_config,
    )

    # Map pair results back onto the input rows, in input order. Result keys are
    # the column names. Explicit dtypes so an empty input still yields
    # float / bool columns.
    output = matches[["patient_id", "trial_id"]].copy()
    for column, dtype in (
        ("exclusion_score", "float64"),
        ("exclusion_criteria_pass", "bool"),
    ):
        output[column] = pd.Series(
            [result[column] for result in results], index=output.index, dtype=dtype
        )

    # Optionally keep only rows that passed exclusion criteria.
    if filter_excluded:
        keep_rows = output["exclusion_criteria_pass"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "exclusion_criteria_check": model_metadata,
            },
        }
        return output, metadata_payload
    return output


__all__ = ["exclusion_criteria_check", "exclusion_criteria_check_pairs"]

"""Reasonable match check helpers."""

from __future__ import annotations

from importlib import resources

import pandas as pd

DEFAULT_REASONABLE_MATCH_TEMPLATE_FILE = "reasonable_match_checker_template.txt"


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
    filter_unreasonable: bool = True,
) -> pd.DataFrame:
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

    filter_unreasonable : bool, default True
        If True, only return rows where ``reasonable_match`` evaluates to True.

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
    """
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

    template = _load_reasonable_match_template(DEFAULT_REASONABLE_MATCH_TEMPLATE_FILE)
    prompts = _build_reasonable_match_prompts(candidate_pairs, template=template)

    # TODO: call backend checker model on prompts and map output to score/label columns.
    _ = prompts
    _ = filter_unreasonable
    raise NotImplementedError(
        "Reasonable match check model call is not implemented yet."
    )

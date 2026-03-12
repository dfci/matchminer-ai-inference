"""Reasonable match check helpers."""

from __future__ import annotations

import pandas as pd


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
    raise NotImplementedError("Reasonable match check not implemented yet.")

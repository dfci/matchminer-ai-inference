"""Exclusion criteria check helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd


def exclusion_criteria_check(
    matches: pd.DataFrame,
    *,
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
        space_trial_id : str
            Trial-space identifier.
        general_exclusion_criteria : str
            Trial-level exclusion criteria text for the clinical space.
        general_exclusion_criteria_evidence : str
            Patient-level evidence related to exclusion criteria.

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
        space_trial_id : str
            Trial-space identifier.
        exclusion_score : float
            Model-generated confidence score associated with exclusion-check label.
        exclusion_criteria_pass : bool
            True when the patient is predicted to pass exclusion criteria for
            this trial space; False otherwise.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    raise NotImplementedError("Exclusion check not implemented in skeleton.")


__all__ = ["exclusion_criteria_check"]

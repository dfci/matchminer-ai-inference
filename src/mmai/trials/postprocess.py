"""Trial postprocessing logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def _strip_numerical_prefix(text: str) -> str:
    """Remove leading numeric list markers (e.g., '1.' or '12.')."""
    if not text:
        return text
    if len(text) > 2 and text[0:2].isnumeric() and text[2] == ".":
        return text[3:].strip()
    if len(text) > 1 and text[0].isnumeric() and text[1] == ".":
        return text[2:].strip()
    return text


REQUIRED_TRIAL_SPACE_KEYWORDS = [
    "Age",
    "Sex",
    "Cancer type",
    "Histology",
    "Cancer burden",
    "Prior treatment required",
    "Prior treatment excluded",
    "Biomarkers required",
    "Biomarkers excluded",
]


def _expand_trial_spaces(
    trials_with_summaries: pd.DataFrame,
    reasoning_marker: str,
    boilerplate_marker: str,
) -> pd.DataFrame:
    """Expand LLM summaries into one row per clinical space."""
    trials_with_summaries = trials_with_summaries.copy()
    trials_with_summaries["space_output_no_reasoning"] = (
        trials_with_summaries["space_reasoning_and_output"]
        .str.split(reasoning_marker, n=1)
        .apply(lambda parts: parts[-1])
    )

    trials_with_summaries[["space_text", "boilerplate_text"]] = trials_with_summaries[
        "space_output_no_reasoning"
    ].str.split(boilerplate_marker, n=1, expand=True, regex=True)
    trials_with_summaries["space_text"] = trials_with_summaries[
        "space_text"
    ].str.strip()
    trials_with_summaries["boilerplate_text"] = (
        trials_with_summaries["boilerplate_text"]
        .str.strip()
        .fillna("None")
        .replace("", "None")
    )

    frames: list[pd.DataFrame] = []
    for i in range(trials_with_summaries.shape[0]):
        space_text = trials_with_summaries.iloc[i]["space_text"]
        if not isinstance(space_text, str):
            continue

        cohorts = pd.Series(space_text.split("\n")).str.strip()
        cohorts = cohorts[cohorts.str.match(r"[1-9]")]
        cohorts = cohorts.apply(_strip_numerical_prefix)
        # clean up spaces
        cohorts = cohorts[
            ~((cohorts.isnull()) | (cohorts == "\n") | (cohorts == ""))
        ].reset_index(drop=True)

        if not len(cohorts):
            continue

        trial_row = trials_with_summaries.iloc[i]
        frame = pd.concat(
            [trial_row.to_frame().T] * len(cohorts),
            ignore_index=True,
        )
        frame["clinical_space_summary"] = cohorts
        frame["clinical_space_number"] = frame.index + 1
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                *trials_with_summaries.columns,
                "clinical_space_summary",
                "clinical_space_number",
            ]
        )

    return pd.concat(frames, axis=0).reset_index(drop=True)


def flatten_trial_to_spaces(
    trials_with_summaries: pd.DataFrame,
    reasoning_marker: str,
    boilerplate_marker: str,
) -> pd.DataFrame:
    """Split LLM summaries into individual clinical spaces and do keyword filtering."""
    cohort_level_trials = _expand_trial_spaces(
        trials_with_summaries,
        reasoning_marker=reasoning_marker,
        boilerplate_marker=boilerplate_marker,
    )
    if cohort_level_trials.empty:
        return cohort_level_trials
    return cohort_level_trials.loc[
        cohort_level_trials["clinical_space_summary"].apply(
            lambda text: all(term in text for term in REQUIRED_TRIAL_SPACE_KEYWORDS)
        )
    ]


def postprocess_trial_summaries(
    trials_with_summaries: pd.DataFrame,
    config: MMAIConfig,
    *,
    return_qc_data: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Postprocess trial summaries into clinical spaces."""
    trial_config = dict(config.trial)
    reasoning_marker = trial_config["reasoning_marker"]
    boilerplate_marker = trial_config["boilerplate_marker"]

    if return_qc_data:
        unfiltered_spaces = _expand_trial_spaces(
            trials_with_summaries,
            reasoning_marker=reasoning_marker,
            boilerplate_marker=boilerplate_marker,
        )
        spaces = unfiltered_spaces.loc[
            unfiltered_spaces["clinical_space_summary"].apply(
                lambda text: all(term in text for term in REQUIRED_TRIAL_SPACE_KEYWORDS)
            )
        ]
    else:
        unfiltered_spaces = pd.DataFrame()
        spaces = flatten_trial_to_spaces(
            trials_with_summaries,
            reasoning_marker=reasoning_marker,
            boilerplate_marker=boilerplate_marker,
        )
    if spaces.empty:
        empty = pd.DataFrame(
            columns=[
                "space_trial_id",
                "trial_id",
                "clinical_space_number",
                "clinical_space_summary",
                "general_exclusion_criteria",
            ]
        )
        return (empty, unfiltered_spaces) if return_qc_data else empty

    output = pd.DataFrame(
        {
            "trial_id": spaces["trial_id"],
            "clinical_space_number": spaces["clinical_space_number"],
            "clinical_space_summary": spaces["clinical_space_summary"],
            "general_exclusion_criteria": spaces["boilerplate_text"],
        }
    )
    output["space_trial_id"] = (
        output["trial_id"].astype(str)
        + "-"
        + output["clinical_space_number"].astype(str)
    )

    if config.debug_mode:
        output["trial_text"] = spaces["trial_text"]
        output["space_reasoning_and_output"] = spaces["space_reasoning_and_output"]
    # qc spaces = unfiltered trial spaces
    return (output, unfiltered_spaces) if return_qc_data else output

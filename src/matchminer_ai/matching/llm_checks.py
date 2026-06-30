"""LLM-based match quality and exclusion screening helpers."""

from __future__ import annotations

import re
from importlib import resources
from typing import TYPE_CHECKING, Any

import pandas as pd

from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


_TRIAL_CHECK_SCORE_PATTERN = re.compile(r"[Ff]inal\s+[Ss]core\s*:\s*(\d)")


def _load_prompt_template(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _build_messages(user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Reasoning: high"},
        {"role": "user", "content": user_content},
    ]


def _parse_match_quality_score(response_text: str) -> tuple[int, str]:
    tail = response_text[-60:].replace("*", "").replace("\u202f", " ")
    match = _TRIAL_CHECK_SCORE_PATTERN.search(tail)
    if match:
        score = min(int(match.group(1)), 5)
        return score, "parsed"

    tail_upper = tail.upper()
    fallback = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
    if fallback:
        score = min(int(fallback.group(1)), 5)
        return score, "parsed_fallback"
    if "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
        return 0, "parsed_fallback"
    return -1, "parse_failed"


def _parse_exclusion_result(response_text: str) -> tuple[bool | None, str]:
    tail = response_text[-10:].upper()
    if "YES!" in tail:
        return False, "parsed"
    if "NO!" in tail:
        return True, "parsed"
    if "YES" in tail:
        return False, "parsed_fallback"
    if "NO" in tail:
        return True, "parsed_fallback"
    return None, "parse_failed"


def _run_llm_check(
    rows: pd.DataFrame,
    *,
    config: "MMAIConfig",
    section_name: str,
    messages_list: list[list[dict[str, str]]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    llm_config = dict(config.raw.get(section_name, {}))
    if not llm_config:
        raise ValueError(f"Config is missing '{section_name}' settings.")
    runtime_config = build_llm_runtime_config(
        section_name,
        llm_config,
        config=config,
    )
    prompt_list = build_prompt_list(messages_list, llm_config=runtime_config)
    backend = get_llm_backend(config)
    generation = backend.generate_llm_outputs(
        prompt_list=prompt_list,
        llm_config=runtime_config,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )
    if len(generation.final_outputs) != len(rows):
        raise ValueError("LLM returned a different number of outputs than input rows.")
    return (
        generation.final_outputs,
        generation.reasoning_outputs,
        generation.model_metadata,
    )


def score_match_quality_with_llm(
    candidate_pairs: pd.DataFrame,
    *,
    config: "MMAIConfig | None" = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Score candidate patient-trial matches with the configured LLM prompt.

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
        MMAI configuration containing LLM match-quality checker settings.
        Uses default preset when omitted.
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
        llm_match_quality_score : int
            Parsed LLM score from 0 to 5, or -1 when parsing failed.
        Debug Columns
        -------------
        llm_match_quality_answer_text : str
            Text the package treated as the LLM answer and used for parsing.
            Included only when debug_mode is true.
        llm_match_quality_reasoning_text : str
            Optional separate reasoning trace returned by the backend or
            extracted by the configured reasoning parser. Included only when
            debug_mode is true.
        llm_match_quality_parse_status : str
            Whether the package could parse the answer text. Values are
            ``parsed``, ``parsed_fallback``, or ``parse_failed``. Included only
            when debug_mode is true.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
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

    resolved_config = config or load_default_preset()
    llm_config = dict(resolved_config.raw.get("llm_match_quality", {}))
    prompt_template = _load_prompt_template(str(llm_config["prompt_file"]).strip())
    messages_list = [
        _build_messages(
            prompt_template.format(
                trial_summary=str(row["clinical_space_summary"]),
                patient_summary=str(row["cancer_history_summary"]),
            )
        )
        for _, row in candidate_pairs.iterrows()
    ]
    responses, reasonings, model_metadata = _run_llm_check(
        candidate_pairs,
        config=resolved_config,
        section_name="llm_match_quality",
        messages_list=messages_list,
    )
    parsed = [_parse_match_quality_score(response) for response in responses]

    output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
    output["llm_match_quality_score"] = [score for score, _ in parsed]
    if resolved_config.debug_mode:
        output["llm_match_quality_answer_text"] = responses
        output["llm_match_quality_reasoning_text"] = reasonings
        output["llm_match_quality_parse_status"] = [
            parse_status for _, parse_status in parsed
        ]
    output = output.reset_index(drop=True)

    if return_metadata:
        metadata_payload = {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "llm_match_quality_checker": model_metadata,
            },
        }
        return output, metadata_payload
    return output


def exclusion_criteria_check_with_llm(
    matches: pd.DataFrame,
    *,
    config: "MMAIConfig | None" = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate trial-level exclusion criteria with the configured LLM prompt.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame of candidate patient-trial pairs to evaluate for exclusion
        checks.

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
        MMAI configuration containing LLM exclusion checker settings.
        Uses default preset when omitted.
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
        llm_exclusion_criteria_pass : bool | None
            Whether the LLM judged that the patient passes exclusion criteria,
            or None when parsing failed.
        Debug Columns
        -------------
        llm_exclusion_criteria_answer_text : str
            Text the package treated as the LLM answer and used for parsing.
            Included only when debug_mode is true.
        llm_exclusion_criteria_reasoning_text : str
            Optional separate reasoning trace returned by the backend or
            extracted by the configured reasoning parser. Included only when
            debug_mode is true.
        llm_exclusion_criteria_parse_status : str
            Whether the package could parse the answer text. Values are
            ``parsed``, ``parsed_fallback``, or ``parse_failed``. Included only
            when debug_mode is true.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    required = [
        "patient_id",
        "trial_id",
        "general_exclusion_criteria",
        "general_exclusion_criteria_evidence",
    ]
    missing = [col for col in required if col not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing required columns: {', '.join(missing)}")

    resolved_config = config or load_default_preset()
    llm_config = dict(resolved_config.raw.get("llm_exclusion_criteria", {}))
    prompt_template = _load_prompt_template(str(llm_config["prompt_file"]).strip())
    messages_list = [
        _build_messages(
            prompt_template.format(
                patient_boilerplate=str(row["general_exclusion_criteria_evidence"]),
                trial_boilerplate=str(row["general_exclusion_criteria"]),
            )
        )
        for _, row in matches.iterrows()
    ]
    responses, reasonings, model_metadata = _run_llm_check(
        matches,
        config=resolved_config,
        section_name="llm_exclusion_criteria",
        messages_list=messages_list,
    )
    parsed = [_parse_exclusion_result(response) for response in responses]

    output = matches[["patient_id", "trial_id"]].copy()
    output["llm_exclusion_criteria_pass"] = [passed for passed, _ in parsed]
    if resolved_config.debug_mode:
        output["llm_exclusion_criteria_answer_text"] = responses
        output["llm_exclusion_criteria_reasoning_text"] = reasonings
        output["llm_exclusion_criteria_parse_status"] = [
            parse_status for _, parse_status in parsed
        ]
    output = output.reset_index(drop=True)

    if return_metadata:
        metadata_payload = {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "llm_exclusion_criteria_checker": model_metadata,
            },
        }
        return output, metadata_payload
    return output


__all__ = [
    "exclusion_criteria_check_with_llm",
    "score_match_quality_with_llm",
]

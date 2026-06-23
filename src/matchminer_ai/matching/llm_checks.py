"""LLM-based match quality and exclusion screening helpers."""

from __future__ import annotations

import re
from importlib import resources
from typing import TYPE_CHECKING, Any

import pandas as pd

from matchminer_ai.config import config_snapshot, load_default_preset
from matchminer_ai.llm.backends import (
    build_summarization_runtime_config,
    get_summarization_backend,
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
        return score, f"Score:{score}"

    tail_upper = tail.upper()
    fallback = re.search(r"SCORE\s*[:\-=]\s*(\d)", tail_upper)
    if fallback:
        score = min(int(fallback.group(1)), 5)
        return score, f"Score:{score}"
    if "NOT REASONABLE" in tail_upper or "NOT A REASONABLE" in tail_upper:
        return 0, "Score:0"
    return -1, "PARSE_FAILED"


def _parse_exclusion_result(response_text: str) -> tuple[bool | None, str]:
    tail = response_text[-10:].upper()
    if "YES!" in tail:
        return False, "YES"
    if "NO!" in tail:
        return True, "NO"
    if "YES" in tail:
        return False, "YES_FALLBACK"
    if "NO" in tail:
        return True, "NO_FALLBACK"
    return None, "PARSE_FAILED"


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
    runtime_config = build_summarization_runtime_config(
        section_name,
        llm_config,
        config=config,
    )
    prompt_list = build_prompt_list(messages_list, llm_config=runtime_config)
    backend = get_summarization_backend(config)
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

    Expected columns are ``patient_id``, ``space_trial_id``,
    ``cancer_history_summary``, and ``clinical_space_summary``.
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
    output["llm_match_quality_verdict"] = [verdict for _, verdict in parsed]
    output["llm_match_quality_response"] = responses
    output["llm_match_quality_reasoning"] = reasonings
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

    Expected columns are ``patient_id``, ``trial_id``,
    ``general_exclusion_criteria``, and
    ``general_exclusion_criteria_evidence``.
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
    output["llm_exclusion_criteria_verdict"] = [verdict for _, verdict in parsed]
    output["llm_exclusion_criteria_response"] = responses
    output["llm_exclusion_criteria_reasoning"] = reasonings
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

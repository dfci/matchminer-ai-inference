import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)


def test_generate_candidate_matches_returns_top_k_per_query():
    """Return top-k matches per query sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1", "T3-1"],
            "trial_id": ["T1", "T2", "T3"],
            "clinical_space_number": [1, 1, 1],
            "embedding": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=2)

    assert list(result.columns[:4]) == [
        "patient_id",
        "space_trial_id",
        "similarity_score",
        "rank",
    ]
    assert (result["patient_id"].value_counts() == 2).all()
    p1 = result[result["patient_id"] == "P1"]["space_trial_id"].tolist()
    p2 = result[result["patient_id"] == "P2"]["space_trial_id"].tolist()
    assert p1[0] == "T1-1"
    assert p2[0] == "T3-1"


def test_generate_candidate_matches_returns_all_when_k_none():
    """Return all corpus rows per query when k is None, sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "embedding": [[1.0, 0.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1"],
            "trial_id": ["T1", "T2"],
            "clinical_space_number": [1, 1],
            "embedding": [[0.0, 1.0], [1.0, 0.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=None)

    assert result["space_trial_id"].tolist() == ["T2-1", "T1-1"]
    assert result["rank"].tolist() == [1, 2]


class _MockCheckerBackend:
    def __init__(self, predictions):
        self.predictions = predictions
        self.last_prompts = None
        self.last_checker_config = None
        self.last_model_metadata_cache_dir = None

    def run_checker(self, prompts, *, checker_config, model_metadata_cache_dir=None):
        self.last_prompts = prompts
        self.last_checker_config = checker_config
        self.last_model_metadata_cache_dir = model_metadata_cache_dir
        return self.predictions, {"model_name": checker_config["model_name"]}


def test_score_match_quality_maps_outputs_and_filters(monkeypatch):
    """Map checker logits to confidence scores and apply cutoff-based filtering."""
    backend = _MockCheckerBackend(
        [
            {"score": 2.0},
            {"score": -2.0},
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.rerank.run_checker", backend.run_checker
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={},
        model_metadata_cache_dir=None,
        raw={
            "match_quality": {
                "model_name": "ksg-dfci/TrialChecker-1225",
                "device": "cpu",
                "prompt_file": "match_quality_checker_template.txt",
                "score_cutoff": 0.2,
            }
        },
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "space_trial_id": "T1-1",
                "cancer_history_summary": "summary a",
                "clinical_space_summary": "space a",
            },
            {
                "patient_id": "P2",
                "space_trial_id": "T2-1",
                "cancer_history_summary": "summary b",
                "clinical_space_summary": "space b",
            },
        ]
    )

    unfiltered = score_match_quality(pairs, config=config, filter_low_quality=False)
    assert list(unfiltered.columns) == [
        "patient_id",
        "space_trial_id",
        "match_quality_score",
        "match_quality_pass",
    ]
    assert unfiltered["match_quality_pass"].tolist() == [True, False]
    assert unfiltered["match_quality_score"].tolist() == pytest.approx(
        [0.880797, 0.119203],
        abs=1e-6,
    )
    assert backend.last_checker_config["model_name"] == "ksg-dfci/TrialChecker-1225"

    filtered = score_match_quality(pairs, config=config, filter_low_quality=True)
    assert len(filtered) == 1
    assert filtered.loc[0, "patient_id"] == "P1"
    assert bool(filtered.loc[0, "match_quality_pass"]) is True


def test_score_match_quality_return_metadata(monkeypatch):
    """Return config snapshot and checker model metadata when requested."""
    backend = _MockCheckerBackend(
        [
            {"score": 2.0},
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.rerank.run_checker", backend.run_checker
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        raw={
            "match_quality": {
                "model_name": "ksg-dfci/TrialChecker-1225",
                "device": "cpu",
                "prompt_file": "match_quality_checker_template.txt",
                "score_cutoff": 0.2,
            }
        },
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "space_trial_id": "T1-1",
                "cancer_history_summary": "summary a",
                "clinical_space_summary": "space a",
            }
        ]
    )

    result, metadata = score_match_quality(
        pairs,
        config=config,
        filter_low_quality=False,
        return_metadata=True,
    )

    assert len(result) == 1
    assert metadata["config_snapshot"]["match_quality"]["model_name"] == (
        "ksg-dfci/TrialChecker-1225"
    )
    assert metadata["model_metadata"]["match_quality_checker"]["model_name"] == (
        "ksg-dfci/TrialChecker-1225"
    )


def test_exclusion_criteria_check_maps_outputs_and_filters(monkeypatch):
    """Map boilerplate checker outputs and apply optional pass filtering."""
    backend = _MockCheckerBackend(
        [
            {"label": "NEGATIVE", "score": 0.81},
            {"label": "POSITIVE", "score": 0.66},
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.exclusion_check.run_checker", backend.run_checker
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={},
        model_metadata_cache_dir=None,
        raw={
            "exclusion_criteria": {
                "model_name": "ksg-dfci/BoilerPlateChecker-1225",
                "device": "cpu",
                "prompt_file": "exclusion_criteria_checker_template.txt",
            }
        },
    )
    matches = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "T1",
                "general_exclusion_criteria": "Trial excludes X",
                "general_exclusion_criteria_evidence": "Patient has no X",
            },
            {
                "patient_id": "P2",
                "trial_id": "T2",
                "general_exclusion_criteria": "Trial excludes Y",
                "general_exclusion_criteria_evidence": "Patient has Y",
            },
        ]
    )

    unfiltered = exclusion_criteria_check(
        matches,
        config=config,
        filter_excluded=False,
    )
    assert list(unfiltered.columns) == [
        "patient_id",
        "trial_id",
        "exclusion_score",
        "exclusion_criteria_pass",
    ]
    assert unfiltered["exclusion_criteria_pass"].tolist() == [True, False]
    assert unfiltered["exclusion_score"].tolist() == [0.81, 0.66]
    assert (
        backend.last_checker_config["model_name"] == "ksg-dfci/BoilerPlateChecker-1225"
    )

    filtered = exclusion_criteria_check(matches, config=config, filter_excluded=True)
    assert len(filtered) == 1
    assert filtered.loc[0, "patient_id"] == "P1"
    assert bool(filtered.loc[0, "exclusion_criteria_pass"]) is True


def test_exclusion_criteria_check_return_metadata(monkeypatch):
    """Return config snapshot and exclusion-checker model metadata."""
    backend = _MockCheckerBackend(
        [
            {"label": "NEGATIVE", "score": 0.81},
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.exclusion_check.run_checker", backend.run_checker
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        raw={
            "exclusion_criteria": {
                "model_name": "ksg-dfci/BoilerPlateChecker-1225",
                "device": "cpu",
                "prompt_file": "exclusion_criteria_checker_template.txt",
            }
        },
    )
    matches = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "T1",
                "general_exclusion_criteria": "Trial excludes X",
                "general_exclusion_criteria_evidence": "Patient has no X",
            }
        ]
    )

    result, metadata = exclusion_criteria_check(
        matches,
        config=config,
        filter_excluded=False,
        return_metadata=True,
    )

    assert len(result) == 1
    assert metadata["config_snapshot"]["exclusion_criteria"]["model_name"] == (
        "ksg-dfci/BoilerPlateChecker-1225"
    )
    assert metadata["model_metadata"]["exclusion_criteria_checker"]["model_name"] == (
        "ksg-dfci/BoilerPlateChecker-1225"
    )

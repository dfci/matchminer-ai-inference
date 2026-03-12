import pandas as pd

from mmai.config import MMAIConfig
from mmai.matching import generate_candidate_matches, reasonable_match_check


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


class _MockReasonableBackend:
    def __init__(self, predictions):
        self.predictions = predictions
        self.last_prompts = None
        self.last_checker_config = None
        self.last_model_metadata_cache_dir = None

    def check_reasonable_matches(
        self, prompts, *, checker_config, model_metadata_cache_dir=None
    ):
        self.last_prompts = prompts
        self.last_checker_config = checker_config
        self.last_model_metadata_cache_dir = model_metadata_cache_dir
        return self.predictions, {"model_name": checker_config["model_name"]}


def test_reasonable_match_check_maps_outputs_and_filters(monkeypatch):
    """Map checker label/score outputs and apply optional filtering."""
    backend = _MockReasonableBackend(
        [
            {"label": "POSITIVE", "score": 0.91},
            {"label": "NEGATIVE", "score": 0.12},
        ]
    )
    monkeypatch.setattr("mmai.matching.reasonable_check.get_backend", lambda _: backend)
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        embedding={},
        model_metadata_cache_dir=None,
        raw={
            "reasonable_match": {
                "model_name": "ksg-dfci/TrialChecker-1225",
                "device": "cpu",
                "batch_size": 16,
                "prompt_file": "reasonable_match_checker_template.txt",
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

    unfiltered = reasonable_match_check(pairs, config=config, filter_unreasonable=False)
    assert list(unfiltered.columns) == [
        "patient_id",
        "space_trial_id",
        "reasonable_match_score",
        "reasonable_match",
    ]
    assert unfiltered["reasonable_match"].tolist() == [True, False]
    assert unfiltered["reasonable_match_score"].tolist() == [0.91, 0.12]
    assert backend.last_checker_config["model_name"] == "ksg-dfci/TrialChecker-1225"

    filtered = reasonable_match_check(pairs, config=config, filter_unreasonable=True)
    assert len(filtered) == 1
    assert filtered.loc[0, "patient_id"] == "P1"
    assert bool(filtered.loc[0, "reasonable_match"]) is True


def test_reasonable_match_check_return_metadata(monkeypatch):
    """Return config snapshot and checker model metadata when requested."""
    backend = _MockReasonableBackend(
        [
            {"label": "POSITIVE", "score": 0.91},
        ]
    )
    monkeypatch.setattr("mmai.matching.reasonable_check.get_backend", lambda _: backend)
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        embedding={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        raw={
            "reasonable_match": {
                "model_name": "ksg-dfci/TrialChecker-1225",
                "device": "cpu",
                "batch_size": 16,
                "prompt_file": "reasonable_match_checker_template.txt",
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

    result, metadata = reasonable_match_check(
        pairs,
        config=config,
        filter_unreasonable=False,
        return_metadata=True,
    )

    assert len(result) == 1
    assert metadata["config_snapshot"]["reasonable_match"]["model_name"] == (
        "ksg-dfci/TrialChecker-1225"
    )
    assert metadata["model_metadata"]["reasonable_match_checker"]["model_name"] == (
        "ksg-dfci/TrialChecker-1225"
    )

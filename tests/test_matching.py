import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    score_match_quality,
    score_match_quality_with_llm,
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


def test_run_checker_reuses_cached_pipeline(monkeypatch):
    """Repeated checker calls with the same config should reuse the pipeline."""
    from matchminer_ai.matching import inference

    inference.clear_checker_pipeline_cache()
    tokenizer_calls = []
    pipeline_calls = []

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            tokenizer_calls.append((model_name, kwargs))
            return f"tokenizer:{model_name}"

    class FakePipeline:
        def __init__(self):
            self.prompt_batches = []

        def __call__(self, prompts):
            self.prompt_batches.append(list(prompts))
            return [
                {"label": "POSITIVE", "score": float(index)}
                for index, _prompt in enumerate(prompts)
            ]

    def fake_pipeline(*args, **kwargs):
        checker_pipeline = FakePipeline()
        pipeline_calls.append((args, kwargs, checker_pipeline))
        return checker_pipeline

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        pipeline=fake_pipeline,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "matchminer_ai.matching.inference.get_model_metadata",
        lambda model_name, cache_dir=None: {"model_name": model_name},
    )

    checker_config = {
        "model_name": "checker/model",
        "device": "cpu",
        "max_length": 3192,
    }
    first_outputs, first_metadata = inference.run_checker(
        ["prompt 1"],
        checker_config=checker_config,
        model_metadata_cache_dir=".cache/models",
    )
    second_outputs, second_metadata = inference.run_checker(
        ["prompt 2", "prompt 3"],
        checker_config=checker_config,
        model_metadata_cache_dir=".cache/models",
    )

    assert first_outputs == [{"label": "POSITIVE", "score": 0.0}]
    assert second_outputs == [
        {"label": "POSITIVE", "score": 0.0},
        {"label": "POSITIVE", "score": 1.0},
    ]
    assert first_metadata == {"model_name": "checker/model"}
    assert second_metadata == {"model_name": "checker/model"}
    assert len(tokenizer_calls) == 1
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0][0] == ("text-classification", "checker/model")
    assert pipeline_calls[0][1]["tokenizer"] == "tokenizer:checker/model"
    assert pipeline_calls[0][1]["device"] == "cpu"
    assert pipeline_calls[0][1]["max_length"] == 3192
    assert pipeline_calls[0][2].prompt_batches == [
        ["prompt 1"],
        ["prompt 2", "prompt 3"],
    ]

    inference.clear_checker_pipeline_cache()


def test_clear_checker_pipeline_cache_forces_reload(monkeypatch):
    """Clearing the checker cache should make the next run rebuild the pipeline."""
    from matchminer_ai.matching import inference

    inference.clear_checker_pipeline_cache()
    pipeline_calls = []

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return f"tokenizer:{model_name}"

    def fake_pipeline(*args, **kwargs):
        pipeline_calls.append((args, kwargs))
        return lambda prompts: [{"label": "NEGATIVE", "score": 0.5} for _ in prompts]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, pipeline=fake_pipeline),
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.inference.get_model_metadata",
        lambda model_name, cache_dir=None: {"model_name": model_name},
    )

    checker_config = {"model_name": "checker/model", "device": "cpu"}
    inference.run_checker(["prompt 1"], checker_config=checker_config)
    inference.clear_checker_pipeline_cache()
    inference.run_checker(["prompt 2"], checker_config=checker_config)

    assert len(pipeline_calls) == 2

    inference.clear_checker_pipeline_cache()


class _MockLLMBackend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.last_prompt_list = None
        self.last_llm_config = None
        self.last_model_metadata_cache_dir = None

    def generate_llm_outputs(
        self,
        *,
        prompt_list,
        llm_config,
        model_metadata_cache_dir=None,
    ):
        self.last_prompt_list = prompt_list
        self.last_llm_config = llm_config
        self.last_model_metadata_cache_dir = model_metadata_cache_dir
        return SimpleNamespace(
            final_outputs=self.outputs,
            reasoning_outputs=[f"reasoning {idx}" for idx, _ in enumerate(self.outputs)],
            model_metadata={"model_name": llm_config["model_name"]},
        )


def _llm_config() -> MMAIConfig:
    return MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={
            "llm_match_quality": {
                "max_model_len": 50000,
                "tensor_parallel_size": 1,
            },
            "llm_exclusion_criteria": {
                "max_model_len": 50000,
                "tensor_parallel_size": 1,
            },
        },
        remote={},
        embedding={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        raw={
            "llm_match_quality": {
                "model_name": "llm/model",
                "reasoning_parser": "auto",
                "chat_template_kwargs": {"enable_thinking": True},
                "sampling_params": {"max_tokens": 15000},
                "prompt_file": "llm_match_quality.user.txt",
            },
            "llm_exclusion_criteria": {
                "model_name": "llm/model",
                "reasoning_parser": "auto",
                "chat_template_kwargs": {"enable_thinking": True},
                "sampling_params": {"max_tokens": 20000},
                "prompt_file": "llm_exclusion_criteria.user.txt",
            },
        },
    )


def test_score_match_quality_with_llm_builds_training_prompt_and_parses(monkeypatch):
    """Run LLM match scoring through package backend config and score parser."""
    captured_messages = []

    def fake_build_prompt_list(messages_list, *, llm_config):
        captured_messages.extend(messages_list)
        return [SimpleNamespace(prompt_text="rendered", max_tokens=15000)]

    backend = _MockLLMBackend(["After review.\nFinal score: 4"])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_summarization_backend",
        lambda config: backend,
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "space_trial_id": "T1-1",
                "cancer_history_summary": "Patient has metastatic lung cancer.",
                "clinical_space_summary": "Trial for metastatic lung cancer.",
            }
        ]
    )

    result, metadata = score_match_quality_with_llm(
        pairs,
        config=_llm_config(),
        return_metadata=True,
    )

    assert result["llm_match_quality_score"].tolist() == [4]
    assert result["llm_match_quality_verdict"].tolist() == ["Score:4"]
    assert result["llm_match_quality_reasoning"].tolist() == ["reasoning 0"]
    assert captured_messages[0][0] == {"role": "system", "content": "Reasoning: high"}
    assert "metastatic lung cancer" in captured_messages[0][1]["content"]
    assert backend.last_llm_config["max_model_len"] == 50000
    assert backend.last_llm_config["sampling_params"]["max_tokens"] == 15000
    assert metadata["model_metadata"]["llm_match_quality_checker"]["model_name"] == (
        "llm/model"
    )


def test_exclusion_criteria_check_with_llm_builds_training_prompt_and_parses(
    monkeypatch,
):
    """Run LLM exclusion scoring through package backend config and yes/no parser."""
    captured_messages = []

    def fake_build_prompt_list(messages_list, *, llm_config):
        captured_messages.extend(messages_list)
        return [SimpleNamespace(prompt_text="rendered", max_tokens=20000)]

    backend = _MockLLMBackend(["No!"])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_summarization_backend",
        lambda config: backend,
    )
    matches = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "T1",
                "general_exclusion_criteria": "Excludes uncontrolled brain mets.",
                "general_exclusion_criteria_evidence": "No active brain mets.",
            }
        ]
    )

    result = exclusion_criteria_check_with_llm(matches, config=_llm_config())

    assert result["llm_exclusion_criteria_pass"].tolist() == [True]
    assert result["llm_exclusion_criteria_verdict"].tolist() == ["NO"]
    assert captured_messages[0][0] == {"role": "system", "content": "Reasoning: high"}
    assert "uncontrolled brain mets" in captured_messages[0][1]["content"]
    assert backend.last_llm_config["max_model_len"] == 50000
    assert backend.last_llm_config["sampling_params"]["max_tokens"] == 20000

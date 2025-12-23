from unittest.mock import MagicMock

import pandas as pd

from mmai.config import MMAIConfig
from mmai.trials.prompt_builder import build_trial_text, get_filled_trial_prompt
from mmai.trials.summarize import run_llm_summarization


def test_build_trial_text_normalizes_whitespace():
    trials = pd.DataFrame(
        [
            {
                "trial_title": "Title",
                "brief_summary": "Brief\nsummary",
                "eligibility_criteria": "Eligibility   criteria",
            }
        ]
    )
    text = build_trial_text(trials).iloc[0]
    assert text == "Title Brief summary Eligibility criteria"


def test_get_filled_trial_prompt_includes_trial_text():
    prompts = get_filled_trial_prompt(
        "FIND ME", "trial.user.primer.txt", "trial.user.question.txt"
    )
    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "FIND ME" in prompts[1]["content"]


def test_run_llm_summarization_returns_metadata(monkeypatch):
    mock_backend = MagicMock()
    mock_backend.get_llm.return_value = (
        MagicMock(),
        {"model_name": "model", "model_sha": "sha"},
    )
    monkeypatch.setattr("mmai.trials.summarize.get_backend", lambda name: mock_backend)

    mock_summarize = MagicMock()
    mock_summarize.return_value = ["SUM0"]
    monkeypatch.setattr(
        "mmai.trials.summarize.summarize_trials_multi_cohort", mock_summarize
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={
            "model_name": "model",
            "max_model_len": 100,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 10,
                "repetition_penalty": 1.0,
            },
            "prompt_files": {
                "primer": "trial.user.primer.txt",
                "question": "trial.user.question.txt",
            },
        },
    )

    trials = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title",
                "brief_summary": "Brief",
                "eligibility_criteria": "Criteria",
            }
        ]
    )

    df, metadata = run_llm_summarization(trials, config)
    assert df["space_reasoning_and_output"].iloc[0] == "SUM0"
    assert "trial_text" in df.columns
    assert metadata["config_snapshot"]["trial"]["model_name"] == "model"
    assert metadata["model_metadata"]["model_sha"] == "sha"

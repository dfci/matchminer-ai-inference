from unittest.mock import MagicMock

import pandas as pd

from mmai.config import MMAIConfig
from mmai.backends import LocalBackend
from mmai.trials.postprocess import flatten_trial_to_spaces
from mmai.trials.prompt_builder import build_trial_text, get_filled_trial_prompt
from mmai.trials import summarize_trials
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
    monkeypatch.setattr("mmai.trials.summarize.get_backend", lambda name: mock_backend)

    mock_summarize = MagicMock()
    mock_summarize.return_value = (
        ["SUM0"],
        {"model_sha": "sha"},
    )
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


def test_flatten_trial_to_spaces():
    df = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "space_reasoning_and_output": (
                    "assistantfinal\n"
                    "1. Cancer type allowed: A.\n"
                    "2. Cancer type allowed: B.\n"
                    "Boilerplate exclusions:\n"
                    "Uncontrolled brain metastases."
                ),
            }
        ]
    )
    result = flatten_trial_to_spaces(
        df,
        reasoning_marker="assistantfinal",
        boilerplate_marker="Boilerplate exclusions:",
    )
    assert len(result) == 2
    assert result["clinical_space_summary"].tolist() == [
        "Cancer type allowed: A.",
        "Cancer type allowed: B.",
    ]


def test_local_backend_generate_from_messages(monkeypatch):
    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    mock_llm.get_tokenizer.return_value = mock_tokenizer
    mock_tokenizer.apply_chat_template.side_effect = ["p1", "p2"]

    mock_outputs = [MagicMock(), MagicMock()]
    for n, output in enumerate(mock_outputs):
        output.outputs[0].text = f"SUM{n}"
    mock_llm.generate.return_value = mock_outputs

    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    mock_vllm.SamplingParams = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "vllm", mock_vllm)
    monkeypatch.setattr(
        "mmai.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_sha": "sha",
            "created_at": "now",
            "last_modified": "now",
        },
    )

    backend = LocalBackend()
    summaries, metadata = backend.generate_from_messages(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        trial_config={
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
        },
    )

    assert summaries == ["SUM0", "SUM1"]
    assert metadata["model_sha"] == "sha"


def test_summarize_trials_includes_debug_columns(monkeypatch):
    def mock_run_llm_summarization(trials, config):
        df = pd.DataFrame(
            [
                {
                    "trial_id": "T1",
                    "trial_text": "TEXT",
                    "space_reasoning_and_output": (
                        "assistantfinal\n"
                        "1. Cancer type allowed: A.\n"
                        "Boilerplate exclusions:\n"
                        "Uncontrolled brain metastases."
                    ),
                }
            ]
        )
        return df, {"model_sha": "sha"}

    monkeypatch.setattr(
        "mmai.trials.summarize.run_llm_summarization", mock_run_llm_summarization
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=True,
        backend="local",
        trial={
            "reasoning_marker": "assistantfinal",
            "boilerplate_marker": "Boilerplate exclusions:",
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

    result = summarize_trials(trials, config=config)
    assert "trial_text" in result.columns
    assert "space_reasoning_and_output" in result.columns

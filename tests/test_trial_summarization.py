from unittest.mock import MagicMock
import pandas as pd

from mmai.config import MMAIConfig
from mmai.backends import LocalBackend
from mmai.trials.postprocess import flatten_trial_to_spaces
from mmai.trials.prompt_builder import build_trial_text, get_filled_trial_prompt
from mmai.trials import summarize_trials
from mmai.trials.summarize import run_llm_summarization


def test_build_trial_text_normalizes_whitespace():
    """Ensure trial text normalization collapses whitespace across inputs."""
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
    """Ensure the prompt contains the trial text and has the right structure."""
    prompts = get_filled_trial_prompt(
        "FIND ME", "trial.user.primer.txt", "trial.user.question.txt"
    )
    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "FIND ME" in prompts[1]["content"]


def test_run_llm_summarization_returns_metadata(monkeypatch, default_config):
    """Verify LLM summarization wiring and metadata return."""
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

    df, metadata = run_llm_summarization(trials, default_config)
    assert df["space_reasoning_and_output"].iloc[0] == "SUM0"
    assert "trial_text" in df.columns
    assert metadata["config_snapshot"]["trial"]["model_name"] == "model"
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_run_llm_summarization_preserves_order(monkeypatch, default_config):
    """Ensure LLM outputs are aligned with the input trial order."""
    mock_backend = MagicMock()
    monkeypatch.setattr("mmai.trials.summarize.get_backend", lambda name: mock_backend)

    mock_summarize = MagicMock()
    mock_summarize.return_value = (
        ["SUM0", "SUM1"],
        {"model_sha": "sha"},
    )
    monkeypatch.setattr(
        "mmai.trials.summarize.summarize_trials_multi_cohort", mock_summarize
    )

    trials = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title 1",
                "brief_summary": "Brief 1",
                "eligibility_criteria": "Criteria 1",
            },
            {
                "trial_id": "T2",
                "trial_title": "Title 2",
                "brief_summary": "Brief 2",
                "eligibility_criteria": "Criteria 2",
            },
        ]
    )

    df, _ = run_llm_summarization(trials, default_config)
    assert df["trial_id"].tolist() == ["T1", "T2"]
    assert df["space_reasoning_and_output"].tolist() == ["SUM0", "SUM1"]


def test_flatten_trial_to_spaces(
    mock_summarized_data: pd.DataFrame, expected_flattened_spaces: pd.DataFrame
):
    """Validate postprocessing output against a fixture-based expected DataFrame."""
    result = flatten_trial_to_spaces(
        mock_summarized_data,
        reasoning_marker="assistantfinal",
        boilerplate_marker="Boilerplate exclusions:",
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected_flattened_spaces
    )


def test_local_backend_generate_llm_outputs(monkeypatch, default_trial_config):
    """Ensure our function running vLLM locally return both raw LLM outputs and model metadata."""
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
    summaries, metadata = backend.generate_llm_outputs(
        messages_list=[
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        trial_config=default_trial_config,
    )

    assert summaries == ["SUM0", "SUM1"]
    assert metadata["model_sha"] == "sha"


def test_summarize_trials_includes_debug_columns(monkeypatch):
    """Confirm debug mode adds trial text and raw LLM output columns to the final DataFrame."""

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
        patient={},
        model_metadata_cache_dir=None,
        raw={},
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


def test_summarize_trials_lightweight_integration(monkeypatch):
    """Run summarize_trials end-to-end with a mocked LLM call (backend.generate_llm_outputs)."""

    captured_messages = {}

    class MockBackend:
        def generate_llm_outputs(
            self, *, messages_list, trial_config, model_metadata_cache_dir=None
        ):
            captured_messages["messages_list"] = messages_list
            return (
                [
                    "assistantfinal\n"
                    "1. Cancer type allowed: A.\n"
                    "Boilerplate exclusions:\n"
                    "Uncontrolled brain metastases."
                ],
                {"model_sha": "sha"},
            )

    monkeypatch.setattr("mmai.trials.summarize.get_backend", lambda name: MockBackend())

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

    result = summarize_trials(trials)
    assert len(result) == 1
    assert result["clinical_space_summary"].iloc[0] == "Cancer type allowed: A."
    assert captured_messages["messages_list"][0][1]["role"] == "user"
    assert (
        "Here is a clinical trial document"
        in captured_messages["messages_list"][0][1]["content"]
    )

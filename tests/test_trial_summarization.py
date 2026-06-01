from unittest.mock import MagicMock
import pandas as pd

from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.backends import LocalBackend, build_summarization_runtime_config
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.trials.postprocess import flatten_trial_to_spaces
from matchminer_ai.trials.prompt_builder import (
    build_trial_text,
    get_filled_trial_prompt,
)
from matchminer_ai.trials import summarize_trials
from matchminer_ai.trials.summarize import run_llm_summarization


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
    assert "ECOG performance status" in prompts[1]["content"]
    assert "Boilerplate exclusions:" in prompts[1]["content"]


def test_run_llm_summarization_returns_metadata(monkeypatch, default_config):
    """Verify LLM summarization wiring and metadata return."""
    mock_backend = MagicMock()
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.get_summarization_backend",
        lambda config: mock_backend,
    )

    mock_summarize = MagicMock()
    mock_summarize.return_value = (
        ["SUM0"],
        {"model_sha": "sha"},
        ["stop"],
    )
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.summarize_trials_multi_cohort", mock_summarize
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

    df, metadata, truncated_llm_qc_artifact = run_llm_summarization(
        trials, default_config
    )
    assert df["space_reasoning_and_output"].iloc[0] == "SUM0"
    assert "trial_text" in df.columns
    assert metadata["config_snapshot"]["trial"]["model_name"] == "model"
    assert metadata["model_metadata"]["model_sha"] == "sha"
    assert truncated_llm_qc_artifact["metric"] == "trials_truncated_llm_response"
    assert truncated_llm_qc_artifact["denominator"] == 1
    assert truncated_llm_qc_artifact["numerator"] == 0
    assert truncated_llm_qc_artifact["ids"] == []


def test_run_llm_summarization_preserves_order(monkeypatch, default_config):
    """Ensure LLM outputs are aligned with the input trial order."""
    mock_backend = MagicMock()
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.get_summarization_backend",
        lambda config: mock_backend,
    )

    mock_summarize = MagicMock()
    mock_summarize.return_value = (
        ["SUM0", "SUM1"],
        {"model_sha": "sha"},
        ["stop", "stop"],
    )
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.summarize_trials_multi_cohort", mock_summarize
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

    df, _, truncated_llm_qc_artifact = run_llm_summarization(trials, default_config)
    assert df["trial_id"].tolist() == ["T1", "T2"]
    assert df["space_reasoning_and_output"].tolist() == ["SUM0", "SUM1"]
    assert truncated_llm_qc_artifact["metric"] == "trials_truncated_llm_response"
    assert truncated_llm_qc_artifact["denominator"] == 2
    assert truncated_llm_qc_artifact["numerator"] == 0
    assert truncated_llm_qc_artifact["ids"] == []


def test_flatten_trial_to_spaces(
    mock_summarized_data: pd.DataFrame, expected_flattened_spaces: pd.DataFrame
):
    """Validate postprocessing output against a fixture-based expected DataFrame."""
    result = flatten_trial_to_spaces(
        mock_summarized_data,
        boilerplate_marker="\\n.*Boilerplate.*\\n",
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected_flattened_spaces
    )


def test_flatten_trial_to_spaces_uses_final_output_and_line_boilerplate():
    """Use parsed final text and remove the boilerplate marker line."""
    trial_space = (
        "1. Age: 18+. Sex: Any. Cancer type allowed: A. Histology allowed: Any. "
        "Cancer burden allowed: Any. Prior treatment required: None. "
        "Prior treatment excluded: None. Biomarkers required: None. "
        "Biomarkers excluded: None."
    )
    boilerplate = "Uncontrolled brain metastases."
    df = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_text": "raw input",
                "space_reasoning_and_output": "raw reasoning text",
                "space_output_no_reasoning": (
                    f"{trial_space}\n" "Boilerplate exclusions:\n" f"{boilerplate}"
                ),
            }
        ]
    )

    result = flatten_trial_to_spaces(
        df,
        boilerplate_marker="Boilerplate",
    )

    assert result["clinical_space_summary"].tolist() == [
        "Age: 18+. Sex: Any. Cancer type allowed: A. Histology allowed: Any. "
        "Cancer burden allowed: Any. Prior treatment required: None. "
        "Prior treatment excluded: None. Biomarkers required: None. "
        "Biomarkers excluded: None."
    ]
    assert result["boilerplate_text"].tolist() == [boilerplate]


def test_local_backend_generate_llm_outputs(monkeypatch, default_config):
    """Ensure our function running vLLM locally return both raw LLM outputs and model metadata."""
    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    mock_llm.get_tokenizer.return_value = mock_tokenizer
    mock_tokenizer.apply_chat_template.side_effect = ["p1", "p2"]

    mock_outputs = [MagicMock(), MagicMock()]
    for n, output in enumerate(mock_outputs):
        output.outputs[0].text = f"SUM{n}"
        output.outputs[0].finish_reason = "stop"
    mock_llm.generate.return_value = mock_outputs

    mock_vllm = MagicMock()
    mock_vllm.LLM.return_value = mock_llm
    mock_vllm.SamplingParams = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "vllm", mock_vllm)
    monkeypatch.setattr(
        "matchminer_ai.llm.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_sha": "sha",
            "created_at": "now",
            "last_modified": "now",
        },
    )
    monkeypatch.setattr(
        "matchminer_ai.llm.backends.parse_reasoning_output",
        lambda text, parser_name, tokenizer: (f"REASON {text}", f"FINAL {text}"),
    )

    backend = LocalBackend()
    default_config.local["trial"]["trust_remote_code"] = True
    default_config.local["trial"]["speculative_config"] = {
        "num_speculative_tokens": 4,
    }
    default_config.trial["reasoning_parser"] = "gemma4"
    default_config.trial["sampling_params"]["seed"] = 123
    llm_config = build_summarization_runtime_config(
        "trial",
        default_config.trial,
        config=default_config,
    )
    summaries, metadata, finish_reasons = backend.generate_llm_outputs(
        prompt_list=[
            Prompt(row_idx=0, prompt_text="p1", max_tokens=10),
            Prompt(row_idx=1, prompt_text="p2", max_tokens=10),
        ],
        llm_config=llm_config,
    )

    assert summaries == ["FINAL SUM0", "FINAL SUM1"]
    assert backend.last_raw_outputs == ["SUM0", "SUM1"]
    assert backend.last_reasoning_outputs == ["REASON SUM0", "REASON SUM1"]
    assert metadata["model_sha"] == "sha"
    assert finish_reasons == ["stop", "stop"]
    assert mock_vllm.LLM.call_args.kwargs["trust_remote_code"] is True
    assert mock_vllm.LLM.call_args.kwargs["speculative_config"] == {
        "num_speculative_tokens": 4,
    }
    assert mock_vllm.SamplingParams.call_args.kwargs["seed"] == 123


def test_summarize_trials_includes_debug_columns(monkeypatch):
    """Confirm debug mode adds trial text and raw LLM output columns to the final DataFrame."""

    def mock_run_llm_summarization(trials, config):
        df = pd.DataFrame(
            [
                {
                    "trial_id": "T1",
                    "trial_text": "TEXT",
                    "space_reasoning_and_output": (
                        "1. Age: 18+. Sex: Any. Cancer type allowed: A. "
                        "Histology allowed: Any. Cancer burden allowed: Any. "
                        "Prior treatment required: None. Prior treatment excluded: None. "
                        "Biomarkers required: None. Biomarkers excluded: None.\n"
                        "Boilerplate exclusions:\n"
                        "Uncontrolled brain metastases."
                    ),
                }
            ]
        )
        return (
            df,
            {"model_sha": "sha"},
            {
                "metric": "trials_truncated_llm_response",
                "numerator": 0,
                "denominator": 1,
                "ids": [],
            },
        )

    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.run_llm_summarization",
        mock_run_llm_summarization,
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=True,
        trial={
            "boilerplate_marker": "\\n.*Boilerplate.*\\n",
        },
        patient={},
        local={},
        remote={},
        model_metadata_cache_dir=None,
        raw={},
        embedding={},
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

    captured_prompts = {}

    class MockBackend:
        def generate_llm_outputs(
            self, *, prompt_list, llm_config, model_metadata_cache_dir=None
        ):
            captured_prompts["prompt_list"] = prompt_list
            return (
                [
                    "1. Age: 18+. Sex: Any. Cancer type allowed: A. "
                    "Histology allowed: Any. Cancer burden allowed: Any. "
                    "Prior treatment required: None. Prior treatment excluded: None. "
                    "Biomarkers required: None. Biomarkers excluded: None.\n"
                    "Boilerplate exclusions:\n"
                    "Uncontrolled brain metastases."
                ],
                {"model_sha": "sha"},
                ["stop"],
            )

    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.build_prompt_list",
        lambda messages_list, llm_config: [
            Prompt(
                row_idx=idx,
                prompt_text=f"Here is a clinical trial document: {messages[-1]['content']}",
                max_tokens=10,
            )
            for idx, messages in enumerate(messages_list)
        ],
    )
    monkeypatch.setattr(
        "matchminer_ai._qc.trials.count_embedding_tokens",
        lambda texts, embedding_config: [1 for _text in texts],
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

    result = summarize_trials(trials)
    assert len(result) == 1
    assert (
        result["clinical_space_summary"].iloc[0]
        == "Age: 18+. Sex: Any. Cancer type allowed: A. Histology allowed: Any. "
        "Cancer burden allowed: Any. Prior treatment required: None. "
        "Prior treatment excluded: None. Biomarkers required: None. "
        "Biomarkers excluded: None."
    )
    assert (
        "Here is a clinical trial document"
        in captured_prompts["prompt_list"][0].prompt_text
    )


def test_summarize_trials_metadata_uses_live_config(monkeypatch, default_config):
    """Return metadata from the live config object after caller mutations."""

    class MockBackend:
        def generate_llm_outputs(
            self, *, prompt_list, llm_config, model_metadata_cache_dir=None
        ):
            return (
                [
                    "1. Age: 18+. Sex: Any. Cancer type allowed: A. "
                    "Histology allowed: Any. Cancer burden allowed: Any. "
                    "Prior treatment required: None. Prior treatment excluded: None. "
                    "Biomarkers required: None. Biomarkers excluded: None.\n"
                    "Boilerplate exclusions:\n"
                    "None."
                ],
                {"model_sha": "sha"},
                ["stop"],
            )

    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.build_prompt_list",
        lambda messages_list, llm_config: [
            Prompt(row_idx=idx, prompt_text="prompt", max_tokens=10)
            for idx, _messages in enumerate(messages_list)
        ],
    )

    default_config.raw = {"remote": {"enabled": False}}
    default_config.remote["enabled"] = True
    default_config.remote["server_urls"] = ["http://localhost:8000/v1"]

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

    _result, metadata = summarize_trials(
        trials,
        config=default_config,
        return_metadata=True,
    )

    assert metadata["config_snapshot"]["remote"]["enabled"] is True
    assert metadata["config_snapshot"]["remote"]["server_urls"] == [
        "http://localhost:8000/v1"
    ]

from unittest.mock import MagicMock

import pandas as pd

from mmai.backends import LocalBackend
from mmai.config import MMAIConfig
from mmai.patients import summarize_from_relevant_sentences, summarize_patients
from mmai.patients.postprocess import parse_boilerplate


def test_summarize_from_relevant_sentences_filters_empty_and_returns_metadata(
    monkeypatch,
):
    """Filter empty patient text rows and return summaries plus metadata."""

    class MockBackend:
        def truncate_texts(self, texts, *, patient_config):
            return texts

        def generate_llm_outputs(
            self, *, messages_list, llm_config, model_metadata_cache_dir=None
        ):
            assert len(messages_list) == 1
            return (
                [
                    "assistantfinal\n"
                    "Cancer summary.\n"
                    "Boilerplate exclusions:\n"
                    "No CNS mets."
                ],
                {"model_name": "model", "model_sha": "sha"},
            )

    monkeypatch.setattr(
        "mmai.patients.summarize.get_backend", lambda name: MockBackend()
    )
    monkeypatch.setattr(
        "mmai.patients.summarize.get_filled_patient_prompt",
        lambda text, primer_filename, question_filename: [
            {"role": "user", "content": text}
        ],
    )

    df = pd.DataFrame(
        [
            {"patient_id": "P1", "patient_long_text": ""},
            {"patient_id": "P2", "patient_long_text": pd.NA},
            {"patient_id": "P3", "patient_long_text": "keep me"},
        ]
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=True,
        backend="local",
        trial={},
        patient={
            "model_name": "model",
            "prompt_files": {
                "primer": "patient.user.primer.txt",
                "question": "patient.user.question.txt",
            },
            "reasoning_marker": "assistantfinal",
            "boilerplate_marker": "Boilerplate exclusions:",
            "text_token_threshold": 100,
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 10,
                "repetition_penalty": 1.0,
            },
            "max_model_len": 100,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
        },
        model_metadata_cache_dir=None,
        raw={"config": "snapshot"},
    )

    result, metadata = summarize_from_relevant_sentences(df, config=config)

    assert len(result) == 1
    assert result.loc[result.index[0], "cancer_history_summary"] == "Cancer summary."
    assert (
        result.loc[result.index[0], "general_exclusion_criteria_evidence"]
        == "No CNS mets."
    )
    assert metadata["model_metadata"]["model_sha"] == "sha"
    assert metadata["config_snapshot"] == {"config": "snapshot"}


def test_parse_boilerplate_splits_summary_and_exclusions():
    """Split patient summaries into cancer history vs exclusion evidence."""
    df = pd.DataFrame(
        [
            {
                "original_patient_summary": (
                    "assistantfinal\n"
                    "Cancer history here.\n"
                    "Boilerplate exclusions:\n"
                    "No CNS mets."
                )
            },
            {"original_patient_summary": "assistantfinal\nCancer only."},
        ]
    )

    parsed = parse_boilerplate(
        df,
        reasoning_marker="assistantfinal",
        boilerplate_marker="Boilerplate exclusions:",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert parsed.loc[0, "general_exclusion_criteria_evidence"] == "No CNS mets."
    assert parsed.loc[1, "general_exclusion_criteria_evidence"] == "Cancer only."


def test_local_backend_truncate_texts_splits_long_inputs(monkeypatch):
    """Truncate long patient text using the tokenizer without loading real models."""

    class MockTokenResult:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class MockTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return MockTokenResult(list(text))

        def decode(self, input_ids):
            return "".join(input_ids)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MockTokenizer()
    monkeypatch.setitem(__import__("sys").modules, "transformers", mock_transformers)

    backend = LocalBackend()
    truncated = backend.truncate_texts(
        ["abcdefghij"],
        patient_config={
            "model_name": "mock",
            "text_token_threshold": 6,
        },
    )

    assert truncated == ["abc ... hij"]


def test_summarize_patients_joins_metadata(monkeypatch):
    """Combine tagger + summarizer metadata when return_metadata=True."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )

    relevant_df = pd.DataFrame([{"patient_id": "P1", "patient_long_text": "Long text"}])
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )

    monkeypatch.setattr(
        "mmai.patients.__init__.extract_relevant_sentences",
        MagicMock(
            return_value=(relevant_df, {"model_metadata": {"model_name": "tag"}})
        ),
    )
    monkeypatch.setattr(
        "mmai.patients.__init__.summarize_from_relevant_sentences",
        MagicMock(
            return_value=(summaries_df, {"model_metadata": {"model_name": "summ"}})
        ),
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={"config": "snapshot"},
    )

    result, metadata = summarize_patients(notes, config=config, return_metadata=True)

    assert result.equals(summaries_df)
    assert metadata["config_snapshot"] == {"config": "snapshot"}
    assert metadata["model_metadata"]["patient_tagger"]["model_name"] == "tag"
    assert metadata["model_metadata"]["patient_summarizer"]["model_name"] == "summ"


def test_summarize_patients_includes_debug_columns(monkeypatch):
    """Confirm debug mode retains patient long text and raw summary columns."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )

    relevant_df = pd.DataFrame([{"patient_id": "P1", "patient_long_text": "Long text"}])
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "patient_long_text": "Long text",
                "original_patient_summary": "assistantfinal\nSummary",
            }
        ]
    )

    monkeypatch.setattr(
        "mmai.patients.__init__.extract_relevant_sentences",
        MagicMock(
            return_value=(relevant_df, {"model_metadata": {"model_name": "tag"}})
        ),
    )
    monkeypatch.setattr(
        "mmai.patients.__init__.summarize_from_relevant_sentences",
        MagicMock(
            return_value=(summaries_df, {"model_metadata": {"model_name": "summ"}})
        ),
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=True,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={},
    )

    result = summarize_patients(notes, config=config)

    assert "patient_long_text" in result.columns
    assert "original_patient_summary" in result.columns


def test_summarize_patients_lightweight_integration(monkeypatch):
    """Run summarize_patients end-to-end with mocked tagger + summarizer outputs."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )

    relevant_df = pd.DataFrame([{"patient_id": "P1", "patient_long_text": "Long text"}])
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )

    mock_extract = MagicMock(
        return_value=(relevant_df, {"model_metadata": {"model_name": "tag"}})
    )
    mock_summarize = MagicMock(
        return_value=(summaries_df, {"model_metadata": {"model_name": "summ"}})
    )
    monkeypatch.setattr(
        "mmai.patients.__init__.extract_relevant_sentences", mock_extract
    )
    monkeypatch.setattr(
        "mmai.patients.__init__.summarize_from_relevant_sentences", mock_summarize
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={},
    )

    result = summarize_patients(notes, config=config)

    mock_extract.assert_called_once()
    mock_summarize.assert_called_once()
    assert result.equals(summaries_df)

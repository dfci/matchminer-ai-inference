from unittest.mock import MagicMock

import pandas as pd

from mmai.backends import LocalBackend
from mmai.config import MMAIConfig
from mmai.patients import summarize_from_relevant_sentences, summarize_patients
from mmai.patients.postprocess import clean_bad_data, parse_boilerplate


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
                ["stop"],
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
        embedding={},
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


def test_summarize_from_relevant_sentences_returns_qc_report(monkeypatch):
    """Return a QC report when requested for patient summarization."""

    class MockBackend:
        def truncate_texts(self, texts, *, patient_config):
            return texts

        def generate_llm_outputs(
            self, *, messages_list, llm_config, model_metadata_cache_dir=None
        ):
            return (
                [
                    "assistantfinal\n"
                    "No information\n"
                    "Boilerplate exclusions:\n"
                    "None"
                ],
                {"model_name": "model", "model_sha": "sha"},
                ["length"],
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

    df = pd.DataFrame([{"patient_id": "P1", "patient_long_text": "keep me"}])
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
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
        embedding={},
    )

    result, metadata, qc_artifacts = summarize_from_relevant_sentences(
        df,
        config=config,
        return_qc=True,
    )

    assert result.empty
    assert metadata["model_metadata"]["model_sha"] == "sha"
    qc_report = qc_artifacts.set_index("metric")
    assert qc_report.loc["patients_dropped_noninformative_summary", "ids"] == ["P1"]


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


def test_clean_bad_data_filters_invalid_summaries():
    """Drop rows with empty or invalid patient summaries."""
    df = pd.DataFrame(
        [
            {"cancer_history_summary": "No evidence of malignancy"},
            {"cancer_history_summary": "No primary identified"},
            {"cancer_history_summary": "No information"},
            {"cancer_history_summary": "Valid summary"},
        ]
    )

    cleaned, qc_artifact = clean_bad_data(df)

    assert cleaned["cancer_history_summary"].tolist() == ["Valid summary"]
    assert qc_artifact["metric"] == "patients_dropped_noninformative_summary"


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

    tagger_qc = pd.DataFrame(
        [
            {
                "metric": "patients_with_no_tagged_notes",
                "value": 1,
                "percent": 50.0,
                "ids": ["P2"],
            }
        ]
    )
    monkeypatch.setattr(
        "mmai.patients.extract_relevant_sentences",
        MagicMock(
            return_value=(
                relevant_df,
                {"model_metadata": {"model_name": "tag"}},
                tagger_qc,
            )
        ),
    )
    monkeypatch.setattr(
        "mmai.patients.summarize_from_relevant_sentences",
        MagicMock(
            return_value=(
                summaries_df,
                {"model_metadata": {"model_name": "summ"}},
                pd.DataFrame(
                    [
                        {
                            "metric": "patients_dropped_noninformative_summary",
                            "value": 0,
                            "percent": 0.0,
                            "ids": [],
                        }
                    ]
                ),
            )
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
        embedding={},
    )

    result, metadata = summarize_patients(notes, config=config, return_metadata=True)

    assert result.equals(summaries_df)
    assert metadata["config_snapshot"] == {"config": "snapshot"}
    assert metadata["model_metadata"]["patient_tagger"]["model_name"] == "tag"
    assert metadata["model_metadata"]["patient_summarizer"]["model_name"] == "summ"


def test_summarize_patients_returns_qc_report(monkeypatch):
    """Return a QC report when return_qc is enabled."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            },
            {
                "patient_id": "P2",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-02",
            },
        ]
    )

    relevant_df = pd.DataFrame(
        [
            {"patient_id": "P1", "patient_long_text": "Long text"},
            {"patient_id": "P2", "patient_long_text": ""},
        ]
    )
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )

    tagger_qc = pd.DataFrame(
        [
            {
                "metric": "patients_with_no_tagged_notes",
                "value": 0,
                "percent": 0.0,
                "ids": [],
            }
        ]
    )
    monkeypatch.setattr(
        "mmai.patients.extract_relevant_sentences",
        MagicMock(
            return_value=(
                relevant_df,
                {"model_metadata": {"model_name": "tag"}},
                tagger_qc,
            )
        ),
    )
    qc_report = pd.DataFrame(
        [
            {
                "metric": "patients_dropped_noninformative_summary",
                "value": 1,
                "percent": 50.0,
                "ids": ["P2"],
            }
        ]
    )
    monkeypatch.setattr(
        "mmai.patients.summarize_from_relevant_sentences",
        MagicMock(
            return_value=(
                summaries_df,
                {"model_metadata": {"model_name": "summ"}},
                qc_report,
            )
        ),
    )
    monkeypatch.setattr(
        "mmai._qc.patients.get_backend",
        MagicMock(
            return_value=MagicMock(
                count_embedding_tokens=MagicMock(return_value=[100]),
            )
        ),
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={},
        embedding={
            "model_path": "mock-model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
        },
    )

    result, qc_report = summarize_patients(notes, config=config, return_qc=True)

    assert result.equals(summaries_df)
    report = qc_report.set_index("metric")
    assert report.loc["patients_dropped_noninformative_summary", "value"] == 1
    assert report.loc["patients_dropped_noninformative_summary", "ids"] == ["P2"]


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
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )

    tagger_qc = pd.DataFrame(
        [
            {
                "metric": "patients_with_no_tagged_notes",
                "value": 0,
                "percent": 0.0,
                "ids": [],
            }
        ]
    )
    monkeypatch.setattr(
        "mmai.patients.extract_relevant_sentences",
        MagicMock(
            return_value=(
                relevant_df,
                {"model_metadata": {"model_name": "tag"}},
                tagger_qc,
            )
        ),
    )
    summary_qc = pd.DataFrame(
        [
            {
                "metric": "patients_dropped_noninformative_summary",
                "value": 0,
                "percent": 0.0,
                "ids": [],
            }
        ]
    )
    monkeypatch.setattr(
        "mmai.patients.summarize_from_relevant_sentences",
        MagicMock(
            return_value=(
                summaries_df,
                {"model_metadata": {"model_name": "summ"}},
                summary_qc,
            )
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
        embedding={},
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
        return_value=(
            relevant_df,
            {"model_metadata": {"model_name": "tag"}},
            pd.DataFrame(
                [
                    {
                        "metric": "patients_with_no_tagged_notes",
                        "value": 0,
                        "percent": 0.0,
                        "ids": [],
                    }
                ]
            ),
        )
    )
    mock_summarize = MagicMock(
        return_value=(
            summaries_df,
            {"model_metadata": {"model_name": "summ"}},
            pd.DataFrame(
                [
                    {
                        "metric": "patients_dropped_noninformative_summary",
                        "value": 0,
                        "percent": 0.0,
                        "ids": [],
                    }
                ]
            ),
        )
    )
    monkeypatch.setattr("mmai.patients.extract_relevant_sentences", mock_extract)
    monkeypatch.setattr(
        "mmai.patients.summarize_from_relevant_sentences", mock_summarize
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={},
        embedding={},
    )

    result = summarize_patients(notes, config=config)

    mock_extract.assert_called_once()
    mock_summarize.assert_called_once()
    assert result.equals(summaries_df)

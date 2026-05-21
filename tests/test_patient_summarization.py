import asyncio
from unittest.mock import MagicMock

import pandas as pd

from matchminer_ai.llm.backends import LocalBackend
from matchminer_ai.config import MMAIConfig
from matchminer_ai.patients import summarize_patients
from matchminer_ai.patients.postprocess import (
    clean_bad_data,
    parse_boilerplate,
    split_reasoning_from_summary,
)
from matchminer_ai.patients.prompt_builder import get_serial_patient_prompt
from matchminer_ai.patients.summarize import summarize_patient_notes
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.llm.remote_inference import generate_remote_llm_outputs


class MockTokenResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return MockTokenResult(list(text))

    def decode(self, input_ids, skip_special_tokens=True):
        return "".join(input_ids)


def _stub_patient_qc(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai._qc.patients.patient_summary_qc_report",
        lambda *args, **kwargs: pd.DataFrame(),
    )


def _patient_config() -> dict:
    return {
        "model_name": "model",
        "prompt_files": {
            "primer": "patient.serial.user.primer.txt",
            "question": "patient.serial.user.question.txt",
        },
        "reasoning_marker": "assistantfinal",
        "boilerplate_marker": "\\n.*Boilerplate.*\\n",
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 10,
            "repetition_penalty": 1.0,
        },
        "chunk_size": 50,
        "chunk_overlap": 5,
        "prompt_margin_tokens": 10,
    }


def _config(debug_mode: bool = False) -> MMAIConfig:
    return MMAIConfig(
        preset_name="default",
        debug_mode=debug_mode,
        trial={},
        patient=_patient_config(),
        local={
            "patient": {
                "max_model_len": 100,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        },
        remote={},
        model_metadata_cache_dir=None,
        raw={"config": "snapshot"},
        embedding={
            "model_path": "mock-model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
        },
    )


def _remote_config(debug_mode: bool = False) -> MMAIConfig:
    config = _config(debug_mode=debug_mode)
    config.remote = {
        "enabled": True,
        "server_urls": ["http://server-a/v1"],
        "max_concurrent_requests": 2,
        "request_timeout": 123,
        "max_retries": 2,
        "batch_size": 1000,
    }
    config.patient = {
        **config.patient,
        "prompt_build_workers": 2,
    }
    return config


def test_split_reasoning_from_summary_matches_original_serial_script():
    """Split raw model output before using it as serial prior summary state."""
    reasoning, summary = split_reasoning_from_summary(
        "analysis text\nassistantfinal\nAge: 70\nBoilerplate:\nNone",
        "assistantfinal",
    )

    assert reasoning == "analysis text"
    assert summary == "Age: 70\nBoilerplate:\nNone"


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


def test_get_serial_patient_prompt_includes_prior_summary_and_chunk_text():
    """Build a serial prompt containing prior summary state and the next note chunk."""
    prompts = get_serial_patient_prompt(
        prior_summary="Age: 70",
        first_date="2024-01-01",
        last_date="2024-01-02",
        chunk_text="Clinical note text.",
        tokenizer=MockTokenizer(),
        max_model_len=100,
        primer_filename="patient.serial.user.primer.txt",
        question_filename="patient.serial.user.question.txt",
        margin_tokens=10,
        model_name="openai/gpt-oss-120b",
    )

    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "Age: 70" in prompts[1]["content"]
    assert "Clinical note text." in prompts[1]["content"]


def test_summarize_patient_notes_updates_running_summary_across_rounds(monkeypatch):
    """Carry each round's summary forward as prior state for the next chunk."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame([{"patient_id": "P1", "last_note_date": "2024-01-02"}]),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk one",
                    },
                    {
                        "patient_id": "P1",
                        "chunk_index": 1,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-02",
                        "chunk_text": "chunk two",
                    },
                ]
            ),
        ),
    )

    seen_prior_summaries = []

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            seen_prior_summaries.extend(item.prior_summary_text for item in work_items)
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )

    class MockBackend:
        def __init__(self):
            self.calls = 0

        def generate_llm_outputs(
            self,
            *,
            prompt_list,
            llm_config,
            model_metadata_cache_dir=None,
        ):
            self.calls += 1
            if self.calls == 1:
                return (
                    ["assistantfinal\nRound 1\nBoilerplate:\nNone"],
                    {"model_name": "model", "model_sha": "sha"},
                    ["stop"],
                )
            return (
                ["assistantfinal\nRound 2\nBoilerplate:\nNone"],
                {"model_name": "model", "model_sha": "sha"},
                ["stop"],
            )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    result, metadata = summarize_patient_notes(notes, config=_config())

    assert seen_prior_summaries == [None, "Round 1\nBoilerplate:\nNone"]
    assert result.loc[result.index[0], "cancer_history_summary"] == "Round 2"
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_summarize_patient_notes_uses_existing_summary_in_first_round(monkeypatch):
    """Use a provided existing summary as the starting state for round 1."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame([{"patient_id": "P1", "last_note_date": "2024-01-02"}]),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-02",
                        "chunk_text": "new chunk",
                    }
                ]
            ),
        ),
    )

    seen_prior_summaries = []

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            seen_prior_summaries.extend(item.prior_summary_text for item in work_items)
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=(
                    ["assistantfinal\nUpdated\nBoilerplate:\nNone"],
                    {"model_name": "model", "model_sha": "sha"},
                    ["stop"],
                )
            )
        ),
    )

    existing_summaries = pd.DataFrame(
        [{"patient_id": "P1", "patient_summary": "Existing summary"}]
    )
    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-02"}]
    )

    result, _ = summarize_patient_notes(
        notes,
        config=_config(),
        existing_summaries=existing_summaries,
    )

    assert seen_prior_summaries == ["Existing summary"]
    assert result.loc[result.index[0], "cancer_history_summary"] == "Updated"


def test_remote_summarize_patient_notes_uses_parallel_prompt_workers(monkeypatch):
    """Remote patient summarization builds pre-rendered prompts via prompt pool."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame(
                [
                    {"patient_id": "P1", "last_note_date": "2024-01-01"},
                    {"patient_id": "P2", "last_note_date": "2024-01-01"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk one",
                    },
                    {
                        "patient_id": "P2",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk two",
                    },
                ]
            ),
        ),
    )

    pool_calls = {}

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            pool_calls["chunksize"] = chunksize
            pool_calls["work_items"] = work_items
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: pool_calls.setdefault("shutdown", True),
    )

    captured = {}

    class MockBackend:
        def generate_llm_outputs(
            self,
            *,
            prompt_list,
            llm_config,
            model_metadata_cache_dir=None,
        ):
            captured["prompt_list"] = prompt_list
            return (
                [
                    "assistantfinal\nRemote 1\nBoilerplate:\nNone",
                    "assistantfinal\nRemote 2\nBoilerplate:\nNone",
                ],
                {"model_name": "model", "model_sha": "sha"},
                ["stop", "stop"],
            )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    result, metadata = summarize_patient_notes(notes, config=_remote_config())

    assert [prompt.prompt_text for prompt in captured["prompt_list"]] == [
        "chunk one",
        "chunk two",
    ]
    assert [item.prior_summary_text for item in pool_calls["work_items"]] == [
        None,
        None,
    ]
    assert pool_calls["shutdown"] is True
    assert result["cancer_history_summary"].tolist() == ["Remote 1", "Remote 2"]
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_generate_remote_llm_outputs_handles_running_event_loop(monkeypatch):
    """Allow the sync remote wrapper to run from notebooks and async shells."""

    async def fake_generate_remote_llm_outputs_async(
        *,
        prompts,
        llm_config,
        server_urls,
        api_key,
    ):
        return ["assistantfinal\nSummary"], ["stop"]

    monkeypatch.setattr(
        "matchminer_ai.llm.remote_inference.generate_remote_llm_outputs_async",
        fake_generate_remote_llm_outputs_async,
    )

    async def invoke_wrapper():
        return generate_remote_llm_outputs(
            prompts=[Prompt(row_idx=0, prompt_text="chunk", max_tokens=7)],
            llm_config={"model_name": "model"},
            server_urls=["http://server-a/v1"],
            api_key="not-needed",
        )

    texts, finish_reasons = asyncio.run(invoke_wrapper())

    assert texts == ["assistantfinal\nSummary"]
    assert finish_reasons == ["stop"]


def test_summarize_patients_returns_metadata_and_qc(monkeypatch):
    """Return metadata and QC from the serial patient summarization entrypoint."""
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
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )
    qc_report = pd.DataFrame(
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
        "matchminer_ai.patients.summarize_patient_notes",
        MagicMock(
            return_value=(
                summaries_df,
                {"model_metadata": {"model_name": "summ", "model_sha": "sha"}},
                qc_report,
            )
        ),
    )

    result, metadata, returned_qc = summarize_patients(
        notes,
        config=_config(),
        return_metadata=True,
        return_qc=True,
    )

    assert result.equals(summaries_df)
    assert returned_qc.equals(qc_report)
    assert metadata["config_snapshot"]["config"] == "snapshot"
    assert metadata["config_snapshot"]["patient"] == _config().patient
    assert metadata["model_metadata"]["patient_summarizer"]["model_sha"] == "sha"

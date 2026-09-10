import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.backends import LLMGenerationResult, LocalBackend
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.llm.remote_inference import generate_remote_llm_outputs
from matchminer_ai.patients import summarize_patients
from matchminer_ai.patients.checkpoints import (
    load_prepared_chunks,
    load_round_checkpoints,
    save_prepared_chunks,
    save_round_checkpoint,
)
from matchminer_ai.patients.postprocess import parse_boilerplate
from matchminer_ai.patients.prompt_builder import (
    PromptWorkItem,
    _RESPONSE_TOKEN_MARGIN,
    build_prompt_worker,
    get_serial_patient_prompt,
)
from matchminer_ai.patients.summarize import summarize_patient_notes


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
        "local": {
            "model_name": "model",
            "engine": {
                "max_model_len": 100,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            },
            "generation": {
                "temperature": 0.0,
                "top_k": 1,
                "max_tokens": 10,
                "repetition_penalty": 1.0,
            },
            "chat_template_kwargs": {},
        },
        "remote": {
            "model_name": "model",
            "tokenizer_name": "model",
            "request_params": {
                "max_tokens": 10,
                "temperature": 0.0,
            },
            "extra_body": {
                "top_k": 1,
                "repetition_penalty": 1.0,
            },
        },
        "prompt_files": {
            "primer": "patient.serial.user.primer.txt",
            "question": "patient.serial.user.question.txt",
        },
        "boilerplate_marker": "Boilerplate conditions",
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
        local={},
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


def test_patient_checkpoint_helpers_round_trip_dataframes(tmp_path):
    """Checkpoint helpers should create their directory and restore saved data."""
    checkpoint_dir = tmp_path / "patient-checkpoints"
    prepared_chunks = pd.DataFrame(
        [{"patient_id": "P1", "chunk_index": 0, "chunk_text": "note chunk"}]
    )
    round_results = pd.DataFrame([{"patient_id": "P1", "summary": "updated summary"}])

    assert load_prepared_chunks(checkpoint_dir) is None
    assert load_round_checkpoints(checkpoint_dir) == {}

    save_prepared_chunks(checkpoint_dir, prepared_chunks)
    save_round_checkpoint(checkpoint_dir, 0, round_results)

    pd.testing.assert_frame_equal(
        load_prepared_chunks(checkpoint_dir),
        prepared_chunks,
    )
    loaded_rounds = load_round_checkpoints(checkpoint_dir)
    assert list(loaded_rounds) == [0]
    pd.testing.assert_frame_equal(loaded_rounds[0], round_results)


def test_parse_boilerplate_splits_summary_and_exclusions():
    """Split patient summaries into cancer history vs exclusion evidence."""
    df = pd.DataFrame(
        [
            {
                "patient_answer_text": (
                    "Cancer history here.\n" "Boilerplate conditions:\n" "No CNS mets."
                )
            },
            {"patient_answer_text": "Cancer only."},
        ]
    )

    parsed = parse_boilerplate(
        df,
        boilerplate_marker="Boilerplate conditions",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert parsed.loc[0, "general_exclusion_criteria_evidence"] == "No CNS mets."
    assert parsed.loc[1, "general_exclusion_criteria_evidence"] == "Cancer only."


def test_parse_boilerplate_accepts_final_only_v22_output():
    """Parsed final content should not need a reasoning marker."""
    df = pd.DataFrame(
        [
            {
                "patient_answer_text": (
                    "Cancer history here.\n"
                    "\n"
                    "Boilerplate conditions:\n"
                    "Remote inactive prostate cancer."
                )
            }
        ]
    )

    parsed = parse_boilerplate(
        df,
        boilerplate_marker="Boilerplate conditions",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert (
        parsed.loc[0, "general_exclusion_criteria_evidence"]
        == "Remote inactive prostate cancer."
    )


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
        model_name="google/gemma-4-31B-it",
    )

    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "Age: 70" in prompts[1]["content"]
    assert "Clinical note text." in prompts[1]["content"]
    assert "Boilerplate conditions:" in prompts[1]["content"]
    assert "contradictory information across notes" in prompts[1]["content"]


def test_build_prompt_worker_leaves_response_token_margin(monkeypatch):
    """Leave a small generation margin for remote chat-template token drift."""

    class FixedPromptTokenizer(MockTokenizer):
        def apply_chat_template(
            self,
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        ):
            return "x" * 600

    monkeypatch.setattr(
        "matchminer_ai.patients.prompt_builder._worker_tokenizer",
        FixedPromptTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.prompt_builder._worker_config",
        {
            **_patient_config(),
            "model_name": "google/gemma-4-31B-it",
            "max_model_len": 1000,
            "sampling_params": {
                **_patient_config()["local"]["generation"],
                "max_tokens": 900,
            },
        },
    )

    prompt = build_prompt_worker(
        PromptWorkItem(
            row_idx=0,
            prior_summary_text=None,
            first_date="2024-01-01",
            last_date="2024-01-02",
            chunk_text="Clinical note text.",
        )
    )

    assert prompt.max_tokens == 1000 - 600 - _RESPONSE_TOKEN_MARGIN


def test_summarize_patient_notes_resumes_after_completed_round(
    caplog,
    monkeypatch,
    tmp_path,
):
    """Retry from the saved summary when a later inference round fails."""
    caplog.set_level("INFO")
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: pd.DataFrame(
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
        fail_second_round = True

        def generate_llm_outputs(
            self,
            *,
            prompt_list,
            llm_config,
            model_metadata_cache_dir=None,
        ):
            if prompt_list[0].prompt_text == "chunk one":
                return LLMGenerationResult(
                    final_outputs=["Round 1\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=[""],
                    raw_outputs=[],
                )
            if self.fail_second_round:
                raise RuntimeError("interrupted during round two")
            return LLMGenerationResult(
                final_outputs=["Round 2\nBoilerplate conditions:\nNone"],
                model_metadata={"model_name": "model", "model_sha": "sha"},
                finish_reasons=["stop"],
                reasoning_outputs=[""],
                raw_outputs=[],
            )

    backend = MockBackend()
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_llm_backend",
        lambda config: backend,
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    with pytest.raises(RuntimeError, match="interrupted during round two"):
        summarize_patient_notes(
            notes,
            config=_config(),
            checkpoint_dir=tmp_path,
        )

    assert (tmp_path / "round_0000.parquet").exists()
    assert not (tmp_path / "round_0001.parquet").exists()

    backend.fail_second_round = False
    seen_prior_summaries.clear()
    result, metadata = summarize_patient_notes(
        notes,
        config=_config(),
        checkpoint_dir=tmp_path,
    )

    assert seen_prior_summaries == ["Round 1\nBoilerplate conditions:\nNone"]
    assert result.loc[result.index[0], "cancer_history_summary"] == "Round 2"
    assert metadata["model_metadata"]["model_sha"] == "sha"
    assert (tmp_path / "prepared_chunks.parquet").exists()
    assert (tmp_path / "round_0001.parquet").exists()
    assert "Loaded 1 completed patient summarization round(s)" in caplog.text
    assert "Resuming patient summarization at round 2 of 2" in caplog.text

    caplog.clear()
    summarize_patient_notes(
        notes,
        config=_config(),
        checkpoint_dir=tmp_path,
    )

    assert "All 2 patient summarization round(s) are already complete" in caplog.text


def test_summarize_patient_notes_reuses_checkpointed_chunks(monkeypatch, tmp_path):
    """A prepared-chunks checkpoint should bypass note preparation on retry."""
    prepared_chunks = pd.DataFrame(
        columns=[
            "patient_id",
            "chunk_index",
            "first_date",
            "last_date",
            "chunk_text",
        ]
    )
    save_prepared_chunks(tmp_path, prepared_chunks)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        MagicMock(side_effect=AssertionError("tokenizer should not be loaded")),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        MagicMock(side_effect=AssertionError("notes should not be prepared")),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    result, metadata = summarize_patient_notes(
        notes,
        config=_config(),
        checkpoint_dir=tmp_path,
    )

    assert result.empty
    assert metadata["model_metadata"] == {}


def test_summarize_patient_notes_uses_existing_summary_in_first_round(monkeypatch):
    """Use a provided existing summary as the starting state for round 1."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: pd.DataFrame(
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
        "matchminer_ai.patients.summarize.get_llm_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=LLMGenerationResult(
                    final_outputs=["Updated\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=[""],
                    raw_outputs=[],
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


def test_summarize_patient_notes_includes_standard_debug_columns(monkeypatch):
    """Debug mode exposes standardized patient LLM answer/reasoning columns."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "chunk_index": 0,
                    "first_date": "2024-01-01",
                    "last_date": "2024-01-01",
                    "chunk_text": "chunk",
                }
            ]
        ),
    )

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
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
        "matchminer_ai.patients.summarize.get_llm_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=LLMGenerationResult(
                    final_outputs=["Debug summary\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=["debug reasoning"],
                    raw_outputs=["raw output"],
                )
            )
        ),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )

    result, _ = summarize_patient_notes(notes, config=_config(debug_mode=True))

    assert result["patient_answer_text"].tolist() == [
        "Debug summary\nBoilerplate conditions:\nNone"
    ]
    assert result["patient_reasoning_text"].tolist() == ["debug reasoning"]


def test_remote_summarize_patient_notes_uses_parallel_prompt_workers(monkeypatch):
    """Remote patient summarization builds pre-rendered prompts via prompt pool."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: pd.DataFrame(
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
            return LLMGenerationResult(
                final_outputs=[
                    "Remote 1\nBoilerplate conditions:\nNone",
                    "Remote 2\nBoilerplate conditions:\nNone",
                ],
                model_metadata={"model_name": "model", "model_sha": "sha"},
                finish_reasons=["stop", "stop"],
                reasoning_outputs=["", ""],
                raw_outputs=[],
            )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_llm_backend",
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
        return ["Summary"], ["thinking"], ["stop"]

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

    texts, reasonings, finish_reasons = asyncio.run(invoke_wrapper())

    assert texts == ["Summary"]
    assert reasonings == ["thinking"]
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
                "metric": "patients_exclusion_criteria_not_extracted",
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


def test_summarize_patients_does_not_request_qc_by_default(monkeypatch, tmp_path):
    """Default patient summarization should skip QC-only embedding token counts."""
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
    summarize_mock = MagicMock(
        return_value=(
            summaries_df,
            {"model_metadata": {"model_name": "summ", "model_sha": "sha"}},
        )
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize_patient_notes",
        summarize_mock,
    )

    result = summarize_patients(
        notes,
        config=_config(),
        checkpoint_dir=tmp_path,
    )

    assert result.equals(summaries_df)
    assert summarize_mock.call_args.kwargs["return_qc"] is False
    assert summarize_mock.call_args.kwargs["checkpoint_dir"] == tmp_path

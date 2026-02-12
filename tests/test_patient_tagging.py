from unittest.mock import MagicMock

import pandas as pd

from mmai.config import MMAIConfig
from mmai.backends import LocalBackend
from mmai.patients.tagging import (
    _extract_relevant_text_from_notes,
    _format_relevant_text,
    split_sentences_and_dedup_notes,
    extract_relevant_sentences,
    extract_relevant_text_from_patient,
)


def test_format_relevant_text_groups_excerpts():
    """Ensure excerpt grouping preserves note_date and note_type ordering."""
    excerpts_frame = pd.DataFrame(
        [
            {"note_date": "D1", "note_type": "foo", "excerpt": "bar"},
            {"note_date": "D1", "note_type": "foo", "excerpt": "baz"},
            {"note_date": "D1", "note_type": "car", "excerpt": "biz"},
            {"note_date": "D2", "note_type": "car", "excerpt": ""},
        ]
    )

    expected = pd.Series(
        {
            "patient_id": "mrn",
            "patient_long_text": "\n".join(
                [
                    "D1 car biz",
                    "D1 foo bar. baz",
                ]
            ),
        }
    )

    pd.testing.assert_series_equal(
        _format_relevant_text(patient_id="mrn", excerpts_frame=excerpts_frame),
        expected,
    )


def test_extract_relevant_text_from_notes_filters_by_cutoffs():
    """Ensure tagger cutoffs filter excerpts and metadata is preserved."""
    excerpts_frame = pd.DataFrame(
        [
            {"excerpt": "A", "note_date": 0, "note_type": "foo"},
            {"excerpt": "B", "note_date": 1, "note_type": "foo"},
            {"excerpt": "C", "note_date": 2, "note_type": "foo"},
            {"excerpt": "D", "note_date": 3, "note_type": "foo"},
            {"excerpt": "E", "note_date": 4, "note_type": "foo"},
        ]
    )

    backend = MagicMock()
    backend.tag_excerpts.return_value = (
        [
            {"label": "NEGATIVE", "score": 0.9},  # A (drop)
            {"label": "POSITIVE", "score": 0.1},  # B (drop)
            {"label": "POSITIVE", "score": 0.8},  # C (keep)
            {"label": "NEGATIVE", "score": 0.2},  # D (keep)
            {"label": "POSITIVE", "score": 0.0},  # E (drop)
        ],
        {"model_name": "tagger"},
    )

    result, metadata = _extract_relevant_text_from_notes(
        patient_id="mrn",
        excerpts_frame=excerpts_frame,
        backend=backend,
        tagger_config={
            "model_name": "tagger",
            "negative_tag_cutoff": 0.5,
            "positive_tag_cutoff": 0.6,
        },
        model_metadata_cache_dir=None,
    )

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            {
                "patient_id": "mrn",
                "patient_long_text": "2 foo C\n3 foo D",
            }
        ),
    )
    assert metadata["model_name"] == "tagger"


def test_extract_relevant_text_from_patient_empty_excerpts(monkeypatch):
    """Ensure empty excerpt sets return a blank patient_long_text."""
    monkeypatch.setattr(
        "mmai.patients.tagging.split_sentences_and_dedup_notes",
        lambda _: pd.DataFrame(columns=["note_date", "note_type", "excerpt"]),
    )

    note_rows = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "foo",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )

    result, metadata = extract_relevant_text_from_patient(
        note_rows,
        patient_id="P1",
        backend=MagicMock(),
        tagger_config={"model_name": "tagger"},
        model_metadata_cache_dir=None,
    )
    assert result["patient_long_text"] == ""
    assert metadata["model_name"] == "tagger"


def test_split_sentences_and_dedup_notes():
    """Ensure notes are split into sentence chunks and deduplicated."""
    note_rows = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Line one. Line two.",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            },
            {
                "patient_id": "P1",
                "note_text": "Line two. Line four.",
                "note_type": "clinical_note",
                "note_date": "2024-01-02",
            },
        ]
    )

    excerpts = split_sentences_and_dedup_notes(note_rows)
    assert set(excerpts["excerpt"].tolist()) == {
        "Line one",
        "Line two",
        "Line two.",
        "Line four.",
    }
    assert excerpts["excerpt"].tolist().count("Line two") == 1


def test_extract_relevant_sentences_returns_metadata(monkeypatch):
    """Ensure tagger metadata and config snapshot are returned."""

    class MockBackend:
        def tag_excerpts(
            self, excerpts, *, tagger_config, model_metadata_cache_dir=None
        ):
            return (
                [{"label": "POSITIVE", "score": 0.9} for _ in excerpts],
                {"model_name": "tagger"},
            )

    monkeypatch.setattr("mmai.patients.tagging.get_backend", lambda name: MockBackend())

    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "A",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            },
            {
                "patient_id": "P2",
                "note_text": "B",
                "note_type": "clinical_note",
                "note_date": "2024-01-02",
            },
        ]
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={
            "tagger": {
                "model_name": "tagger",
                "device": "cpu",
                "batch_size": 2,
                "positive_tag_cutoff": 0.1,
                "negative_tag_cutoff": 0.9,
            }
        },
        model_metadata_cache_dir=None,
        raw={"version": 0},
    )

    result, metadata = extract_relevant_sentences(notes, config=config)
    assert len(result) == 2
    assert metadata["config_snapshot"]["version"] == 0
    assert metadata["model_metadata"]["model_name"] == "tagger"


def test_extract_relevant_sentences_returns_qc_report(monkeypatch):
    """Return a QC report with patients missing tagged notes."""

    def mock_extract(
        note_rows, patient_id, backend, *, tagger_config, model_metadata_cache_dir
    ):
        if patient_id == "P1":
            return (
                pd.Series({"patient_id": patient_id, "patient_long_text": ""}),
                {"model_name": "tagger"},
            )
        return (
            pd.Series({"patient_id": patient_id, "patient_long_text": "has text"}),
            {"model_name": "tagger"},
        )

    monkeypatch.setattr(
        "mmai.patients.tagging.extract_relevant_text_from_patient", mock_extract
    )
    monkeypatch.setattr("mmai.patients.tagging.get_backend", lambda name: MagicMock())

    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "A",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            },
            {
                "patient_id": "P2",
                "note_text": "B",
                "note_type": "clinical_note",
                "note_date": "2024-01-02",
            },
        ]
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={
            "tagger": {
                "model_name": "tagger",
                "device": "cpu",
                "batch_size": 2,
                "positive_tag_cutoff": 0.1,
                "negative_tag_cutoff": 0.9,
            }
        },
        model_metadata_cache_dir=None,
        raw={"version": 0},
    )

    result, metadata, qc_report = extract_relevant_sentences(
        notes, config=config, return_qc=True
    )

    assert len(result) == 2
    assert metadata["model_metadata"]["model_name"] == "tagger"
    report = qc_report.set_index("metric")
    assert report.loc["patients_with_no_tagged_notes", "value"] == 1
    assert report.loc["patients_with_no_tagged_notes", "ids"] == ["P1"]


def test_local_backend_tag_excerpts(monkeypatch):
    """Ensure the local tagger returns predictions and model metadata."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {"label": "POSITIVE", "score": 0.9},
        {"label": "NEGATIVE", "score": 0.1},
    ]

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
    mock_transformers.pipeline.return_value = mock_pipeline
    monkeypatch.setitem(__import__("sys").modules, "transformers", mock_transformers)

    monkeypatch.setattr(
        "mmai.backends.get_model_metadata",
        lambda model_name, cache_dir=None: {
            "model_name": model_name,
            "model_sha": "sha",
        },
    )

    backend = LocalBackend()
    predictions, metadata = backend.tag_excerpts(
        ["A", "B"],
        tagger_config={
            "model_name": "tagger",
            "device": "cpu",
            "batch_size": 2,
        },
        model_metadata_cache_dir=None,
    )

    assert predictions == [
        {"label": "POSITIVE", "score": 0.9},
        {"label": "NEGATIVE", "score": 0.1},
    ]
    assert metadata["model_name"] == "tagger"


def test_extract_relevant_sentences_lightweight_integration(monkeypatch):
    """Run extract_relevant_sentences end-to-end with a mocked backend tagger."""
    captured = {}

    class MockBackend:
        def tag_excerpts(
            self, excerpts, *, tagger_config, model_metadata_cache_dir=None
        ):
            captured["excerpts"] = excerpts
            return (
                [{"label": "POSITIVE", "score": 0.9} for _ in excerpts],
                {"model_name": "tagger"},
            )

    monkeypatch.setattr("mmai.patients.tagging.get_backend", lambda name: MockBackend())

    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Alpha. Beta.",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={
            "tagger": {
                "model_name": "tagger",
                "device": "cpu",
                "batch_size": 2,
                "positive_tag_cutoff": 0.1,
                "negative_tag_cutoff": 0.9,
            }
        },
        model_metadata_cache_dir=None,
        raw={"version": 0},
    )

    result, metadata = extract_relevant_sentences(notes, config=config)
    assert len(result) == 1
    assert "patient_long_text" in result.columns
    assert captured["excerpts"] == ["Alpha", "Beta."]
    assert metadata["model_metadata"]["model_name"] == "tagger"


def test_extract_relevant_sentences_preserves_patient_ids(monkeypatch):
    """Ensure patient_ids are preserved from group keys in the output."""

    class MockBackend:
        def tag_excerpts(
            self, excerpts, *, tagger_config, model_metadata_cache_dir=None
        ):
            return (
                [{"label": "POSITIVE", "score": 0.9} for _ in excerpts],
                {"model_name": "tagger"},
            )

    monkeypatch.setattr("mmai.patients.tagging.get_backend", lambda name: MockBackend())

    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Alpha.",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            },
            {
                "patient_id": "P2",
                "note_text": "Beta.",
                "note_type": "clinical_note",
                "note_date": "2024-01-02",
            },
        ]
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={
            "tagger": {
                "model_name": "tagger",
                "device": "cpu",
                "batch_size": 2,
                "positive_tag_cutoff": 0.1,
                "negative_tag_cutoff": 0.9,
            }
        },
        model_metadata_cache_dir=None,
        raw={"version": 0},
    )

    result, _ = extract_relevant_sentences(notes, config=config)
    assert set(result["patient_id"].tolist()) == {"P1", "P2"}

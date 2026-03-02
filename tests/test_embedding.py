import pandas as pd
import pytest

from mmai.config import MMAIConfig
from mmai.embedding.embed import embed_for_matching


class MockBackend:
    def __init__(self):
        self.last_texts = None
        self.last_embedding_config = None

    def generate_embeddings(self, texts, *, embedding_config):
        self.last_texts = texts
        self.last_embedding_config = embedding_config
        return [[float(len(text))] for text in texts]


def test_embed_for_matching_patient(monkeypatch):
    """Embed patient summaries and pass embedding config through to backend."""
    backend = MockBackend()
    monkeypatch.setattr("mmai.embedding.embed.get_backend", lambda name: backend)
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

    df = pd.DataFrame([{"patient_id": "P1", "cancer_history_summary": "abc"}])
    result = embed_for_matching(df, entity_type="patient", config=config)

    assert list(result.columns) == ["patient_id", "embedding"]
    assert result.loc[0, "embedding"] == [3.0]
    assert result.loc[0, "patient_id"] == "P1"
    assert backend.last_texts == ["abc"]
    assert backend.last_embedding_config["model_path"] == "mock-model"


def test_embed_for_matching_trial(monkeypatch):
    """Embed trial space summaries using the trial summary text column."""
    backend = MockBackend()
    monkeypatch.setattr("mmai.embedding.embed.get_backend", lambda name: backend)
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

    df = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "clinical_space_number": 1,
                "space_trial_id": "T1_1",
                "clinical_space_summary": "abcd",
            }
        ]
    )
    result = embed_for_matching(df, entity_type="trial", config=config)

    assert result.loc[0, "embedding"] == [4.0]
    assert result.loc[0, "trial_id"] == "T1"
    assert result.loc[0, "clinical_space_number"] == 1
    assert result.loc[0, "space_trial_id"] == "T1_1"


def test_embed_for_matching_missing_column(monkeypatch):
    """Raise a clear error when the required summary column is missing."""
    backend = MockBackend()
    monkeypatch.setattr("mmai.embedding.embed.get_backend", lambda name: backend)
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

    with pytest.raises(ValueError, match="missing required column"):
        embed_for_matching(
            pd.DataFrame([{"foo": "bar"}]),
            entity_type="trial",
            config=config,
        )


def test_embed_for_matching_reads_config(monkeypatch):
    """Read embedding model/device/prompt settings from config."""
    backend = MockBackend()
    monkeypatch.setattr("mmai.embedding.embed.get_backend", lambda name: backend)

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        model_metadata_cache_dir=None,
        raw={},
        embedding={
            "model_path": "cfg-model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
        },
    )
    result = embed_for_matching(
        pd.DataFrame([{"patient_id": "P2", "cancer_history_summary": "hello"}]),
        entity_type="patient",
        config=config,
    )
    assert result.loc[0, "embedding"] == [5.0]
    assert backend.last_embedding_config["model_path"] == "cfg-model"
    assert backend.last_embedding_config["device"] == "cpu"
    assert backend.last_embedding_config["prompt_file"] == "embedding.txt"

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

    assert "embedding" in result.columns
    assert result.loc[0, "embedding"] == [3.0]
    assert backend.last_texts == ["abc"]
    assert backend.last_embedding_config["model_path"] == "mock-model"


def test_embed_for_matching_trial(monkeypatch):
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

    df = pd.DataFrame([{"space_trial_id": "T1-1", "clinical_space_summary": "abcd"}])
    result = embed_for_matching(df, entity_type="trial", config=config)

    assert result.loc[0, "embedding"] == [4.0]


def test_embed_for_matching_allows_empty_model_path(monkeypatch):
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
        embedding={"model_path": "", "device": "cpu", "prompt_file": "embedding.txt"},
    )
    result = embed_for_matching(
        pd.DataFrame([{"cancer_history_summary": "x"}]),
        entity_type="patient",
        config=config,
    )
    assert result.loc[0, "embedding"] == [1.0]
    assert backend.last_embedding_config["model_path"] == ""


def test_embed_for_matching_missing_column(monkeypatch):
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


def test_embedding_prompt_is_set_on_model(monkeypatch):
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

    embed_for_matching(
        pd.DataFrame([{"cancer_history_summary": "hello"}]),
        entity_type="patient",
        config=config,
    )
    assert backend.last_embedding_config["prompt_file"] == "embedding.txt"


def test_embed_for_matching_reads_config(monkeypatch):
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
        pd.DataFrame([{"cancer_history_summary": "hello"}]),
        entity_type="patient",
        config=config,
    )
    assert result.loc[0, "embedding"] == [5.0]
    assert backend.last_embedding_config["model_path"] == "cfg-model"
    assert backend.last_embedding_config["device"] == "cpu"
    assert backend.last_embedding_config["prompt_file"] == "embedding.txt"

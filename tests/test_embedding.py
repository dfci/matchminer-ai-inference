import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.embedding.embed import embed_for_matching
from matchminer_ai.embedding import inference as embedding_inference
from matchminer_ai.embedding.inference import generate_embeddings


class MockBackend:
    def __init__(self):
        self.last_texts = None
        self.last_embedding_config = None
        self.last_model_metadata_cache_dir = None

    def generate_embeddings(
        self, texts, *, embedding_config, model_metadata_cache_dir=None
    ):
        self.last_texts = texts
        self.last_embedding_config = embedding_config
        self.last_model_metadata_cache_dir = model_metadata_cache_dir
        return (
            [[float(len(text))] for text in texts],
            {"model_name": embedding_config["model_path"], "model_sha": "mock-sha"},
        )


def test_embed_for_matching_patient(monkeypatch):
    """Embed patient summaries and pass embedding config through to backend."""
    backend = MockBackend()
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed.generate_embeddings", backend.generate_embeddings
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
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
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed.generate_embeddings", backend.generate_embeddings
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
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
                "space_trial_id": "T1_1",
                "clinical_space_summary": "abcd",
            }
        ]
    )
    result = embed_for_matching(df, entity_type="trial", config=config)

    assert result.loc[0, "embedding"] == [4.0]
    assert result.loc[0, "space_trial_id"] == "T1_1"


def test_embed_for_matching_missing_column(monkeypatch):
    """Raise a clear error when the required summary column is missing."""
    backend = MockBackend()
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed.generate_embeddings", backend.generate_embeddings
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
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
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed.generate_embeddings", backend.generate_embeddings
    )

    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
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


def test_embed_for_matching_return_metadata(monkeypatch):
    """Return embedding metadata payload when requested."""
    backend = MockBackend()
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed.generate_embeddings", backend.generate_embeddings
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        raw={"preset_name": "default"},
        embedding={
            "model_path": "cfg-model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
        },
    )

    result, metadata = embed_for_matching(
        pd.DataFrame([{"patient_id": "P2", "cancer_history_summary": "hello"}]),
        entity_type="patient",
        config=config,
        return_metadata=True,
    )

    assert list(result.columns) == ["patient_id", "embedding"]
    assert metadata["config_snapshot"]["preset_name"] == "default"
    assert metadata["config_snapshot"]["embedding"]["model_path"] == "cfg-model"
    assert metadata["model_metadata"]["embedding_model"]["model_name"] == "cfg-model"
    assert backend.last_model_metadata_cache_dir == ".mmai_cache/model_metadata"


def test_generate_embeddings_applies_configured_max_seq_length(monkeypatch):
    """Set SentenceTransformer max_seq_length from embedding config."""
    loaded_models = []

    class FakeSentenceTransformer:
        def __init__(self, model_path, device):
            self.model_path = model_path
            self.device = device
            self.prompts = {}
            self.max_seq_length = None
            loaded_models.append(self)

        def encode(self, texts, prompt):
            assert prompt == "query"
            return [[float(len(text))] for text in texts]

    monkeypatch.setattr(
        "matchminer_ai.embedding.inference._load_prompt_text",
        lambda filename: "Represent this sentence for retrieval:",
    )
    monkeypatch.setattr(
        "matchminer_ai.embedding.inference.get_model_metadata",
        lambda model_name, cache_dir=None: {"model_name": model_name},
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        type(
            "FakeSentenceTransformersModule",
            (),
            {"SentenceTransformer": FakeSentenceTransformer},
        ),
    )

    embeddings, metadata = generate_embeddings(
        ["abc"],
        embedding_config={
            "model_path": "embedder/model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
            "max_seq_length": 2500,
        },
    )

    assert embeddings == [[3.0]]
    assert metadata == {"model_name": "embedder/model"}
    assert loaded_models[0].prompts["query"] == "Represent this sentence for retrieval:"
    assert loaded_models[0].max_seq_length == 2500


def test_count_embedding_tokens_uses_tokenizer_without_loading_model(monkeypatch):
    """Token counting should not allocate the embedding model on CUDA."""
    embedding_inference._get_embedding_tokenizer.cache_clear()
    embedding_inference._get_embedding_model.cache_clear()
    loaded_tokenizers = []

    class FakeTokenizer:
        def __call__(self, texts, add_special_tokens=True, truncation=False):
            assert texts == ["query prompt abc", "query prompt de"]
            assert add_special_tokens is True
            assert truncation is False
            return {"input_ids": [[1, 2, 3], [1, 2]]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path, trust_remote_code=True):
            loaded_tokenizers.append((model_path, trust_remote_code))
            return FakeTokenizer()

    class FailingSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SentenceTransformer should not be loaded")

    monkeypatch.setattr(
        "matchminer_ai.embedding.inference._load_prompt_text",
        lambda filename: "query prompt",
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FailingSentenceTransformer),
    )

    counts = embedding_inference.count_embedding_tokens(
        ["abc", "de"],
        embedding_config={
            "model_path": "embedder/model",
            "device": "cuda",
            "prompt_file": "embedding.txt",
            "max_seq_length": 2500,
        },
    )

    assert counts == [3, 2]
    assert loaded_tokenizers == [("embedder/model", True)]

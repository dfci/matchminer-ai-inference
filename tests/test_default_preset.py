from matchminer_ai.config import load_default_preset


def test_default_preset_matches_training_runtime_defaults():
    """Keep public inference defaults aligned with the training scripts."""
    config = load_default_preset()

    assert config.local == {}
    assert config.trial["local"]["engine"]["max_model_len"] == 30000
    assert config.trial["local"]["generation"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 20000,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
    }

    assert config.trial["remote"] == {
        "served_model_name": "google/gemma-4-31B-it",
        "max_tokens_param": "max_tokens",
        "max_tokens": 20000,
        "request_params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
        },
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }

    assert config.patient["local"]["engine"]["max_model_len"] == 100000
    assert config.patient["chunk_size"] == 50000
    assert config.patient["chunk_overlap"] == 500
    assert config.patient["local"]["generation"]["temperature"] == 0.0
    assert config.patient["local"]["generation"]["top_k"] == 1
    assert config.patient["local"]["generation"]["max_tokens"] == 20000

    assert config.embedding["model_path"] == "ksg-dfci/TrialSpace-0526"
    assert config.embedding["max_seq_length"] == 2500
    assert config.raw["match_quality"]["model_name"] == "ksg-dfci/TrialChecker-0526"
    assert config.raw["match_quality"]["max_length"] == 4096
    assert config.raw["exclusion_criteria"]["model_name"] == (
        "ksg-dfci/BoilerplateChecker-0526"
    )
    assert config.raw["exclusion_criteria"]["max_length"] == 3192

    assert config.raw["llm_match_quality"]["local"]["engine"]["max_model_len"] == 50000
    assert config.raw["llm_match_quality"]["local"]["generation"]["temperature"] == 0.0
    assert config.raw["llm_match_quality"]["local"]["generation"]["max_tokens"] == (
        15000
    )
    assert (
        config.raw["llm_exclusion_criteria"]["local"]["engine"]["max_model_len"]
        == 50000
    )
    assert (
        config.raw["llm_exclusion_criteria"]["local"]["generation"]["temperature"]
        == 0.0
    )
    assert (
        config.raw["llm_exclusion_criteria"]["local"]["generation"]["max_tokens"]
        == 20000
    )
    assert config.raw["llm_match_quality"]["remote"]["request_params"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "presence_penalty": 0.0,
    }
    assert config.raw["llm_match_quality"]["remote"]["max_tokens"] == 15000
    assert config.raw["llm_match_quality"]["remote"]["extra_body"]["top_k"] == 1

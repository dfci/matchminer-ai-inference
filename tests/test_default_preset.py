from matchminer_ai.config import load_default_preset


def test_default_preset_matches_training_runtime_defaults():
    """Keep public inference defaults aligned with the training scripts."""
    config = load_default_preset()

    assert config.local["trial"]["max_model_len"] == 30000
    assert config.trial["sampling_params"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "max_tokens": 20000,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
    }

    assert config.local["patient"]["max_model_len"] == 100000
    assert config.patient["chunk_size"] == 50000
    assert config.patient["chunk_overlap"] == 500
    assert config.patient["sampling_params"]["temperature"] == 0.0
    assert config.patient["sampling_params"]["top_k"] == 1
    assert config.patient["sampling_params"]["max_tokens"] == 20000

    assert config.embedding["model_path"] == "ksg-dfci/TrialSpace-0526"
    assert config.embedding["max_seq_length"] == 2500
    assert config.raw["match_quality"]["model_name"] == "ksg-dfci/TrialChecker-0526"
    assert config.raw["match_quality"]["max_length"] == 4096
    assert config.raw["exclusion_criteria"]["model_name"] == (
        "ksg-dfci/BoilerplateChecker-0526"
    )
    assert config.raw["exclusion_criteria"]["max_length"] == 3192

    assert config.local["llm_match_quality"]["max_model_len"] == 50000
    assert config.raw["llm_match_quality"]["sampling_params"]["temperature"] == 0.0
    assert config.raw["llm_match_quality"]["sampling_params"]["max_tokens"] == 15000
    assert config.local["llm_exclusion_criteria"]["max_model_len"] == 50000
    assert config.raw["llm_exclusion_criteria"]["sampling_params"]["temperature"] == (
        0.0
    )
    assert config.raw["llm_exclusion_criteria"]["sampling_params"]["max_tokens"] == (
        20000
    )
    assert config.remote["served_model_name"] == "google/gemma-4-31B-it"
    assert config.remote["send_vllm_extra_body"] is True

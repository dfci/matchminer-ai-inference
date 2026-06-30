import pytest

from matchminer_ai import load_config
from matchminer_ai.config import load_preset


def test_load_config_from_user_path(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
debug_mode: true
model_metadata_cache_dir: ".custom_cache/model_metadata"
remote:
  enabled: true
  server_urls:
    - http://localhost:8000/v1
trial:
  model_name: custom-trial-model
  local:
    engine:
      max_model_len: 4096
    generation:
      max_tokens: 128
  remote:
    served_model_name: custom-trial-model
    max_tokens_param: max_tokens
    max_tokens: 128
    request_params: {}
    extra_body: {}
  prompt_files:
    primer: trial.user.primer.txt
    question: trial.user.question.txt
  boilerplate_marker: Boilerplate exclusions
patient:
  model_name: custom-patient-model
  local:
    engine:
      max_model_len: 4096
    generation:
      max_tokens: 256
  remote:
    served_model_name: custom-patient-model
    max_tokens_param: max_tokens
    max_tokens: 256
    request_params: {}
    extra_body: {}
  prompt_files:
    primer: patient.serial.user.primer.txt
    question: patient.serial.user.question.txt
  boilerplate_marker: Boilerplate conditions
embedding:
  model_path: custom-embedding-model
  device: cpu
  prompt_file: embedding.txt
  max_seq_length: 2500
match_quality:
  model_name: custom-match-model
exclusion_criteria:
  model_name: custom-exclusion-model
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.preset_name == str(config_path)
    assert config.debug_mode is True
    assert config.remote["enabled"] is True
    assert config.trial["model_name"] == "custom-trial-model"
    assert config.patient["model_name"] == "custom-patient-model"
    assert config.embedding["device"] == "cpu"
    assert config.embedding["max_seq_length"] == 2500
    assert config.raw["match_quality"]["model_name"] == "custom-match-model"


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="did not parse into a mapping"):
        load_config(config_path)


def test_load_preset_keeps_default_name():
    config = load_preset("default")

    assert config.preset_name == "default"
    assert config.trial["model_name"]

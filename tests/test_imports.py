import matchminer_ai
from matchminer_ai import MMAIPipeline, load_config, load_default_preset, load_preset
from matchminer_ai.embedding import embed_for_matching, embed_texts
from matchminer_ai.llm.vllm_server import start_vllm_servers
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_pairs,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    rank_similarity,
    score_match_quality,
    score_match_quality_pairs,
    score_match_quality_with_llm,
)
from matchminer_ai.patients import summarize_patient_notes, summarize_patients
from matchminer_ai.trials import summarize_trials


def test_imports():
    assert matchminer_ai is not None
    assert isinstance(matchminer_ai.__version__, str)
    assert MMAIPipeline is not None
    assert load_config is not None
    assert load_default_preset is not None
    assert load_preset is not None
    assert summarize_trials is not None
    assert summarize_patients is not None
    assert summarize_patient_notes is not None
    assert embed_for_matching is not None
    assert embed_texts is not None
    assert generate_candidate_matches is not None
    assert rank_similarity is not None
    assert score_match_quality is not None
    assert score_match_quality_pairs is not None
    assert score_match_quality_with_llm is not None
    assert exclusion_criteria_check is not None
    assert exclusion_criteria_check_pairs is not None
    assert exclusion_criteria_check_with_llm is not None
    assert start_vllm_servers is not None

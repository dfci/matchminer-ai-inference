import matchminer_ai
from matchminer_ai import MMAIPipeline
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)
from matchminer_ai.patients import summarize_patients
from matchminer_ai.trials import summarize_trials


def test_imports():
    assert matchminer_ai is not None
    assert MMAIPipeline is not None
    assert summarize_trials is not None
    assert summarize_patients is not None
    assert embed_for_matching is not None
    assert generate_candidate_matches is not None
    assert score_match_quality is not None
    assert exclusion_criteria_check is not None

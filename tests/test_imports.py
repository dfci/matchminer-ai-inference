import mmai
from mmai import MMAIPipeline
from mmai.embedding import embed_for_matching
from mmai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    reasonable_match_check,
)
from mmai.patients import summarize_patients
from mmai.trials import summarize_trials


def test_imports():
    assert mmai is not None
    assert MMAIPipeline is not None
    assert summarize_trials is not None
    assert summarize_patients is not None
    assert embed_for_matching is not None
    assert generate_candidate_matches is not None
    assert reasonable_match_check is not None
    assert exclusion_criteria_check is not None

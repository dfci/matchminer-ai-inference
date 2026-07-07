# Matching

::: matchminer_ai.matching.match
    options:
      members:
        - generate_candidate_matches

::: matchminer_ai.matching.rerank
    options:
      members:
        - score_match_quality

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - score_match_quality_with_llm
        - exclusion_criteria_check_with_llm

::: matchminer_ai.matching.exclusion_check
    options:
      members:
        - exclusion_criteria_check

# Exclusion Criteria

`exclusion_criteria_check` is the notebook layer: DataFrame in, DataFrame out, for
interactive analysis. `exclusion_criteria_check_pairs` is the service layer it's built
on: plain patient exclusion evidence and trial exclusion criteria text lists in, one
`{"exclusion_score", "exclusion_criteria_pass"}` per pair plus model metadata out, for
callers already working in their own data types (e.g. a live matching service).

::: matchminer_ai.matching.exclusion_check
    options:
      members:
        - exclusion_criteria_check
        - exclusion_criteria_check_pairs

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - exclusion_criteria_check_with_llm

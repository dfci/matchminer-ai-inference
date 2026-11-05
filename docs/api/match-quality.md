# Match Quality

`score_match_quality` is the notebook layer: DataFrame in, DataFrame out, for
interactive analysis. `score_match_quality_pairs` is the service layer it's built on:
plain patient and clinical-space text lists in, one
`{"match_quality_score", "match_quality_pass"}` per pair plus model metadata out, for
callers already working in their own data types (e.g. a live matching service).

::: matchminer_ai.matching.rerank
    options:
      members:
        - score_match_quality
        - score_match_quality_pairs

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - score_match_quality_with_llm

# Candidate Matching

`generate_candidate_matches` is the notebook layer: DataFrame in, DataFrame out, for
interactive analysis. `rank_similarity` is the service layer it's built on: plain
embedding lists in, `(corpus_index, score)` pairs out, for callers already working in
their own data types (e.g. a live matching service).

::: matchminer_ai.matching.match
    options:
      members:
        - generate_candidate_matches
        - rank_similarity

# Embedding

`embed_for_matching` is the notebook layer: DataFrame in, DataFrame out, for
interactive analysis. `embed_texts` is the service layer it's built on: plain text
list in, vectors plus metadata out, for callers already working in their own data
types (e.g. a live matching service).

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts

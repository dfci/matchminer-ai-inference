# Embedding

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts

For requests that already have summary strings, use the list interface directly:

```python
from matchminer_ai.embedding import embed_texts

vectors, model_metadata = embed_texts(["Patient summary"], config=config)
```

Vectors follow the input order. Model metadata includes `model_name` and
`model_sha`. The DataFrame API uses the same inference implementation.

# API Reference

The API pages are generated from package docstrings with `mkdocstrings`.

For a guided workflow, start with the [quickstart](../getting-started/quickstart.md).
The API reference is intended for looking up function signatures and parameter
details.

Most external workflows use these import paths:

```python
from matchminer_ai import load_config, load_default_preset, load_preset
from matchminer_ai.trials import summarize_trials
from matchminer_ai.patients import summarize_patients
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)
```

The LLM utilities page documents `start_vllm_server`, the public helper for
starting a local OpenAI-compatible vLLM server from package configuration.

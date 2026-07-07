# API Overview

The API pages are generated from package docstrings with `mkdocstrings`.

Most external workflows should start with these import paths:

```python
from matchminer_ai import load_config, load_preset
from matchminer_ai.trials import summarize_trials
from matchminer_ai.patients import summarize_patients
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)
```

The package also exposes lower-level LLM utilities for remote vLLM-compatible
server handling.

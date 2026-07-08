# Quickstart

The example notebook provides a runnable walkthrough using sample input data:

[Open `examples/run_examples.ipynb` on GitHub](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

Most external workflows should start with these import paths:

```python
from matchminer_ai import load_config
from matchminer_ai.trials import summarize_trials
from matchminer_ai.patients import summarize_patients
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    score_match_quality,
    score_match_quality_with_llm,
)
```

The patient-centric workflow has these main steps:

1. Turn trial descriptions and eligibility criteria into structured trial
   summaries with `summarize_trials`.
2. Turn patient notes into cancer history summaries with `summarize_patients`.
3. Convert the trial and patient summaries into embeddings with
   `embed_for_matching` so they can be compared.
4. Find likely patient-trial pairs with `generate_candidate_matches`.
5. Score how reasonable each patient-trial match looks with either
   `score_match_quality` or `score_match_quality_with_llm`.
6. Screen matched patient-trial pairs for possible exclusion criteria with
   either `exclusion_criteria_check` or `exclusion_criteria_check_with_llm`.

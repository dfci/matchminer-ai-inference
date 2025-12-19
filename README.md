# mmai

MatchMiner-AI Python package skeleton.

## Status

This is a minimal scaffold based on `docs/skeleton.md`. All functions are stubs and
raise `NotImplementedError`.

## Install (dev)

```sh
python -m pip install -e ".[dev]"
```

## Quick usage

```python
from mmai import MMAIPipeline
from mmai.trials import summarize_trials
from mmai.patients import summarize_patients
from mmai.embedding import embed_for_matching
from mmai.matching import (
    generate_candidate_matches,
    reasonable_match_check,
    exclusion_criteria_check,
)

pipeline = MMAIPipeline()
```

## Tests

```sh
pytest
```

## Development

```sh
pre-commit install
pre-commit run --all-files
```

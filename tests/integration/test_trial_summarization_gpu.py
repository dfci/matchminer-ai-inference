from pathlib import Path

import pandas as pd
import pytest

from mmai.trials import summarize_trials


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        pytest.skip(f"torch not available: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")


@pytest.mark.resource_heavy
def test_trial_summarization_gpu_smoke():
    """Run trial summarization end-to-end on GPU for a small sample."""
    _require_cuda()
    data_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "data"
        / "scheduled__2025-09-04T230000+0000.trials_for_summarize.csv"
    )
    trials = pd.read_csv(data_path, nrows=2)

    summaries, metadata = summarize_trials(trials, return_metadata=True)

    assert not summaries.empty
    assert "space_trial_id" in summaries.columns
    assert "trial_id" in summaries.columns
    assert "clinical_space_summary" in summaries.columns
    assert "model_metadata" in metadata

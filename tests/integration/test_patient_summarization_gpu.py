from pathlib import Path

import pandas as pd
import pytest

from mmai.patients import summarize_patients


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        pytest.skip(f"torch not available: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")


@pytest.mark.resource_heavy
def test_patient_summarization_gpu_smoke():
    """Run tagging + summarization end-to-end on GPU for a small sample."""
    _require_cuda()
    data_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "data"
        / "2026_01_16_dummy_notes.csv"
    )
    notes = pd.read_csv(data_path, nrows=3)

    summaries, metadata = summarize_patients(notes, return_metadata=True)

    assert not summaries.empty
    assert "patient_id" in summaries.columns
    assert "cancer_history_summary" in summaries.columns
    assert "general_exclusion_criteria_evidence" in summaries.columns
    assert "model_metadata" in metadata

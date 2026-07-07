import pandas as pd

from matchminer_ai._qc.patients import patient_summary_qc_report
from matchminer_ai.config import MMAIConfig


def test_patient_summary_qc_report_metrics(monkeypatch):
    """Validate summary QC metrics for embedding limits and content checks."""
    summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Cancer type: Lung. Histology: NSCLC.",
                "general_exclusion_criteria_evidence": "None",
            },
            {
                "patient_id": "P2",
                "cancer_history_summary": "",
                "general_exclusion_criteria_evidence": "None",
            },
            {
                "patient_id": "P3",
                "cancer_history_summary": "Cancer type: Breast.",
                "general_exclusion_criteria_evidence": "Cancer type: Breast.",
            },
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai._qc.patients.count_embedding_tokens",
        lambda texts, *, embedding_config: [100, 3001, 50],
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={
            "model_path": "m",
            "device": "cpu",
            "prompt_file": "embedding.txt",
            "max_seq_length": 2500,
        },
        model_metadata_cache_dir=None,
        raw={},
    )

    report = patient_summary_qc_report(
        summaries,
        config=config,
        expected_keywords=["Cancer type", "Histology"],
    ).set_index("metric")

    assert report.loc["patients_exceed_embedding_token_limit", "value"] == 1
    assert report.loc["patients_exceed_embedding_token_limit", "ids"] == ["P2"]
    assert report.loc["patients_exclusion_criteria_not_extracted", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2


def test_patient_summary_qc_uses_embedding_max_seq_length(monkeypatch):
    """Use embedding.max_seq_length as the default QC token limit."""
    summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Cancer type: Lung.",
                "general_exclusion_criteria_evidence": "None",
            },
            {
                "patient_id": "P2",
                "cancer_history_summary": "Cancer type: Breast.",
                "general_exclusion_criteria_evidence": "None",
            },
        ]
    )
    monkeypatch.setattr(
        "matchminer_ai._qc.patients.count_embedding_tokens",
        lambda texts, *, embedding_config: [1500, 2100],
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={
            "model_path": "m",
            "device": "cpu",
            "prompt_file": "embedding.txt",
            "max_seq_length": 2000,
        },
        model_metadata_cache_dir=None,
        raw={},
    )

    report = patient_summary_qc_report(
        summaries,
        config=config,
        expected_keywords=[],
    ).set_index("metric")

    assert report.loc["patients_exceed_embedding_token_limit", "ids"] == ["P2"]

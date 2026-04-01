import pandas as pd

from mmai._qc.patients import patient_summary_qc_report
from mmai.config import MMAIConfig


def test_patient_summary_qc_report_metrics(monkeypatch):
    """Validate summary QC metrics for drops, embedding limit, and content checks."""
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
        "mmai._qc.patients.get_backend",
        lambda _: type(
            "MockBackend",
            (),
            {
                "count_embedding_tokens": lambda self, texts, embedding_config: [
                    100,
                    3001,
                    50,
                ]
            },
        )(),
    )
    config = MMAIConfig(
        preset_name="default",
        debug_mode=False,
        backend="local",
        trial={},
        patient={},
        embedding={"model_path": "m", "device": "cpu", "prompt_file": "embedding.txt"},
        model_metadata_cache_dir=None,
        raw={},
    )

    report = patient_summary_qc_report(
        summaries,
        noninformative_summary_qc_artifact={
            "metric": "patients_dropped_noninformative_summary",
            "numerator": 1,
            "denominator": 3,
            "ids": ["P2"],
        },
        config=config,
        max_embedding_input_tokens=2500,
        expected_keywords=["Cancer type", "Histology"],
    ).set_index("metric")

    assert report.loc["patients_dropped_noninformative_summary", "value"] == 1
    assert report.loc["patients_exceed_embedding_token_limit", "value"] == 1
    assert report.loc["patients_exceed_embedding_token_limit", "ids"] == ["P2"]
    assert report.loc["patients_exclusion_criteria_not_extracted", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2

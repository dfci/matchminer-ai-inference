import pandas as pd

from mmai._qc.patients import (
    patient_qc_report,
    patient_summary_qc_report,
    tagger_qc_report,
)
from mmai.config import MMAIConfig


def test_tagger_qc_report_metrics():
    """Validate tagger QC metrics for patients with no tagged notes."""
    notes = pd.DataFrame(
        [
            {"patient_id": "P1"},
            {"patient_id": "P2"},
        ]
    )
    tagged = pd.DataFrame(
        [
            {"patient_id": "P1", "patient_long_text": "notes"},
            {"patient_id": "P2", "patient_long_text": ""},
        ]
    )

    report = tagger_qc_report(tagged, patient_note_source=notes).set_index("metric")

    assert report.loc["patients_with_no_tagged_notes", "value"] == 1
    assert report.loc["patients_with_no_tagged_notes", "ids"] == ["P2"]


def test_patient_summary_qc_report_metrics(monkeypatch):
    """Validate summary-only QC metrics for drops, truncation, and keywords."""
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
    finish_reasons = pd.Series(
        ["stop", "length", "stop"],
        index=summaries["patient_id"].astype(str),
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
        noninformative_summary_drop_ids=["P2"],
        finish_reasons=finish_reasons,
        config=config,
        max_embedding_input_tokens=2500,
        expected_keywords=["Cancer type", "Histology"],
    ).set_index("metric")

    assert report.loc["patients_dropped_noninformative_summary", "value"] == 1
    assert report.loc["patients_truncated_llm_response", "value"] == 1
    assert report.loc["patients_truncated_llm_response", "ids"] == ["P2"]
    assert report.loc["patients_exceed_embedding_token_limit", "value"] == 1
    assert report.loc["patients_exceed_embedding_token_limit", "ids"] == ["P2"]
    assert report.loc["patients_exclusion_criteria_not_extracted", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2


def test_patient_qc_report_metrics(monkeypatch):
    """Validate patient QC metrics for missing summaries, boilerplate, and keywords."""
    tagged = pd.DataFrame(
        [
            {"patient_id": "P1", "patient_long_text": "notes"},
            {"patient_id": "P2", "patient_long_text": ""},
            {"patient_id": "P3", "patient_long_text": "notes"},
        ]
    )
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
    finish_reasons = pd.Series(
        ["stop", "length", "stop"],
        index=summaries["patient_id"].astype(str),
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

    summary_report = patient_summary_qc_report(
        summaries,
        noninformative_summary_drop_ids=["P2"],
        finish_reasons=finish_reasons,
        config=config,
        max_embedding_input_tokens=2500,
        expected_keywords=["Cancer type", "Histology"],
    )
    tagger_report = tagger_qc_report(
        tagged,
        patient_note_source=tagged[["patient_id"]],
    )

    report = patient_qc_report(
        summaries,
        patient_note_source=tagged[["patient_id"]],
        summary_qc_report=summary_report,
        tagger_qc_report=tagger_report,
        expected_keywords=["Cancer type", "Histology"],
    ).set_index("metric")

    assert report.loc["patients_with_no_tagged_notes", "value"] == 1
    assert report.loc["patients_missing_summaries", "value"] == 1
    assert report.loc["patients_dropped_noninformative_summary", "value"] == 1
    assert report.loc["patients_dropped_noninformative_summary", "ids"] == ["P2"]
    assert report.loc["patients_truncated_llm_response", "ids"] == ["P2"]
    assert report.loc["patients_exceed_embedding_token_limit", "ids"] == ["P2"]
    assert report.loc["patients_exclusion_criteria_not_extracted", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2

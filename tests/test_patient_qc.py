import pandas as pd

from mmai._qc.patients import patient_qc_report


def test_patient_qc_report_metrics():
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

    report = patient_qc_report(
        summaries,
        patient_note_source=tagged[["patient_id"]],
        tagged_notes=tagged,
        noninformative_summary_drop_ids=["P2"],
        expected_keywords=["Cancer type", "Histology"],
        max_summary_length=40,
    ).set_index("metric")

    assert report.loc["patients_with_no_tagged_notes", "value"] == 1
    assert report.loc["patients_missing_summaries", "value"] == 1
    assert report.loc["patients_dropped_noninformative_summary", "value"] == 1
    assert report.loc["patients_dropped_noninformative_summary", "ids"] == ["P2"]
    assert report.loc["patients_exclusion_criteria_not_extracted", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2
    assert report.loc["patient_summaries_excessive_length", "value"] >= 0

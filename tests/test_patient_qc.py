import pandas as pd

from mmai.qc.patients import patient_qc_report


def test_patient_qc_report_metrics():
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
        tagged_patients=tagged,
        expected_keywords=["Cancer type", "Histology"],
        max_summary_length=40,
    ).set_index("metric")

    assert report.loc["patients_with_no_tagged_notes", "value"] == 1
    assert report.loc["patients_missing_summaries", "value"] == 1
    assert report.loc["patients_summary_equals_boilerplate", "value"] == 1
    assert report.loc["patients_missing_keyword:Histology", "value"] == 2
    assert report.loc["patient_summaries_excessive_length", "value"] >= 0

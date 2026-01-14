import pandas as pd
import pytest

from mmai.trials.postprocess import _strip_numerical_prefix


TRIAL_SPACE_1 = "1. Cancer type allowed: A."
TRIAL_SPACE_2 = "2. Cancer type allowed: B."
TRIAL_SPACE_3 = "3. Cancer type allowed: C."
TRIAL_SPACE_4 = "1. Cancer type allowed: D."
BOILERPLATE_1 = "Uncontrolled brain metastases."
BOILERPLATE_2 = "History of pneumonitis."


@pytest.fixture
def mock_summarized_data() -> pd.DataFrame:
    summarized = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title 1",
                "brief_summary": "Brief 1",
                "eligibility_criteria": "Criteria 1",
                "trial_text": "Text 1",
                "space_reasoning_and_output": (
                    "assistantfinal\n"
                    f"{TRIAL_SPACE_1}\n"
                    f"{TRIAL_SPACE_2}\n"
                    f"{TRIAL_SPACE_3}\n"
                    "Boilerplate exclusions:\n"
                    f"{BOILERPLATE_1}"
                ),
            },
            {
                "trial_id": "T2",
                "trial_title": "Title 2",
                "brief_summary": "Brief 2",
                "eligibility_criteria": "Criteria 2",
                "trial_text": "Text 2",
                "space_reasoning_and_output": (
                    "assistantfinal\n"
                    f"{TRIAL_SPACE_4}\n"
                    "Boilerplate exclusions:\n"
                    f"{BOILERPLATE_2}"
                ),
            },
        ]
    )
    return summarized


@pytest.fixture
def mock_data_for_embed(mock_summarized_data: pd.DataFrame) -> pd.DataFrame:
    with_summaries = mock_summarized_data.copy()
    with_summaries["space_output_no_reasoning"] = (
        with_summaries["space_reasoning_and_output"]
        .str.split("assistantfinal", n=1)
        .apply(lambda parts: parts[-1])
    )
    with_summaries["space_text"] = [
        (TRIAL_SPACE_1 + "\n" + TRIAL_SPACE_2 + "\n" + TRIAL_SPACE_3).strip(),
        TRIAL_SPACE_4.strip(),
    ]
    with_summaries["boilerplate_text"] = [
        BOILERPLATE_1.strip(),
        BOILERPLATE_2.strip(),
    ]

    for_embed = with_summaries.iloc[[0, 0, 0, 1]].copy()
    for_embed["clinical_space_summary"] = [
        _strip_numerical_prefix(TRIAL_SPACE_1),
        _strip_numerical_prefix(TRIAL_SPACE_2),
        _strip_numerical_prefix(TRIAL_SPACE_3),
        _strip_numerical_prefix(TRIAL_SPACE_4),
    ]
    for_embed["clinical_space_number"] = [1, 2, 3, 1]

    return for_embed.reset_index(drop=True)

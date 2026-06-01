import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.trials.postprocess import _strip_numerical_prefix


TRIAL_SPACE_1 = (
    "1. Age: 18+. Sex: Any. Cancer type allowed: A. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_2 = (
    "2. Age: 18+. Sex: Any. Cancer type allowed: B. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_3 = (
    "3. Age: 18+. Sex: Any. Cancer type allowed: C. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
TRIAL_SPACE_4 = (
    "1. Age: 18+. Sex: Any. Cancer type allowed: D. Histology allowed: Any. "
    "Cancer burden allowed: Any. Prior treatment required: None. "
    "Prior treatment excluded: None. Biomarkers required: None. "
    "Biomarkers excluded: None."
)
BOILERPLATE_1 = "Uncontrolled brain metastases."
BOILERPLATE_2 = "History of pneumonitis."


@pytest.fixture
def default_trial_config() -> dict:
    return {
        "model_name": "model",
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 10,
            "repetition_penalty": 1.0,
        },
        "prompt_files": {
            "primer": "trial.user.primer.txt",
            "question": "trial.user.question.txt",
        },
        "boilerplate_marker": "Boilerplate exclusions",
        "model_metadata_cache_dir": ".mmai_cache/model_metadata",
    }


@pytest.fixture
def default_config(default_trial_config: dict) -> MMAIConfig:
    return MMAIConfig(
        preset_name="default",
        debug_mode=False,
        trial=default_trial_config,
        patient={},
        local={
            "trial": {
                "max_model_len": 100,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        },
        remote={},
        model_metadata_cache_dir=None,
        raw={},
        embedding={},
    )


@pytest.fixture
def mock_summarized_data() -> pd.DataFrame:
    # post llm summarization, raw LLM output
    summarized = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title 1",
                "brief_summary": "Brief 1",
                "eligibility_criteria": "Criteria 1",
                "trial_text": "Text 1",
                "space_output_no_reasoning": (
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
                "space_output_no_reasoning": (
                    f"{TRIAL_SPACE_4}\n" "Boilerplate exclusions:\n" f"{BOILERPLATE_2}"
                ),
            },
        ]
    )
    return summarized


@pytest.fixture
def expected_flattened_spaces(mock_summarized_data: pd.DataFrame) -> pd.DataFrame:
    with_summaries = mock_summarized_data.copy()
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

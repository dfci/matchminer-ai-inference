# MatchMiner-AI Tutorial

This tutorial provides background as well as specific information regarding how to run the `run_examples.ipynb` notebook.

There are two main ways patients and clinical trials can be matched:
1. Patient-centric matching: Find a set of clinical trials for a patient; and
2. Trial-centric matching: Find a set of patients for a clinical trial.

Currently, MatchMiner-AI is most fully developed for Patient-centric matching. The `run_examples.ipynb` notebook and the following information mainly support this method of finding patient-trial matches.

## How MatchMiner-AI works

MatchMiner-AI has 6 main steps (Figure 1):
1. Summarize Trials
2. Summarize Patients
3. Embed Trial Spaces[1] and Patient Summaries (TrialSpace Model)
4. Generate Candidate Matches
5. Evaluate Candidate Match Quality (TrialChecker Model)
6. Check for Exclusions (Boilerplate Checker Model)

[1] A clinical trial is divided into one or more Trial Spaces. Please see the [Trial Spaces](#trial-spaces) section below for more information.

![Figure 1. MatchMiner-AI Process Overview](./images/MM-AI_process-overview_publicpkg.png) 

Table 1 summarizes the models used for each step. For the most GPU-intensive steps (Trial and Patient Summarization), we offer two different server modes:
1. Local Server Mode; and
2. Remote Server Mode. 

These are described more in the [Server Handling](#server-handling) section below.

Table 1: MatchMiner-AI Steps and Models

| Step | Package Function | Model | Notes |
|------|-------|-------|--------|
| 1. Summarize Trials | `summarize_trials` | public LLM | by default, currently uses `gpt-oss-120b` |
| 2. Summarize Patients | `summarize_patients` | public LLM | by default, currently uses `gpt-oss-120b` |
| 3. Embed Trial Spaces and Patient Summaries | `embed_for_matching` | `TrialSpace`, a trained Sentence Transformers model | provided on Hugging Face at `https://huggingface.co/ksg-dfci` |
| 4. Generate Candidate Matches | `generate_candidate_matches` | NA | |
| 5. Evaluate Candidate Match Quality | `score_match_quality` | `TrialChecker`, a trained ModernBERT model | provided on Hugging Face at `https://huggingface.co/ksg-dfci` |
| 6. Check for Exclusions | `exclusion_criteria_check` | `BoilerplateChecker`, a trained ModernBERT model | provided on Hugging Face at `https://huggingface.co/ksg-dfci` |

## Trial Spaces

A Clinical Trial may have multiple arms with different target populations.  For each trial, MatchMiner-AI will extract a list of clinical “spaces” for the trial from its eligibility criteria, where each space is defined as a unique combination of core clinical concepts (age, sex, cancer type, histology, burden of disease, prior treatment, and biomarkers) that might render the patient eligible. Some trials have only one “space,” whereas others, such as basket or umbrella trials, have several. At the end of `summarize_trials`, each trial space is listed as a separate entity or cohort in the dataframe. 

## Server Handling

For the most GPU-intensive steps (Trial and Patient Summarization), we offer two different server modes:
1. Local mode (Figure 2): uses the local in-memory `vLLM` backend by default.

![Figure 2: Trial and Patient Summarization can be run in Local Mode.](./images/local_server_mode.png)\

2. Remote mode (Figure 3): sends summarization requests to an existing OpenAI-compatible endpoint.

![Figure 3: Trial and Patient Summarization can be run in Remote Mode.](./images/remote_server_mode.png)

Remote mode can also be run with a server started on your local machine. (Figure 4) MatchMiner-AI provides the `start_vllm_server()` function to start a `vllm_server` on your machine. In this scenario, the URL would be a localhost URL. 

![Figure 4: Running Remote Mode with a server located on your Local Machine.](./images/local_remote_server_mode.png)

 

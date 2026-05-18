# Preset Configuration Reference

Preset files are YAML mappings loaded by `matchminer_ai.config.load_preset`.
`default.yaml` is loaded by `load_default_preset()`.

## Root Keys

### `version`

Preset schema version.

### `debug_mode`

Boolean flag used by summarization postprocessing. When true, selected
intermediate columns are retained in output tables.

### `model_metadata_cache_dir`

Directory path used by model metadata helpers to cache Hugging Face model
metadata JSON files.

## `local`

Configuration used when `remote.enabled` is false and trial/patient
summarization runs through an in-process vLLM engine.

### `local.trial`

Keyword arguments passed to `vllm.LLM(...)` for trial summarization. The package
adds `model=trial.model_name` separately.

Required by the default preset:

- `max_model_len`
- `tensor_parallel_size`
- `gpu_memory_utilization`

Additional keys may be included if they are valid `vllm.LLM` keyword arguments.
vLLM validates those keys when the engine is created.

### `local.patient`

Keyword arguments passed to `vllm.LLM(...)` for patient summarization. The
package adds `model=patient.model_name` separately.

Required by the default preset:

- `max_model_len`
- `tensor_parallel_size`
- `gpu_memory_utilization`

Additional keys may be included if they are valid `vllm.LLM` keyword arguments.
vLLM validates those keys when the engine is created.

`max_model_len` is also read by patient prompt construction to determine the
maximum context window used for chunk truncation.

## `remote`

Configuration used when `remote.enabled` is true and trial/patient
summarization sends OpenAI-compatible HTTP requests to external vLLM servers.
The remote backend reads the API key from the `OPENAI_API_KEY` environment
variable. API keys are not stored in preset files.

### `remote.enabled`

Selects the remote summarization backend when true.

### `remote.server_urls`

List of OpenAI-compatible base URLs. Values are passed to the OpenAI client as
`base_url`.

### `remote.max_concurrent_requests`

Maximum number of concurrent requests per remote server.

### `remote.request_timeout`

Request timeout in seconds.

### `remote.max_retries`

Maximum retry attempts for a failed remote request.

### `remote.batch_size`

Number of prompts processed per remote-server batch.

### `remote.retry_backoff_base`

Base value, in seconds, for exponential retry backoff.

## `trial`

Task configuration for trial summarization.

### `trial.model_name`

Model identifier used for:

- tokenizer/chat-template rendering
- Hugging Face model metadata lookup
- `vllm.LLM(model=...)` in local mode
- OpenAI-compatible request `model` in remote mode

For remote mode, the vLLM server must expose a served model name matching this
value.

### `trial.sampling_params`

Keyword arguments passed to `vllm.SamplingParams(...)` in local mode.

Remote mode maps known OpenAI-compatible fields and selected vLLM-specific
fields from this mapping into completion request parameters.

Additional keys may be included if they are valid `vllm.SamplingParams` keyword
arguments. vLLM validates those keys in local mode.

### `trial.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `trial.reasoning_marker`

Marker used by trial postprocessing to split/remove reasoning text from model
output.

### `trial.boilerplate_marker`

Regular expression used by trial postprocessing to identify boilerplate
exclusion text.

## `patient`

Task configuration for patient summarization.

### `patient.model_name`

Model identifier used for:

- tokenizer/chat-template rendering
- Hugging Face model metadata lookup
- `vllm.LLM(model=...)` in local mode
- OpenAI-compatible request `model` in remote mode

For remote mode, the vLLM server must expose a served model name matching this
value.

### `patient.chunk_size`

Maximum character count used when splitting patient notes into serial summary
chunks.

### `patient.chunk_overlap`

Character overlap between adjacent patient-note chunks.

### `patient.prompt_margin_tokens`

Token margin reserved when truncating patient chunks before prompt rendering.

### `patient.sampling_params`

Keyword arguments passed to `vllm.SamplingParams(...)` in local mode.

Remote mode maps known OpenAI-compatible fields and selected vLLM-specific
fields from this mapping into completion request parameters.

Additional keys may be included if they are valid `vllm.SamplingParams` keyword
arguments. vLLM validates those keys in local mode.

### `patient.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `patient.reasoning_marker`

Marker used by patient postprocessing to split/remove reasoning text from model
output.

### `patient.boilerplate_marker`

Regular expression used by patient postprocessing to remove boilerplate marker
text.

### `patient.text_token_threshold`

Maximum token count used by local truncation before patient summarization.

## `embedding`

Configuration for summary embedding.

### `embedding.model_path`

Sentence-transformer model path passed to `SentenceTransformer(...)`.

### `embedding.device`

Device string passed to `SentenceTransformer(...)`.

### `embedding.prompt_file`

Prompt filename loaded from `matchminer_ai.prompts` and used as the embedding
query prompt.

## `match_quality`

Configuration for the match-quality checker model.

### `match_quality.model_name`

Text-classification model identifier used by the checker pipeline and model
metadata lookup.

### `match_quality.device`

Device passed to the checker pipeline.

### `match_quality.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `match_quality.score_cutoff`

Minimum sigmoid-transformed checker score required for
`match_quality_pass == true`.

## `exclusion_criteria`

Configuration for the exclusion-criteria checker model.

### `exclusion_criteria.model_name`

Text-classification model identifier used by the checker pipeline and model
metadata lookup.

### `exclusion_criteria.device`

Device passed to the checker pipeline.

### `exclusion_criteria.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

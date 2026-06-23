# Preset Configuration Reference

Preset files are YAML mappings loaded by `matchminer_ai.config.load_preset`.
`default.yaml` is loaded by `load_default_preset()`.

## Custom Config Files

For package installs, treat the built-in preset files as read-only package data.
To customize configuration, copy the default preset values into a YAML file in
your project, edit them, and load that file by path:

```python
from matchminer_ai import load_config

config = load_config("my_config.yaml")
```

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

Configuration used when `remote.enabled` is true and LLM tasks send
OpenAI-compatible chat completion requests to external endpoints. This can be a
vLLM server or any endpoint compatible with the OpenAI chat completions API. The
remote backend reads the API key from the `OPENAI_API_KEY` environment variable.
API keys are not stored in preset files.

### `remote.enabled`

Selects the remote summarization backend when true.

### `remote.server_urls`

List of OpenAI-compatible base URLs. Values are passed to the OpenAI client as
`base_url`. For the default Gemma 4 configuration, the server should be a vLLM
chat endpoint launched with the `gemma4` reasoning parser; the package
`start_vllm_server()` helper adds this flag from `trial.reasoning_parser` or
`patient.reasoning_parser`.

### `remote.served_model_name`

Optional model name sent in OpenAI-compatible chat completion requests. This can
differ from `trial.model_name` / `patient.model_name`, which remain the
tokenizer and chat-template model names used while building prompts. This is
useful for endpoints exposing aliases or non-Hugging-Face model names.

### `remote.send_vllm_extra_body`

When true, vLLM-specific sampling parameters such as `top_k`,
`repetition_penalty`, and `chat_template_kwargs` are sent in request
`extra_body`. Set this to false for standard OpenAI-compatible endpoints that
reject vLLM-specific fields.

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
- OpenAI-compatible request `model` in remote mode unless
  `remote.served_model_name` is set

For remote mode, use `remote.served_model_name` when the endpoint exposes a
different model alias.

The default preset uses a Gemma 4 model for summarization. The specific Gemma
variant that will run successfully may depend on the available GPU type and
memory.

### `trial.sampling_params`

Keyword arguments passed to `vllm.SamplingParams(...)` in local mode.

Remote mode maps known OpenAI-compatible fields and selected vLLM-specific
fields from this mapping into chat completion request parameters.

Additional keys may be included if they are valid `vllm.SamplingParams` keyword
arguments. vLLM validates those keys in local mode.

### `trial.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `trial.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model.

### `trial.chat_template_kwargs`

Keyword arguments passed to tokenizer chat-template rendering in local mode and
to vLLM request `extra_body.chat_template_kwargs` in remote mode. The default
sets `enable_thinking: true` for Gemma 4.

### `trial.boilerplate_marker`

Line marker used by trial postprocessing to identify the boilerplate exclusion
section heading.

## `patient`

Task configuration for patient summarization.

### `patient.model_name`

Model identifier used for:

- tokenizer/chat-template rendering
- Hugging Face model metadata lookup
- `vllm.LLM(model=...)` in local mode
- OpenAI-compatible request `model` in remote mode unless
  `remote.served_model_name` is set

For remote mode, use `remote.served_model_name` when the endpoint exposes a
different model alias.

The default preset uses a Gemma 4 model for summarization. The specific Gemma
variant that will run successfully may depend on the available GPU type and
memory.

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
fields from this mapping into chat completion request parameters.

Additional keys may be included if they are valid `vllm.SamplingParams` keyword
arguments. vLLM validates those keys in local mode.

### `patient.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `patient.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model.

### `patient.chat_template_kwargs`

Keyword arguments passed to tokenizer chat-template rendering in local mode and
to vLLM request `extra_body.chat_template_kwargs` in remote mode. The default
sets `enable_thinking: true` for Gemma 4.

### `patient.boilerplate_marker`

Line marker used by patient postprocessing to identify the boilerplate
conditions section heading.

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

### `embedding.max_seq_length`

Runtime truncation cutoff for embedding inputs. `SentenceTransformer`
uses this value during `encode()`, so inputs longer than this limit are
truncated before embedding generation. QC reports use the same value when
flagging summaries that exceed the embedding input limit.

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

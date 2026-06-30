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

## `remote`

Global transport settings used when `remote.enabled` is true and LLM tasks send
OpenAI-compatible chat completion requests to external endpoints. This can be a
vLLM server or any endpoint compatible with the OpenAI chat completions API.
Task-specific request payload settings live under each task's `remote` block.
The remote backend reads the API key from the `OPENAI_API_KEY` environment
variable. API keys are not stored in preset files.

### `remote.enabled`

Selects the remote LLM backend when true.

### `remote.server_urls`

List of OpenAI-compatible base URLs. Values are passed to the OpenAI client as
`base_url`. For the default Gemma 4 configuration, the server should be a vLLM
chat endpoint launched with the `gemma4` reasoning parser; the package
`start_vllm_server()` helper adds this flag from `trial.reasoning_parser` or
`patient.reasoning_parser`.

Model names and request parameters are configured per LLM task under that
task's `remote` block. For example, `trial.remote.served_model_name` is the
model string sent to the endpoint, `trial.remote.request_params` contains
top-level chat completion request fields, and `trial.remote.extra_body`
contains fields sent through request `extra_body`.

Separate reasoning output is backend-dependent. The default vLLM/Gemma setup
can expose reasoning via vLLM reasoning parser support. Other
OpenAI-compatible endpoints may return only final message content, leaving
reasoning output columns empty even when debug mode is enabled.

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
- remote model metadata fallback; the API request model is
  `trial.remote.served_model_name` when set

The default preset uses a Gemma 4 model for summarization. The specific Gemma
variant that will run successfully may depend on the available GPU type and
memory.

### `trial.local`

Local in-process vLLM runtime settings:

- `engine`: keyword arguments passed to `vllm.LLM(...)`; the package adds
  `model=trial.model_name` separately.
- `generation`: keyword arguments passed to `vllm.SamplingParams(...)`.
- `chat_template_kwargs`: keyword arguments passed to tokenizer chat-template
  rendering.

Additional `engine` and `generation` keys may be included if they are valid vLLM
keyword arguments. vLLM validates those keys when the engine/request is created.

### `trial.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `trial.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model. This setting applies to local vLLM execution
and vLLM server launch helpers; non-vLLM remote endpoints may ignore it or
return no separate reasoning field.

### `trial.remote`

Task-specific remote chat completion request settings:

- `served_model_name`: model name sent in OpenAI-compatible chat completion
  requests. Use this when the endpoint exposes the model under an alias that
  differs from `trial.model_name`.
- `max_tokens_param`: output-token parameter name, usually `max_tokens` for
  vLLM and many compatible endpoints, or `max_completion_tokens` for endpoints
  that require it.
- `max_tokens`: remote output-token budget.
- `request_params`: top-level chat completion request fields sent as-is.
- `extra_body`: provider-specific fields sent as request `extra_body` when
  non-empty.

The package interprets `served_model_name`, `max_tokens_param`, and
`max_tokens`. Values inside `request_params` and `extra_body` are pass-through:
the package does not validate those keys, and the remote endpoint is responsible
for accepting or rejecting them.

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
- remote model metadata fallback; the API request model is
  `patient.remote.served_model_name` when set

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

### `patient.local`

Local in-process vLLM runtime settings. See `trial.local`.

### `patient.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `patient.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model. Non-vLLM remote endpoints may return no
separate reasoning field.

### `patient.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

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

### `match_quality.max_length`

Maximum token length passed to the text-classification checker pipeline.

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

### `exclusion_criteria.max_length`

Maximum token length passed to the text-classification checker pipeline.

## `llm_match_quality`

Configuration for the LLM-based match-quality checker.

### `llm_match_quality.model_name`

Model identifier used for tokenizer/chat-template rendering, local vLLM
execution, and model metadata lookup. In remote mode,
`llm_match_quality.remote.served_model_name` is the API request model when set.

### `llm_match_quality.local`

Local in-process vLLM runtime settings. See `trial.local`.

### `llm_match_quality.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `llm_match_quality.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_match_quality.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

## `llm_exclusion_criteria`

Configuration for the LLM-based exclusion-criteria checker.

### `llm_exclusion_criteria.model_name`

Model identifier used for tokenizer/chat-template rendering, local vLLM
execution, and model metadata lookup. In remote mode,
`llm_exclusion_criteria.remote.served_model_name` is the API request model when
set.

### `llm_exclusion_criteria.local`

Local in-process vLLM runtime settings. See `trial.local`.

### `llm_exclusion_criteria.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `llm_exclusion_criteria.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_exclusion_criteria.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

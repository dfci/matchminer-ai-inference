# Local vs Remote Inference

For LLM-backed steps, `matchminer-ai` supports local and remote inference modes.

## Local Mode

Local mode uses the local in-memory `vLLM` backend by default.

![Figure 2: Trial and Patient Summarization can be run in Local Mode.](../assets/images/local_server_mode.png)

## Remote Mode

Remote mode sends requests to an existing OpenAI-compatible chat completions
endpoint. That endpoint can be a vLLM server, but it does not have to be. Use
an endpoint that is approved for your data and institution.

![Figure 3: Trial and Patient Summarization can be run in Remote Mode.](../assets/images/remote_server_mode.png)

If you want to host the endpoint yourself, `matchminer-ai` provides the
`start_vllm_server()` helper to start a local OpenAI-compatible vLLM server from
package configuration. In that scenario, the remote URL is a localhost URL.

![Figure 4: Running Remote Mode with a server located on your Local Machine.](../assets/images/local_remote_server_mode.png)

# LLM Server Helper

Use `start_vllm_servers()` to start one local vLLM server for each URL in
`config.remote.server_urls`. With the default configuration, this starts one
server.

::: matchminer_ai.llm.vllm_server
    options:
      members:
        - start_vllm_servers

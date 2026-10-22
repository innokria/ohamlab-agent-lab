# Local LLM setup

1. Start a local model server that provides an OpenAI-compatible chat-completions API.
2. Choose a model/server combination that supports tool calling.
3. Set `llm.provider: openai-compatible`, `llm.base_url` to the server's `/v1` endpoint, `llm.model` to the served model name, and `llm.api_key_env` to an environment variable.
4. Export the key variable (a placeholder may work for local servers that do not enforce keys).
5. Run `ohamlab-agent --config config.yaml check`, then `list-tools`, then `ask`.

A model that only generates text and does not support tool calling cannot reliably perform autonomous MCP tool selection.

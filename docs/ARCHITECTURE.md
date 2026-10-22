# Architecture and integration contract

- Agent Lab is a command-line MCP client and LLM tool-use loop.
- The default MCP server is supplied by the PyPI dependency `ohamlab-agent`, launched as `ohamlab mcp-stdio`.
- The same client can optionally connect to a hosted OhamLab MCP server over Streamable HTTP.
- The OhamLab MCP server owns web-search, fetch, storage, history, and other tool implementations.
- Provider adapters map dynamically discovered MCP schemas into the model provider's tool format.
- Runtime calls the model, dispatches requested MCP calls, returns tool results, and repeats up to `max_tool_rounds`.
- Agent definitions are stored locally; agents can be created, cloned, run individually, or run concurrently for the same task.
- The project does not currently expose its agents as an inbound MCP server.

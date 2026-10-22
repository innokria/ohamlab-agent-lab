# Claude instructions for OhamLab Agent Lab


The default local MCP connection uses the installed PyPI dependency `ohamlab-agent` by launching `ohamlab mcp-stdio`; use Streamable HTTP only when you want to connect to a hosted server instead.

Read `README.md` and `AGENTS.md` first. This repository is an MCP client + LLM tool-use runner, not an MCP server.

When asked to use OhamLab tools:
1. Use `config.yaml` to configure a reachable MCP server and provider.
2. Run `ohamlab-agent --config config.yaml list-tools` to discover the server's actual tools.
3. Use `ohamlab-agent --config config.yaml ask "..."` for a model-driven task.
4. Do not invent tool names or claim a tool was called unless execution output confirms it.
5. Keep `OHAMLAB_MCP_TOKEN`, `ANTHROPIC_API_KEY`, and other secrets out of source control and logs.

For Claude Code itself to call the OhamLab MCP server, configure the MCP server in Claude Code using its current documented MCP configuration. That is separate from this project's YAML. This repository currently does not expose Agent Lab agents as an inbound MCP server.


## Multi-agent behavior
- Agents are persisted as JSON definitions under `.ohamlab/agents/`.
- `create-agent` creates a new agent; `clone-agent` copies its role, system prompt, and tool allowlist with a new ID.
- `run-agent` assigns one task and writes a Markdown report.
- `run-swarm --agents ...` executes the same task concurrently for selected agents and writes a summary plus per-agent reports.
- Do not claim dynamic task decomposition, agent-to-agent communication, or consensus: the current swarm is independent parallel execution.
- For local or hosted LLM use, the configured model must support tool calling. MCP tools are discovered from the configured server at runtime.

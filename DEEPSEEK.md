# DeepSeek and DeepSeek Harness instructions

## DeepSeek API as the Agent Lab model

The default local MCP connection uses the installed PyPI dependency `ohamlab-agent` by launching `ohamlab mcp-stdio`; use Streamable HTTP only when you want to connect to a hosted server instead.

Configure `llm.provider: openai-compatible`, `llm.base_url: https://api.deepseek.com`, `llm.api_key_env: DEEPSEEK_API_KEY`, and a currently supported tool-calling model. Export `DEEPSEEK_API_KEY` before running the CLI.

## DeepSeek Harness calling the OhamLab MCP server
DeepSeek Harness should configure the **OhamLab MCP server endpoint**, not this CLI's `config.yaml`. Use the harness's MCP server configuration for Streamable HTTP, the actual `/mcp/` URL, and the authentication header/token required by your deployment. Discover tool names from `tools/list` and check that the harness exposes the MCP tools to its model/runtime.

## Important distinction
The Agent Lab currently acts as an MCP client. It does not currently publish its agents as MCP tools for DeepSeek Harness to call. To make DSH invoke Agent Lab agents as tools, a separate authenticated inbound MCP server adapter would need to be implemented and deployed.


## Multi-agent behavior
- Agents are persisted as JSON definitions under `.ohamlab/agents/`.
- `create-agent` creates a new agent; `clone-agent` copies its role, system prompt, and tool allowlist with a new ID.
- `run-agent` assigns one task and writes a Markdown report.
- `run-swarm --agents ...` executes the same task concurrently for selected agents and writes a summary plus per-agent reports.
- Do not claim dynamic task decomposition, agent-to-agent communication, or consensus: the current swarm is independent parallel execution.
- For local or hosted LLM use, the configured model must support tool calling. MCP tools are discovered from the configured server at runtime.

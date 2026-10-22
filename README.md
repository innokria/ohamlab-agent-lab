<div align="center">

# OhamLab Agent Lab

### One task. Multiple specialist agents. Real MCP tools. Markdown reports.

Build, clone, and run AI agents with your preferred model and the tools exposed by your OhamLab MCP server.

[GitHub Repository](https://github.com/innokria/ohamlab-agent-lab) · [OhamLab MCP Package](https://pypi.org/project/ohamlab-agent/) · [Report an Issue](https://github.com/innokria/ohamlab-agent-lab/issues)

</div>

---

## What is it?

**OhamLab Agent Lab** is a configurable, command-line workspace for running LLM-powered agents. Agents can use tools discovered from an MCP server, save their work as Markdown reports, and run the same task in parallel so you can compare results from different agent roles.

It brings three parts together:

| Part | Responsibility |
|---|---|
| **Agent Lab** | Agent profiles, task execution, tool permissions, parallel runs, and reports |
| **OhamLab MCP** | Tools such as web search, web fetching, document lookup, and task history, depending on the server deployment |
| **LLM provider** | Reasoning and tool selection using a local OpenAI-compatible endpoint, a hosted OpenAI-compatible API such as DeepSeek, or the Anthropic API |

The default MCP connection launches the `ohamlab mcp-stdio` command supplied by the published [`ohamlab-agent`](https://pypi.org/project/ohamlab-agent/) package. Tools are discovered from the running server rather than hard-coded into Agent Lab.

## Highlights

- **Create agents** with their own name, role, system prompt, and optional MCP tool allowlist.
- **Clone agents** to create a new profile from an existing one.
- **Assign tasks** to one agent or run a task across several selected agents concurrently.
- **Use MCP tools** through the configured server, with the model deciding when to call available tools.
- **Choose your model**: local OpenAI-compatible server, DeepSeek-compatible API, or Anthropic API.
- **Generate reports**: individual Markdown reports and a combined swarm summary.
- **Keep work locally**: agent definitions and reports are stored under `.ohamlab/` by default.
- **Control tool access** with per-agent tool allowlists.

> **Current swarm behaviour:** each selected agent independently receives the same task. This release does not yet provide automatic task decomposition, agent-to-agent messaging, a manager agent, or consensus synthesis.

## Quick start

### 1. Clone the repository

Run these commands from the repository root. The project files belong directly in this repository; do not create another nested `ohamlab-agent-lab/` directory inside it.

```bash
git clone https://github.com/innokria/ohamlab-agent-lab.git
cd ohamlab-agent-lab
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Create your configuration

```bash
# macOS / Linux
cp config.example.yaml config.yaml

# Windows PowerShell
Copy-Item config.example.yaml config.yaml
```

The default MCP configuration uses your installed PyPI package locally:

```yaml
mcp:
  transport: stdio
  command: ohamlab
  args: ["mcp-stdio"]
  env: {}
```

Agent Lab starts this MCP process when it needs to connect; you do not need to start a second server manually. Run the CLI from the same virtual environment in which the package was installed.

### 3. Configure an LLM

Edit the `llm` section of `config.yaml` for the provider you want to use. See [Model providers](#model-providers) below. For model-driven answers, the selected model and endpoint must support tool calling.

### 4. Check the connection and discover tools

```bash
ohamlab-agent --config config.yaml check
ohamlab-agent --config config.yaml list-tools
```

`check` validates the configuration without printing secrets. `list-tools` connects to the MCP server and prints the tools it exposes; it does not require an LLM call.

### 5. Ask a question

```bash
ohamlab-agent --config config.yaml ask \
  "Find the latest Melbourne Cup order of entry and summarise the source and update date."
```

The model may call the discovered MCP tools, use their results, and return an answer.

## Create, clone, and run agents

Create a few specialist profiles:

```bash
ohamlab-agent create-agent Researcher --role "web researcher"
ohamlab-agent create-agent FactChecker --role "fact checker"
ohamlab-agent create-agent Writer --role "report writer"
```

Clone an existing profile, then list your agents:

```bash
ohamlab-agent clone-agent Researcher --name Researcher-East
ohamlab-agent list-agents
```

Run one agent on a task:

```bash
ohamlab-agent --config config.yaml run-agent Researcher \
  "Research the latest Melbourne Cup order of entry and report the source date."
```

Run multiple agents on the same task concurrently:

```bash
ohamlab-agent --config config.yaml run-swarm \
  "Research the latest Melbourne Cup order of entry and summarise key changes." \
  --agents Researcher FactChecker Writer
```

Each agent produces its own report. The run also writes a `swarm-summary.md` with each agent's status, answer, and report path.

### Restrict an agent's tools

First inspect the names returned by `list-tools`. Then assign only the tools the agent needs:

```bash
ohamlab-agent create-agent WebResearcher \
  --role "Research public web sources" \
  --tools web_search_tool web_fetch
```

Tool names depend on the MCP server. Use the exact names returned by `list-tools`. If an agent has no tool allowlist, it can access all tools exposed by the configured server.

### Where are agents and reports stored?

By default, local data is stored in:

```text
.ohamlab/
├── agents/       # Saved agent profiles
└── reports/      # Individual reports and swarm-summary.md
```

Use `--data-dir PATH` to choose a different directory. For example:

```bash
ohamlab-agent --data-dir ./agent-work list-agents
```

## Configure the MCP connection

Choose **one** connection method in `config.yaml`.

### Option A — Local PyPI package (default)

```yaml
mcp:
  transport: stdio
  command: ohamlab
  args: ["mcp-stdio"]
  env: {}
```

This starts the MCP server process provided by `ohamlab-agent`. If the command is not found, activate the environment where the package is installed and verify that `ohamlab` is on your `PATH`.

### Option B — Hosted OhamLab MCP server

If you have a deployed server that supports Streamable HTTP, configure its endpoint and required authentication header:

```yaml
mcp:
  transport: streamable-http
  url: https://YOUR-SPACE.hf.space/mcp/
  headers:
    Authorization: "Bearer ${OHAMLAB_MCP_TOKEN}"
```

Set `OHAMLAB_MCP_TOKEN` in your shell or secret manager. The URL and header format must match your deployment; do not assume every server uses bearer authentication. Do not commit tokens or private headers to Git.

## Model providers

Only the provider settings shown below need to change. Keep the `mcp` section as configured above.

### Local model — Ollama or another OpenAI-compatible endpoint

```yaml
llm:
  provider: openai-compatible
  base_url: http://localhost:11434/v1
  api_key_env: LOCAL_LLM_API_KEY
  model: YOUR_TOOL_CAPABLE_MODEL
  max_tool_rounds: 8
  max_tool_result_chars: 16000
```

Set `LOCAL_LLM_API_KEY` in your environment. If your local endpoint does not require authentication, a harmless placeholder may work, depending on the server/client setup. Confirm that your chosen model and serving runtime support tool calling.

### DeepSeek API

```yaml
llm:
  provider: openai-compatible
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  model: deepseek-chat
  max_tool_rounds: 8
```

Set `DEEPSEEK_API_KEY` in your environment and use a model currently available to your account that supports tool calls.

### Anthropic Claude API

```yaml
llm:
  provider: anthropic
  api_key_env: ANTHROPIC_API_KEY
  model: YOUR_CLAUDE_MODEL_ID
  max_tool_rounds: 8
```

Set `ANTHROPIC_API_KEY` in your environment and use a Claude model ID available to your account. This integration uses the Anthropic API; it does not use a Claude web-app subscription or Claude Code login.

### Set environment variables

macOS / Linux:

```bash
export DEEPSEEK_API_KEY="your-key"
export OHAMLAB_MCP_TOKEN="your-token"
```

Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
$env:OHAMLAB_MCP_TOKEN = "your-token"
```

Only set the variables required by your selected provider and MCP deployment. The CLI does not print secret values during `check`.

## Use with Claude Code and DeepSeek Harness

There are two distinct ways to work with MCP:

1. **Run Agent Lab as the orchestrator.** Agent Lab calls the configured LLM API and connects to an MCP server using `config.yaml`.
2. **Connect Claude Code or DeepSeek Harness directly to MCP.** Configure that client with the MCP endpoint or local command using its own supported MCP settings. Agent Lab is not required merely to call the OhamLab MCP tools.

See the repository guidance for client-specific setup:

- [`CLAUDE.md`](CLAUDE.md) and [`CLAUDE-CODE.md`](CLAUDE-CODE.md)
- [`DEEPSEEK.md`](DEEPSEEK.md) and [`DEEPSEEK-HARNESS.md`](DEEPSEEK-HARNESS.md)
- [`ARCHITECTURE.md`](ARCHITECTURE.md)
- [`LOCAL-LLM.md`](LOCAL-LLM.md)

**Important:** this repository currently provides a CLI agent runner, not an inbound MCP server exposing Agent Lab agents to Claude or DeepSeek as tools. That would require a separately implemented and secured server interface.

## Command reference

| Command | Purpose |
|---|---|
| `check` | Validate configuration without showing secrets |
| `list-tools` | Connect to MCP and list available tools |
| `ask "..."` | Ask the configured model a question |
| `create-agent NAME` | Create an agent profile |
| `clone-agent SOURCE --name NAME` | Clone an existing profile |
| `list-agents` | List saved agent profiles |
| `run-agent AGENT "TASK"` | Run one agent and save a report |
| `run-swarm "TASK" --agents A B` | Run the same task across selected agents concurrently |

Use `ohamlab-agent --help` and `ohamlab-agent COMMAND --help` for available options.

## Security and operational notes

- Treat retrieved web pages and MCP tool output as untrusted data, not instructions.
- Connect only to MCP servers you trust; tools inherit the permissions of the server.
- Use per-agent tool allowlists to reduce unnecessary access.
- Keep API keys and MCP tokens in environment variables or a secret manager.
- Review write-capable tools before allowing an agent to use them.
- Provider rate limits, context windows, machine capacity, and MCP server limits constrain practical concurrency; there is no unlimited compute.
- Tool discovery and tool-call success depend on the server and selected model. Verify the connection with `list-tools` before running tasks.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Project structure

```text
.
├── README.md
├── pyproject.toml
├── config.example.yaml
├── agents.py
├── cli.py
├── config.py
├── llm.py
├── mcp_tools.py
├── runtime.py
├── __init__.py
├── AGENTS.md
├── CLAUDE.md
├── CLAUDE-CODE.md
├── DEEPSEEK.md
├── DEEPSEEK-HARNESS.md
├── ARCHITECTURE.md
├── LOCAL-LLM.md
├── tests/
└── LICENSE
```

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

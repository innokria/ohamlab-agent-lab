## pypy
```
pip install ohamlab-agent-lab==7.0.0
```
# OhamLab Agent Lab

Standalone agent platform with Gradio UI, REST API, inbound MCP, optional LLMs,
outbound MCP-compatible tool calls (initially via configured HTTP tools), Redis task
queue and persistent artifacts. It does not require DeepSeek Harness.

## Architecture

- **Agent definitions and reports:** `OHAM_DATA_DIR` (mount a writable HF Storage Bucket at `/data`).
- **Redis:** task queue and task state when `REDIS_URL` is configured.
- **REST API:** `/api/...`
- **Inbound MCP:** `/mcp`
- **Gradio:** `/`
- **Health:** `/health`

## Hugging Face Space deployment

1. Create a Docker Space and put these repository files at the Space repository root.
2. Attach a writable Storage Bucket/volume and mount it at `/data`.
3. Set Space variable `OHAM_DATA_DIR=/data/ohamlab-agent-lab`.
4. Add provider credentials as Space Secrets, e.g. `LLM_API_KEY`.
5. Set the Space Secret `OHAM_MCP_API_KEY` to a long random secret.
6. Optionally set `REDIS_URL` to an external Redis service URL.
7. Restart the Space and check `/health`.

Do not put API keys in the agent definition or in source control.

## Install as a Python package

The package name is `ohamlab-agent-lab` and the Python import package is
`ohamlab_agent_lab`. During development, install the full service and UI extras:

```bash
python -m pip install "ohamlab-agent-lab[all]"
```

Once a release has been published, the core package can be installed with:

```bash
python -m pip install ohamlab-agent-lab
```

The core install does not include the web server, MCP server, Redis client, or Gradio.
For a server install use `ohamlab-agent-lab[server]`; for the UI use
`ohamlab-agent-lab[ui]`; use `[all]` for everything.

### Publish a release

1. Confirm the PyPI project name `ohamlab-agent-lab` is available or that you own it.
2. On PyPI, configure a Trusted Publisher for GitHub owner `innokria`, repository
   `ohamlab-agent-lab`, workflow `publish.yml`, and environment `pypi`.
3. Commit and push the code.
4. Update `version` in `pyproject.toml` for each release.
5. Create and push a matching version tag, for example `v0.1.0`.

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow builds and checks the wheel and source distribution, then publishes
using PyPI Trusted Publishing. It does not require a long-lived PyPI API token.
Do not reuse a version that is already published.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export OHAM_DATA_DIR="$PWD/data"
# Optional OpenAI-compatible LLM:
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
export LLM_API_KEY="your-secret"
# Optional Redis:
export REDIS_URL="redis://localhost:6379/0"
python app.py
```

Open `http://localhost:7860`.

## REST API examples

Create a deterministic workflow agent:

```bash
curl -X POST http://localhost:7860/api/agents \
  -H 'Content-Type: application/json' \
  -H 'X-OhamLab-Key: YOUR_SECRET' \
  -d '{"name":"Task Reporter","description":"Formats a task as a report","execution_mode":"workflow","workflow":"report"}'
```

List agents (protected by the same OhamLab header as MCP):

```bash
curl http://localhost:7860/api/agents \
  -H 'X-OhamLab-Key: YOUR_SECRET'
```

All `/api/agents...`, `/api/tasks...`, and `/mcp...` endpoints require
`X-OhamLab-Key`, matching `OHAM_MCP_API_KEY`. `/health` and the Gradio UI are
not protected by this shared-key middleware. Never log or commit the secret.

Run an agent:

```bash
curl -X POST http://localhost:7860/api/agents/AGENT_ID/run \
  -H 'Content-Type: application/json' \
  -H 'X-OhamLab-Key: YOUR_SECRET' \
  -d '{"task":"Summarize the first implementation milestone","wait":true}'
```

Queue a long-running task:

```bash
curl -X POST http://localhost:7860/api/tasks \
  -H 'Content-Type: application/json' \
  -H 'X-OhamLab-Key: YOUR_SECRET' \
  -d '{"agent_id":"AGENT_ID","task":"Write a project plan"}'
curl http://localhost:7860/api/tasks/TASK_ID
```

## Inbound MCP

The app mounts a Streamable HTTP MCP server at `/mcp`. Configure an MCP client to use:

`https://YOUR-SPACE.hf.space/mcp`

This starter exposes `list_agents`, `run_agent`, `start_agent_task`, `get_task_status`,
and `get_task_result`. MCP HTTP requests must include `X-OhamLab-Key: YOUR_SECRET`. The server compares it
against `OHAM_MCP_API_KEY` using a constant-time comparison. A missing/mismatched key
returns HTTP 401; an unset server secret returns HTTP 503. The same header is required
for REST agent and task endpoints.

## LLMs optional

- `workflow` agents run registered deterministic handlers and do not need an LLM.
- `llm` agents use an OpenAI-compatible chat completions endpoint if configured.
- No arbitrary Python code is executed by the runtime.
- Provider-specific adapters can be added behind `AgentRuntime`.

## Outbound OhamLab Plugin MCP

Configure the existing OhamLab Plugin's remote MCP endpoint as a tool source in the
future outbound-MCP adapter. This starter deliberately keeps agent execution and
transport independent. It does not assume the plugin's private authentication or
transport configuration. See `docs/ROADMAP.md`.

## Redis and task durability

When `REDIS_URL` is configured, task records and queue items are stored in Redis.
Agent definitions and reports remain in `OHAM_DATA_DIR`. If Redis is unavailable,
the app fails over to a local file/queue mode and displays a warning. Local queue
mode is for testing and is not durable across restarts.

Tasks running during process shutdown are marked `interrupted`; they are not
automatically rerun because they may have side effects. A future always-on worker
service is required for guaranteed execution while a Space is stopped.

## API docs

After launch, open `/docs` for interactive OpenAPI documentation.

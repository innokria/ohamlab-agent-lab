# Roadmap

## Phase 1: standalone smoke test
- Build Docker image.
- Create a workflow agent in Gradio.
- Run a task without configuring an LLM.
- Verify the report is written to `OHAM_DATA_DIR`.

## Phase 2: provider adapters
- Add explicit adapters for OpenAI-compatible APIs, Anthropic, and local endpoints.
- Keep API keys in environment variables or a secret manager.
- Add tool-call support only after validating the selected provider's API.

## Phase 3: outbound MCP
- Use the official MCP client SDK to connect to OhamLab Plugin's Streamable HTTP endpoint.
- Discover tools, apply an allowlist per agent, and handle auth/timeouts.
- Treat tool results as untrusted input.

## Phase 4: durability and workers
- Redis-backed task queue and task status.
- Persist checkpoints and artifacts to the mounted Bucket.
- Add a separate worker process/service for continuous work.
- Add lease expiry, idempotency keys, retries with limits, cancellation and observability.

## Phase 5: production security
- API key or bearer authentication.
- Per-agent access policy and tool permissions.
- Request/task quotas, concurrency limits and audit events.
- Secure inbound MCP endpoint.
- Add sandboxed code execution only as a separate isolated service.

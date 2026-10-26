"""Remote OhamLab MCP tool-calling integration for Agent Lab.

Requires: mcp>=1.0, openai>=1.0
Environment: OHAMLAB_MCP_URL, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
Optional: OHAMLAB_MCP_KEY, OHAMLAB_MCP_AUTH_HEADER, OHAMLAB_MCP_TIMEOUT
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


@asynccontextmanager
async def remote_mcp_session():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    import httpx

    url = os.environ.get("OHAMLAB_MCP_URL", "").strip()
    if not url.startswith(("https://", "http://")):
        raise RuntimeError("OHAMLAB_MCP_URL must be a valid HTTP(S) MCP endpoint")

    headers = {}
    key = os.getenv("OHAMLAB_MCP_KEY", "").strip()
    if key:
        headers[os.getenv("OHAMLAB_MCP_AUTH_HEADER", "X-OhamLab-Key")] = key
    timeout = float(os.getenv("OHAMLAB_MCP_TIMEOUT", "60"))
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as client:
        async with streamablehttp_client(url, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def discover_remote_tools() -> list[dict]:
    """Useful for testing connection and discovering server-advertised tools."""
    async with remote_mcp_session() as session:
        result = await session.list_tools()
        return [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                for t in result.tools]


async def run_with_remote_tools(agent: Any, task: str) -> str:
    """Use an OpenAI-compatible LLM to choose and call the remote MCP tools."""
    from openai import AsyncOpenAI

    base_url = (getattr(agent, "base_url", None) or os.getenv("LLM_BASE_URL", "")).strip()
    model = (getattr(agent, "model", None) or os.getenv("LLM_MODEL", "")).strip()
    key_name = getattr(agent, "api_key_env", None) or "LLM_API_KEY"
    api_key = os.getenv(key_name) or os.getenv("LLM_API_KEY")
    if not model or not api_key:
        raise RuntimeError(f"Remote MCP research requires LLM_MODEL (or agent model) and {key_name}.")
    max_calls = max(1, min(20, int(os.getenv("OHAMLAB_MAX_TOOL_CALLS", "8"))))

    async with remote_mcp_session() as session:
        available = (await session.list_tools()).tools
        if not available:
            raise RuntimeError("Remote OhamLab MCP endpoint advertised no tools")
        tools = [{"type": "function", "function": {"name": t.name,
                  "description": t.description or t.name, "parameters": t.inputSchema or {"type": "object", "properties": {}}}}
                 for t in available]
        allowed = {t.name for t in available}
        messages = [
            {"role": "system", "content": (getattr(agent, "system_prompt", None) or "You are a research assistant.") +
             " Use available MCP tools for real web research. Do not invent search results or citations. "
             "Clearly distinguish missing evidence. Return a useful final answer with source URLs."},
            {"role": "user", "content": task},
        ]
        client_args = {"api_key": api_key, "timeout": 120.0}
        if base_url:
            client_args["base_url"] = base_url
        async with AsyncOpenAI(**client_args) as llm:
            for _ in range(max_calls + 1):
                reply = await llm.chat.completions.create(model=model, messages=messages,
                                                          tools=tools, tool_choice="auto")
                message = reply.choices[0].message
                calls = message.tool_calls or []
                messages.append(message.model_dump(exclude_none=True))
                if not calls:
                    answer = message.content or ""
                    if not answer.strip():
                        raise RuntimeError("LLM returned no final answer")
                    return answer
                for call in calls:
                    name = call.function.name
                    if name not in allowed:
                        output = {"error": f"Tool {name!r} not advertised by MCP server"}
                    else:
                        try:
                            args = json.loads(call.function.arguments or "{}")
                            if not isinstance(args, dict):
                                raise ValueError("Tool arguments must be a JSON object")
                            result = await session.call_tool(name, arguments=args)
                            output = _dump(result)
                        except Exception as exc:
                            output = {"error": str(exc)}
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": json.dumps(output, ensure_ascii=False, default=str)[:40000]})
            raise RuntimeError(f"Agent exceeded maximum of {max_calls} tool-calling rounds")


def install_runtime_integration(runtime: Any) -> None:
    """Wrap the existing shared runtime, so UI/API/MCP callers can use remote tools.

    Only agents explicitly configured for MCP research are intercepted. Existing
    agent workflows remain unchanged otherwise.
    """
    if getattr(runtime, "_ohamlab_remote_mcp_installed", False):
        return
    original = runtime.run

    async def run(agent: Any, task: str, *args, **kwargs):
        mode = str(getattr(agent, "execution_mode", "") or "").lower()
        workflow = str(getattr(agent, "workflow", "") or "").lower()
        if os.getenv("OHAMLAB_MCP_URL") and (workflow in {"mcp_research", "remote_mcp"} or mode == "mcp_research"):
            return await run_with_remote_tools(agent, task)
        return await original(agent, task, *args, **kwargs)

    runtime.run = run
    runtime._ohamlab_remote_mcp_installed = True

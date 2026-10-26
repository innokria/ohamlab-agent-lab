"""Remote OhamLab MCP tool-calling integration for Agent Lab.

Requires: mcp>=1.0, openai>=1.0
Environment: OHAMLAB_MCP_URL (required for MCP workflows); LLM_* only for mcp_research
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


async def run_mcp_search(task: str) -> str:
    """LLM-free search using the remote MCP server; return actual tool output."""
    query = task.strip()
    if not query:
        raise ValueError("Search query cannot be empty")
    async with remote_mcp_session() as session:
        tools = (await session.list_tools()).tools
        # Prefer the server's search tool, not an arbitrary web-related tool.
        candidates = sorted(
            (t for t in tools if "search" in t.name.lower()),
            key=lambda t: (t.name.lower() not in {"search", "web_search", "ohamlab_search"}, t.name),
        )
        if not candidates:
            raise RuntimeError("Remote MCP server does not advertise a search tool")
        for tool in candidates:
            schema = tool.inputSchema or {}
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            for field in ("query", "q", "search_query", "text"):
                if field in props and required.issubset({field}):
                    result = await session.call_tool(tool.name, arguments={field: query})
                    if result.isError:
                        raise RuntimeError(f"MCP tool {tool.name} failed: {_dump(result)}")
                    blocks = []
                    for part in result.content:
                        if getattr(part, "type", "") == "text":
                            blocks.append(part.text)
                        else:
                            blocks.append(json.dumps(_dump(part), ensure_ascii=False, default=str))
                    if getattr(result, "structuredContent", None):
                        blocks.append(json.dumps(result.structuredContent, ensure_ascii=False, default=str))
                    return "\n\n".join(blocks) or "Search returned no content."
        raise RuntimeError(
            "No compatible search tool input schema found. Available: "
            + json.dumps([{"name": t.name, "inputSchema": t.inputSchema} for t in candidates], default=str)
        )


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
        if workflow == "mcp_search" or mode == "mcp_search":
            return await run_mcp_search(task)
        if workflow in {"mcp_research", "remote_mcp"} or mode == "mcp_research":
            return await run_with_remote_tools(agent, task)
        return await original(agent, task, *args, **kwargs)

    runtime.run = run
    runtime._ohamlab_remote_mcp_installed = True

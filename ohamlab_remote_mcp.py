
"""Remote OhamLab MCP integration for Agent Lab.

Workflows:
    mcp_search   - Direct MCP web search without an LLM.
    mcp_research - LLM-powered research with MCP tool calling.

Required:
    pip install mcp httpx

Optional for mcp_research:
    pip install openai

Environment:
    OHAMLAB_MCP_URL
    OHAMLAB_MCP_KEY
    OHAMLAB_MCP_AUTH_HEADER
    OHAMLAB_MCP_TIMEOUT

Optional LLM configuration:
    LLM_BASE_URL
    LLM_MODEL
    LLM_API_KEY
    OHAMLAB_MAX_TOOL_CALLS
"""

from __future__ import annotations

import json
import os

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any
import logging

logger = logging.getLogger(__name__)

def _root_errors(exc: BaseException) -> list[BaseException]:
    """Flatten AnyIO ExceptionGroups to expose underlying HTTP/MCP errors."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _root_errors(sub)]
    return [exc]

def _explain_connection_error(exc: BaseException, url: str) -> str:
    errors = _root_errors(exc)
    details = "; ".join(f"{type(e).__name__}: {e}" for e in errors)
    if any("401" in str(e) or "403" in str(e) for e in errors):
        hint = "Check OHAMLAB_MCP_KEY and X-OhamLab-Key authentication."
    elif any("404" in str(e) for e in errors):
        hint = "Check the MCP URL path (including trailing /mcp/)."
    elif any("307" in str(e) or "308" in str(e) for e in errors):
        hint = "Use the canonical endpoint URL without redirects."
    elif any("timeout" in type(e).__name__.lower() or "timed out" in str(e).lower() for e in errors):
        hint = "Check server health and OHAMLAB_MCP_TIMEOUT."
    else:
        hint = "Check MCP server logs and the full exception traceback."
    return f"Remote OhamLab MCP connection to {url} failed: {details}. {hint}"


def _dump(value: Any) -> Any:
    """Convert MCP SDK objects to JSON-compatible data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json",
            exclude_none=True,
        )

    return value


@asynccontextmanager
async def remote_mcp_session():
    """Connect to the configured remote OhamLab MCP server."""

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    url = os.getenv("OHAMLAB_MCP_URL", "").strip()

    if not url.startswith(("https://", "http://")):
        raise RuntimeError(
            "OHAMLAB_MCP_URL must be a valid HTTP(S) MCP endpoint"
        )

    key = os.getenv("OHAMLAB_MCP_KEY", "").strip()

    timeout = float(
        os.getenv("OHAMLAB_MCP_TIMEOUT", "60")
    )

    headers = {}

    if key:
        header_name = os.getenv(
            "OHAMLAB_MCP_AUTH_HEADER",
            "X-OhamLab-Key",
        )

        headers[header_name] = key

    try:
        async with streamablehttp_client(
            url=url,
            headers=headers,
            timeout=timedelta(seconds=timeout),
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception as exc:
        # MCP's AnyIO transport can wrap HTTP errors in an ExceptionGroup.
        # Keep the original traceback in logs, but surface the actual cause.
        logger.exception("Remote MCP session failed: %s", url)
        raise RuntimeError(_explain_connection_error(exc, url)) from exc


async def discover_remote_tools() -> list[dict]:
    """Discover tools advertised by the OhamLab MCP server."""

    async with remote_mcp_session() as session:

        response = await session.list_tools()

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in response.tools
        ]


def _search_arguments(tool: Any, query: str) -> dict:
    """Build search arguments from the advertised MCP schema."""

    schema = tool.inputSchema or {}
    properties = schema.get("properties", {})

    for field in ("query", "q", "search_query", "search_term"):
        if field in properties:
            return {field: query}

    raise RuntimeError(
        f"Search tool '{tool.name}' does not advertise "
        "a supported query parameter. "
        f"Schema: {json.dumps(schema, default=str)}"
    )


def _format_mcp_result(result: Any) -> str:
    """Convert a tool result to a readable string."""

    if getattr(result, "isError", False):
        raise RuntimeError(
            f"Remote MCP tool returned an error: {_dump(result)}"
        )

    parts = []

    for item in getattr(result, "content", []):
        if getattr(item, "type", "") == "text":
            parts.append(item.text)
        else:
            parts.append(
                json.dumps(
                    _dump(item),
                    ensure_ascii=False,
                    default=str,
                )
            )

    if parts:
        return "\n\n".join(parts)

    structured = getattr(result, "structuredContent", None)

    if structured is not None:
        return json.dumps(
            structured,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return json.dumps(
        _dump(result),
        indent=2,
        ensure_ascii=False,
        default=str,
    )


async def run_mcp_search(task: str) -> str:
    """Execute a direct MCP search without requiring an LLM."""

    query = task.strip()

    if not query:
        raise ValueError("Search query cannot be empty")

    async with remote_mcp_session() as session:

        response = await session.list_tools()
        available = response.tools

        if not available:
            raise RuntimeError(
                "Remote OhamLab MCP endpoint advertised no tools"
            )

        search_tools = [
            tool
            for tool in available
            if "search" in tool.name.lower()
        ]

        if not search_tools:
            names = [tool.name for tool in available]

            raise RuntimeError(
                "No search tool found on OhamLab MCP server. "
                f"Available tools: {names}"
            )

        # Prefer an exact search tool name when available.
        preferred_names = (
            "web_search",
            "search_web",
            "search",
            "ohamlab_search",
        )

        search_tool = next(
            (
                tool
                for name in preferred_names
                for tool in search_tools
                if tool.name.lower() == name
            ),
            search_tools[0],
        )

        arguments = _search_arguments(
            search_tool,
            query,
        )

        result = await session.call_tool(
            search_tool.name,
            arguments=arguments,
        )

        return _format_mcp_result(result)


async def run_with_remote_tools(
    agent: Any,
    task: str,
) -> str:
    """Run LLM-powered research using remote MCP tools."""

    from openai import AsyncOpenAI

    base_url = (
        getattr(agent, "base_url", None)
        or os.getenv("LLM_BASE_URL", "")
    ).strip()

    model = (
        getattr(agent, "model", None)
        or os.getenv("LLM_MODEL", "")
    ).strip()

    key_name = (
        getattr(agent, "api_key_env", None)
        or "LLM_API_KEY"
    )

    api_key = (
        os.getenv(key_name)
        or os.getenv("LLM_API_KEY")
    )

    if not model or not api_key:
        raise RuntimeError(
            "Remote MCP research requires LLM_MODEL "
            f"(or agent model) and {key_name}."
        )

    max_calls = max(
        1,
        min(
            20,
            int(os.getenv("OHAMLAB_MAX_TOOL_CALLS", "8")),
        ),
    )

    async with remote_mcp_session() as session:

        response = await session.list_tools()
        available = response.tools

        if not available:
            raise RuntimeError(
                "Remote OhamLab MCP endpoint advertised no tools"
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (
                        tool.description or tool.name
                    ),
                    "parameters": (
                        tool.inputSchema
                        or {
                            "type": "object",
                            "properties": {},
                        }
                    ),
                },
            }
            for tool in available
        ]

        allowed = {
            tool.name
            for tool in available
        }

        system_prompt = (
            getattr(agent, "system_prompt", None)
            or "You are a research assistant."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\nUse available MCP tools for real "
                    "web research. Do not invent search "
                    "results or citations. Clearly distinguish "
                    "missing evidence. Return a useful final "
                    "answer with source URLs."
                ),
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        client_args = {
            "api_key": api_key,
            "timeout": 120.0,
        }

        if base_url:
            client_args["base_url"] = base_url

        async with AsyncOpenAI(**client_args) as llm:

            for _ in range(max_calls + 1):

                reply = await llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

                message = reply.choices[0].message
                calls = message.tool_calls or []

                messages.append(
                    message.model_dump(
                        exclude_none=True
                    )
                )

                if not calls:

                    answer = message.content or ""

                    if not answer.strip():
                        raise RuntimeError(
                            "LLM returned no final answer"
                        )

                    return answer

                for call in calls:

                    name = call.function.name

                    if name not in allowed:

                        output = {
                            "error": (
                                f"Tool {name!r} not "
                                "advertised by MCP server"
                            )
                        }

                    else:

                        try:
                            arguments = json.loads(
                                call.function.arguments or "{}"
                            )

                            if not isinstance(arguments, dict):
                                raise ValueError(
                                    "Tool arguments must be "
                                    "a JSON object"
                                )

                            result = await session.call_tool(
                                name,
                                arguments=arguments,
                            )

                            output = _dump(result)

                        except Exception as exc:

                            output = {
                                "error": str(exc)
                            }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                output,
                                ensure_ascii=False,
                                default=str,
                            )[:40000],
                        }
                    )

            raise RuntimeError(
                f"Agent exceeded maximum of "
                f"{max_calls} tool-calling rounds"
            )


def install_runtime_integration(runtime: Any) -> None:
    """Enable MCP workflows on the existing Agent Lab runtime.

    Applies to tasks submitted through Gradio, REST and MCP
    when those paths use this shared runtime.
    """

    if getattr(
        runtime,
        "_ohamlab_remote_mcp_installed",
        False,
    ):
        return

    original = runtime.run

    async def run(
        agent: Any,
        task: str,
        *args,
        **kwargs,
    ):
        mode = str(
            getattr(agent, "execution_mode", "") or ""
        ).lower()

        workflow = str(
            getattr(agent, "workflow", "") or ""
        ).lower()

        if workflow == "mcp_search" or mode == "mcp_search":
            return await run_mcp_search(task)

        if (
            workflow in {"mcp_research", "remote_mcp"}
            or mode == "mcp_research"
        ):
            return await run_with_remote_tools(
                agent,
                task,
            )

        return await original(
            agent,
            task,
            *args,
            **kwargs,
        )

    runtime.run = run
    runtime._ohamlab_remote_mcp_installed = True

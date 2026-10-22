from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    return {
        "name": tool.name,
        "description": getattr(tool, "description", None) or "",
        "inputSchema": _jsonable(schema),
    }


class MCPTools:
    """Async MCP client that discovers tools and calls them within one session."""

    def __init__(self, config):
        self.config = config
        self._stack = None
        self.session: ClientSession | None = None
        self.tools: list[dict[str, Any]] = []

    async def __aenter__(self):
        if self.config.transport == "stdio":
            child_env = os.environ.copy()
            child_env.update(self.config.env)
            params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=child_env,
            )
            self._stack = stdio_client(params)
            read_stream, write_stream = await self._stack.__aenter__()
        else:
            self._stack = streamablehttp_client(
                self.config.url,
                headers=self.config.headers or None,
            )
            read_stream, write_stream, _get_session_id = await self._stack.__aenter__()

        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        response = await self.session.list_tools()
        self.tools = [_tool_schema(tool) for tool in response.tools]
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session is not None:
            await self.session.__aexit__(exc_type, exc, tb)
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError("MCP session is not connected")
        result = await self.session.call_tool(name, arguments=arguments)
        payload = {
            "isError": bool(getattr(result, "isError", False)),
            "content": [
                {
                    "type": getattr(item, "type", "unknown"),
                    "text": getattr(item, "text", None),
                    "data": getattr(item, "data", None),
                    "mimeType": getattr(item, "mimeType", None),
                }
                for item in (getattr(result, "content", None) or [])
            ],
            "structuredContent": _jsonable(getattr(result, "structuredContent", None)),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

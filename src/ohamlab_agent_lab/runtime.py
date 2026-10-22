from __future__ import annotations

import json
from typing import Any

from .llm import LLM
from .mcp_tools import MCPTools


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[tool result truncated]"


class AgentRuntime:
    def __init__(self, config):
        self.config = config
        self.llm = LLM(config.llm)

    async def list_tools(self) -> list[dict[str, Any]]:
        async with MCPTools(self.config.mcp) as mcp:
            return mcp.tools

    async def ask(self, question: str, *, system_prompt: str | None = None, allowed_tools: list[str] | None = None) -> str:
        async with MCPTools(self.config.mcp) as mcp:
            available_tools = mcp.tools
            if allowed_tools:
                requested = set(allowed_tools)
                available_tools = [tool for tool in mcp.tools if tool["name"] in requested]
                missing = requested - {tool["name"] for tool in available_tools}
                if missing:
                    raise ValueError("Configured agent tools not exposed by MCP server: " + ", ".join(sorted(missing)))
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt or self.config.agent.system_prompt},
                {"role": "user", "content": question},
            ]
            tool_by_name = {tool["name"]: tool for tool in available_tools}

            for _round in range(self.config.llm.max_tool_rounds):
                response = await self.llm.respond(messages, available_tools)
                calls = response["tool_calls"]

                if not calls:
                    return response["text"].strip() or "(The model returned no text.)"

                if self.config.llm.provider == "openai-compatible":
                    raw = response["raw_message"]
                    messages.append({
                        "role": "assistant",
                        "content": raw.content,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.function.name,
                                    "arguments": c.function.arguments,
                                },
                            }
                            for c in (raw.tool_calls or [])
                        ],
                    })
                    for call in calls:
                        name = call["name"]
                        if name not in tool_by_name:
                            result = json.dumps({"error": f"Unknown tool: {name}"})
                        else:
                            result = await mcp.call(name, call["arguments"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": _truncate(result, self.config.llm.max_tool_result_chars),
                        })
                else:
                    raw = response["raw_message"]
                    messages.append({
                        "role": "assistant",
                        "content": [
                            block.model_dump() if hasattr(block, "model_dump") else block
                            for block in raw.content
                        ],
                    })
                    tool_results = []
                    for call in calls:
                        name = call["name"]
                        if name not in tool_by_name:
                            result = json.dumps({"error": f"Unknown tool: {name}"})
                            is_error = True
                        else:
                            result = await mcp.call(name, call["arguments"])
                            is_error = False
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": _truncate(result, self.config.llm.max_tool_result_chars),
                            "is_error": is_error,
                        })
                    messages.append({"role": "user", "content": tool_results})

            return (
                "Stopped because the maximum tool-call rounds were reached. "
                "Try a narrower question or increase llm.max_tool_rounds."
            )

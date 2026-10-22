from __future__ import annotations

import json
import os
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


def openai_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
            },
        }
        for tool in mcp_tools
    ]


def anthropic_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
        }
        for tool in mcp_tools
    ]


class LLM:
    def __init__(self, config):
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if config.provider == "openai-compatible":
            # Some local servers ignore the key but the OpenAI client requires a value.
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=api_key or "local-no-key",
            )
        else:
            if not api_key:
                raise ValueError(
                    f"Environment variable {config.api_key_env} is not set for Anthropic."
                )
            self.client = AsyncAnthropic(api_key=api_key)

    async def respond(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        if self.config.provider == "openai-compatible":
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=openai_tools(tools) or None,
                tool_choice="auto" if tools else None,
            )
            msg = response.choices[0].message
            calls = []
            for call in (msg.tool_calls or []):
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": args,
                })
            return {
                "text": msg.content or "",
                "tool_calls": calls,
                "raw_message": msg,
            }

        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=4096,
            system=messages[0]["content"] if messages and messages[0]["role"] == "system" else "",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in messages[1 if messages and messages[0]["role"] == "system" else 0:]
            ],
            tools=anthropic_tools(tools) or [],
        )
        text_parts = []
        calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return {
            "text": "\n".join(text_parts),
            "tool_calls": calls,
            "raw_message": response,
        }

from __future__ import annotations

import os
from typing import Awaitable, Callable

import httpx

from .models import AgentDefinition

Workflow = Callable[[AgentDefinition, str], Awaitable[str]]


async def report_workflow(agent: AgentDefinition, task: str) -> str:
    """A deterministic workflow; no model or API key is required."""
    return (
        f"## Task received\n\n{task}\n\n"
        f"## Agent\n\n{agent.name}: {agent.description or 'No description provided.'}\n\n"
        "## Execution mode\n\n"
        "This is the built-in deterministic report workflow. It did not call an LLM "
        "or execute arbitrary code. Register a dedicated workflow handler to perform "
        "additional deterministic actions."
    )


async def echo_workflow(agent: AgentDefinition, task: str) -> str:
    return f"{agent.name} received this task:\n\n{task}"


class AgentRuntime:
    def __init__(self):
        self.workflows: dict[str, Workflow] = {
            "report": report_workflow,
            "echo": echo_workflow,
        }

    def register_workflow(self, name: str, handler: Workflow) -> None:
        if not name.strip():
            raise ValueError("Workflow name cannot be empty")
        self.workflows[name] = handler

    async def run(self, agent: AgentDefinition, task: str) -> str:
        if agent.execution_mode in ("workflow", "custom"):
            handler = self.workflows.get(agent.workflow)
            if not handler:
                raise ValueError(f"Unknown workflow: {agent.workflow}")
            return await handler(agent, task)

        if agent.execution_mode == "llm":
            return await self._run_llm(agent, task)

        raise ValueError(f"Unsupported execution mode: {agent.execution_mode}")

    async def _run_llm(self, agent: AgentDefinition, task: str) -> str:
        base_url = (agent.base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        model = agent.model or os.getenv("LLM_MODEL", "")
        api_key = os.getenv(agent.api_key_env) or ""
        if not base_url or not model:
            raise ValueError(
                "LLM agent requires base_url and model on the agent or "
                "LLM_BASE_URL and LLM_MODEL environment variables."
            )
        if not api_key:
            raise ValueError(
                f"Missing API key: set the environment variable {agent.api_key_env}."
            )

        # OpenAI-compatible chat completions. Tool calling is intentionally not
        # implemented here yet; add provider-specific tool loops behind this adapter.
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": task},
            ],
            "temperature": 0.2,
        }
        timeout = httpx.Timeout(90.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Provider response did not contain choices[0].message.content") from exc

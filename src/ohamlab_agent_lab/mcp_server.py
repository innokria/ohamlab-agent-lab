from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .api import runtime, service, storage, task_manager
from .models import CreateAgentRequest

mcp = FastMCP("OhamLab Agent Lab")


@mcp.tool()
def list_agents() -> list[dict]:
    """List agents published for MCP invocation."""
    return [
        agent.model_dump()
        for agent in service.list_agents()
        if agent.published_mcp
    ]


@mcp.tool()
def create_agent(
    name: str,
    description: str = "",
    workflow: str = "mcp_search",
    system_prompt: str = "You are a helpful research assistant. Cite source URLs and do not invent findings.",
) -> dict:
    """Create and publish a workflow agent for subsequent MCP task execution.

    Supported workflows depend on the Agent Lab runtime configuration.
    Defaults to the existing MCP web-search workflow.
    """
    if not name.strip():
        return {"error": "Agent name cannot be empty"}
    if not workflow.strip():
        return {"error": "Workflow cannot be empty"}

    try:
        request = CreateAgentRequest(
            name=name.strip(),
            description=description.strip(),
            execution_mode="workflow",
            workflow=workflow.strip(),
            system_prompt=system_prompt.strip(),
            published_api=True,
            published_mcp=True,
        )
        agent = service.create_agent(request)
        return agent.model_dump(mode="json")
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
async def run_agent(agent_id: str, task: str) -> dict:
    """Run a published agent and return its result synchronously."""
    agent = service.get_agent(agent_id)
    if not agent:
        return {"error": "Agent not found"}
    if not agent.published_mcp:
        return {"error": "Agent is not published through MCP"}
    try:
        result = await runtime.run(agent, task)
        return {"agent_id": agent_id, "status": "completed", "result": result}
    except Exception as exc:
        return {"agent_id": agent_id, "status": "failed", "error": str(exc)}


@mcp.tool()
def start_agent_task(agent_id: str, task: str) -> dict:
    """Queue an agent task and return its task ID immediately."""
    agent = service.get_agent(agent_id)
    if not agent:
        return {"error": "Agent not found"}
    if not agent.published_mcp:
        return {"error": "Agent is not published through MCP"}
    record = task_manager.submit(agent, task)
    return {"task_id": record.task_id, "status": record.status}


@mcp.tool()
def get_task_status(task_id: str) -> dict:
    """Get the current status of an asynchronous task."""
    record = task_manager.get(task_id)
    return record.model_dump() if record else {"error": "Task not found"}


@mcp.tool()
def get_task_result(task_id: str) -> dict:
    """Retrieve the result or error from a task."""
    record = task_manager.get(task_id)
    if not record:
        return {"error": "Task not found"}
    return {
        "task_id": record.task_id,
        "status": record.status,
        "result": record.result,
        "error": record.error,
    }


def main():
    # For local standalone MCP use: python -m ohamlab_agent_lab.mcp_server
    # The default transport can be overridden using MCP_TRANSPORT.
    mcp.run(transport=os.getenv("MCP_TRANSPORT", "stdio"))


if __name__ == "__main__":
    main()

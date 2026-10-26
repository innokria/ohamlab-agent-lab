import asyncio

from ohamlab_agent_lab.models import AgentDefinition
from ohamlab_agent_lab.runtime import AgentRuntime


def test_workflow_runs_without_llm():
    runtime = AgentRuntime()
    agent = AgentDefinition(
        id="test-agent",
        name="Test",
        execution_mode="workflow",
        workflow="echo",
    )
    result = asyncio.run(runtime.run(agent, "hello"))
    assert "hello" in result

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentDefinition(BaseModel):
    id: str
    name: str
    description: str = ""
    execution_mode: Literal["workflow", "llm", "custom"] = "workflow"
    workflow: str = "report"
    system_prompt: str = "You are a helpful assistant."
    provider: str = "openai-compatible"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "LLM_API_KEY"
    allowed_tools: list[str] = Field(default_factory=list)
    published_api: bool = True
    published_mcp: bool = True
    created_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateAgentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    execution_mode: Literal["workflow", "llm", "custom"] = "workflow"
    workflow: str = "report"
    system_prompt: str = "You are a helpful assistant."
    provider: str = "openai-compatible"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "LLM_API_KEY"
    allowed_tools: list[str] = Field(default_factory=list)
    published_api: bool = True
    published_mcp: bool = True


class RunAgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=100_000)
    wait: bool = True


class CreateTaskRequest(BaseModel):
    agent_id: str
    task: str = Field(min_length=1, max_length=100_000)


class TaskRecord(BaseModel):
    task_id: str
    agent_id: str
    task: str
    status: Literal[
        "queued", "running", "waiting_for_input", "completed",
        "failed", "cancelled", "interrupted"
    ] = "queued"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: str = ""
    report_path: str = ""
    error: str = ""

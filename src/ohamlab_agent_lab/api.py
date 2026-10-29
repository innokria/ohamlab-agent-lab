from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .models import CreateAgentRequest, CreateTaskRequest, RunAgentRequest
from .auth import install_auth_middleware
from .runtime import AgentRuntime
from .service import AgentService
from .storage import FileStorage
from .tasks import TaskManager

def _default_data_dir() -> Path:
    configured = os.getenv("OHAM_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.getenv("SPACE_ID"):
        return Path("/data/ohamlab-agent-lab")
    return Path.home() / ".ohamlab-agent-lab"


DATA_DIR = _default_data_dir()
storage = FileStorage(DATA_DIR)
runtime = AgentRuntime()
task_manager = TaskManager(storage, runtime, os.getenv("REDIS_URL", ""))
service = AgentService(storage, runtime, task_manager)


class CloneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


def create_api() -> FastAPI:
    api = FastAPI(title="OhamLab Agent Lab API", version="7.0.0")
    install_auth_middleware(api)

    @api.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "ohamlab-agent-lab",
            "storage": str(DATA_DIR),
            "task_backend": task_manager.backend_status,
        }

    @api.post("/api/agents")
    async def create_agent(body: CreateAgentRequest):
        try:
            agent = service.create_agent(body)
            return agent.model_dump()
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @api.get("/api/agents")
    async def list_agents():
        return [agent.model_dump() for agent in service.list_agents()]

    @api.get("/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        agent = service.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, "Agent not found")
        return agent.model_dump()

    @api.post("/api/agents/{agent_id}/clone")
    async def clone_agent(agent_id: str, body: CloneRequest):
        try:
            return service.clone_agent(agent_id, body.name).model_dump()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @api.post("/api/agents/{agent_id}/run")
    async def run_agent(agent_id: str, body: RunAgentRequest):
        agent = service.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, "Agent not found")
        if not agent.published_api:
            raise HTTPException(403, "This agent is not published through the API")
        if body.wait:
            try:
                result = await runtime.run(agent, body.task)
                return {"agent_id": agent_id, "status": "completed", "result": result}
            except Exception as exc:
                raise HTTPException(500, str(exc)) from exc
        task = task_manager.submit(agent, body.task)
        return {"task_id": task.task_id, "status": task.status}

    @api.post("/api/tasks", status_code=202)
    async def create_task(body: CreateTaskRequest):
        agent = service.get_agent(body.agent_id)
        if not agent:
            raise HTTPException(404, "Agent not found")
        if not agent.published_api:
            raise HTTPException(403, "This agent is not published through the API")
        task = task_manager.submit(agent, body.task)
        return task.model_dump()

    @api.get("/api/tasks")
    async def list_tasks():
        return [task.model_dump() for task in task_manager.list()[:100]]

    @api.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        task = task_manager.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task.model_dump()

    @api.get("/api/tasks/{task_id}/result")
    async def get_task_result(task_id: str):
        task = task_manager.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if task.status not in {"completed", "failed", "cancelled", "interrupted"}:
            return JSONResponse(status_code=202, content={"task_id": task_id, "status": task.status})
        return {
            "task_id": task_id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "report_path": str(storage.reports_dir / f"{task_id}.md"),
        }

    return api


app = create_api()

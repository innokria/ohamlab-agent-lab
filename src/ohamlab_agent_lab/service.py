from __future__ import annotations

import os
import uuid

from .models import AgentDefinition, CreateAgentRequest
from .runtime import AgentRuntime
from .storage import FileStorage
from .tasks import TaskManager


class AgentService:
    def __init__(self, storage: FileStorage, runtime: AgentRuntime, tasks: TaskManager):
        self.storage = storage
        self.runtime = runtime
        self.tasks = tasks

    def create_agent(self, request: CreateAgentRequest) -> AgentDefinition:
        agent = AgentDefinition(id=uuid.uuid4().hex, **request.model_dump())
        return self.storage.save_agent(agent)

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        return self.storage.get_agent(agent_id)

    def list_agents(self) -> list[AgentDefinition]:
        return self.storage.list_agents()

    def clone_agent(self, agent_id: str, new_name: str) -> AgentDefinition:
        source = self.storage.get_agent(agent_id)
        if not source:
            raise KeyError(f"Agent not found: {agent_id}")
        clone = source.model_copy(deep=True)
        clone.id = uuid.uuid4().hex
        clone.name = new_name
        clone.created_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        return self.storage.save_agent(clone)

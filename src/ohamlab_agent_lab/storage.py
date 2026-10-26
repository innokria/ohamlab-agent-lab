from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .models import AgentDefinition, TaskRecord, utc_now


class FileStorage:
    """File storage rooted on a mounted volume/Bucket path."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.agents_dir = self.root / "agents"
        self.tasks_dir = self.root / "tasks"
        self.reports_dir = self.root / "reports"
        self.artifacts_dir = self.root / "artifacts"
        for path in (self.agents_dir, self.tasks_dir, self.reports_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise ValueError("Invalid identifier")
        return value

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def save_agent(self, agent: AgentDefinition) -> AgentDefinition:
        self._write_json(self.agents_dir / f"{self._safe_id(agent.id)}.json", agent.model_dump())
        return agent

    def get_agent(self, agent_id: str) -> AgentDefinition | None:
        path = self.agents_dir / f"{self._safe_id(agent_id)}.json"
        if not path.exists():
            return None
        return AgentDefinition.model_validate_json(path.read_text(encoding="utf-8"))

    def list_agents(self) -> list[AgentDefinition]:
        agents = []
        for path in sorted(self.agents_dir.glob("*.json")):
            try:
                agents.append(AgentDefinition.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(agents, key=lambda item: item.created_at, reverse=True)

    def save_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        self._write_json(self.tasks_dir / f"{self._safe_id(task.task_id)}.json", task.model_dump())
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        path = self.tasks_dir / f"{self._safe_id(task_id)}.json"
        if not path.exists():
            return None
        return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_tasks(self) -> list[TaskRecord]:
        items = []
        for path in self.tasks_dir.glob("*.json"):
            try:
                items.append(TaskRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def save_report(self, task: TaskRecord, agent: AgentDefinition) -> str:
        path = self.reports_dir / f"{task.task_id}.md"
        content = (
            f"# {agent.name} — Task Report\n\n"
            f"- Task ID: `{task.task_id}`\n"
            f"- Agent ID: `{agent.id}`\n"
            f"- Status: `{task.status}`\n"
            f"- Created: {task.created_at}\n\n"
            "## Task\n\n"
            f"{task.task}\n\n"
            "## Result\n\n"
            f"{task.result or task.error or '(No result)'}\n"
        )
        path.write_text(content, encoding="utf-8")
        return str(path)

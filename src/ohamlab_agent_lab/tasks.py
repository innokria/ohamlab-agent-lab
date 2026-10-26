from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from typing import Any

from .models import AgentDefinition, TaskRecord, utc_now
from .runtime import AgentRuntime
from .storage import FileStorage


class TaskManager:
    """Single-process worker with optional Redis-backed queue and task records.

    For production, run workers as a separate always-on service. This class is a
    first deployment implementation and does not promise exactly-once execution.
    """

    TASK_PREFIX = "ohamlab-agent-lab:task:"
    QUEUE_KEY = "ohamlab-agent-lab:queue"

    def __init__(self, storage: FileStorage, runtime: AgentRuntime, redis_url: str = ""):
        self.storage = storage
        self.runtime = runtime
        self.redis = None
        self.redis_error = ""
        self.local_queue: queue.Queue[str] = queue.Queue()
        if redis_url:
            try:
                import redis
                client = redis.Redis.from_url(
                    redis_url, decode_responses=True,
                    socket_connect_timeout=3, socket_timeout=3,
                    health_check_interval=30,
                )
                client.ping()
                self.redis = client
            except Exception as exc:
                self.redis_error = f"{type(exc).__name__}: {exc}"

        self._recover_tasks()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="agent-task-worker")
        self.worker.start()

    @property
    def backend_status(self) -> str:
        if self.redis:
            return "Redis connected; queue and task records use Redis."
        if self.redis_error:
            return f"Redis unavailable; using local fallback ({self.redis_error[:160]})."
        return "Local queue mode; queued tasks are not durable across process restarts."

    def _get(self, task_id: str) -> TaskRecord | None:
        if self.redis:
            raw = self.redis.get(self.TASK_PREFIX + task_id)
            if raw:
                return TaskRecord.model_validate_json(raw)
        return self.storage.get_task(task_id)

    def _save(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        if self.redis:
            self.redis.set(self.TASK_PREFIX + task.task_id, task.model_dump_json())
        self.storage.save_task(task)
        return task

    def _all(self) -> list[TaskRecord]:
        if not self.redis:
            return self.storage.list_tasks()
        tasks = []
        for key in self.redis.scan_iter(match=self.TASK_PREFIX + "*", count=100):
            raw = self.redis.get(key)
            if raw:
                try:
                    tasks.append(TaskRecord.model_validate_json(raw))
                except Exception:
                    pass
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def _recover_tasks(self) -> None:
        for task in self._all():
            if task.status == "running":
                task.status = "interrupted"
                task.error = "Process restarted while this task was running; inspect before retrying."
                self._save(task)
            elif task.status == "queued":
                self._enqueue(task.task_id)

    def _enqueue(self, task_id: str) -> None:
        if self.redis:
            self.redis.lpush(self.QUEUE_KEY, task_id)
        else:
            self.local_queue.put(task_id)

    def submit(self, agent: AgentDefinition, prompt: str) -> TaskRecord:
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            agent_id=agent.id,
            task=prompt,
            status="queued",
        )
        self._save(task)
        self._enqueue(task.task_id)
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        return self._get(task_id)

    def list(self) -> list[TaskRecord]:
        return self._all()

    def _pop(self) -> str | None:
        if self.redis:
            item = self.redis.brpop(self.QUEUE_KEY, timeout=2)
            return item[1] if item else None
        try:
            return self.local_queue.get(timeout=1)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        while True:
            try:
                task_id = self._pop()
                if not task_id:
                    continue
                task = self._get(task_id)
                if not task or task.status != "queued":
                    continue
                agent = self.storage.get_agent(task.agent_id)
                if not agent:
                    task.status = "failed"
                    task.error = f"Agent not found: {task.agent_id}"
                    task.finished_at = utc_now()
                    self._save(task)
                    continue

                task.status = "running"
                task.started_at = utc_now()
                self._save(task)
                try:
                    task.result = asyncio.run(self.runtime.run(agent, task.task))
                    task.status = "completed"
                except Exception as exc:
                    task.status = "failed"
                    task.error = f"{type(exc).__name__}: {exc}"
                task.finished_at = utc_now()
                self.storage.save_report(task, agent)
                self._save(task)
            except Exception:
                # Prevent a transient queue/storage error from killing the worker.
                time.sleep(2)

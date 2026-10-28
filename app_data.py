from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import uvicorn
from datasets import DatasetDict, load_dataset
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from ohamlab_agent_lab.auth import install_auth_middleware
from ohamlab_agent_lab.api import (
    app as api_app,
    service,
    task_manager,
    runtime,
)
from ohamlab_agent_lab.mcp_server import mcp
from mcp.server.transport_security import TransportSecuritySettings
from ohamlab_agent_lab.models import CreateAgentRequest
from ohamlab_remote_mcp import install_runtime_integration, discover_remote_tools


# Enable remote MCP tool calling on the shared runtime (UI / API / MCP tasks).
install_runtime_integration(runtime)


# ============================================================
# Configuration
# ============================================================

DEFAULT_DATASET_ID = "rahul7star/agentic-ai-test-checklist-100"

DATASET_CACHE_DIR = os.getenv(
    "DATASET_CACHE_DIR",
    "/tmp/ohamlab-dataset-cache",
)

os.makedirs(DATASET_CACHE_DIR, exist_ok=True)


# ============================================================
# Agent initialisation
# ============================================================

def ensure_starter_agent():
    """Create a starter agent when persistent storage is empty."""

    try:
        existing = service.list_agents()

        if existing:
            return

        agent = service.create_agent(
            CreateAgentRequest(
                name="Researcher",
                description="Starter agent for research and report tasks.",
                execution_mode="workflow",
                workflow="mcp_search",
                system_prompt=(
                    "You are a helpful research agent. "
                    "Analyse the task, organise your findings, "
                    "and produce a clear report."
                ),
                published_api=True,
                published_mcp=True,
            )
        )

        print(f"Created starter agent: {agent.id}")

    except Exception as exc:
        print(f"Starter agent initialisation failed: {exc}")


ensure_starter_agent()


# ============================================================
# Agent data helpers
# ============================================================

def agent_choices():
    return [
        (f"{agent.name} ({agent.id[:8]})", agent.id)
        for agent in service.list_agents()
    ]


def agent_rows():
    return [
        [
            agent.id,
            agent.name,
            agent.description,
            agent.execution_mode,
            agent.workflow,
            agent.published_api,
            agent.published_mcp,
        ]
        for agent in service.list_agents()
    ]


def format_task_result(value):
    """Render persisted task results consistently across every UI view."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def task_rows():
    return [
        [
            task.task_id,
            task.agent_id,
            task.status,
            str(task.created_at),
            format_task_result(task.result),
            task.report_path,
            task.error,
        ]
        for task in task_manager.list()[:100]
    ]


def refresh_all():
    choices = agent_choices()
    selected = choices[0][1] if choices else None

    agents = agent_rows()
    tasks = task_rows()

    return (
        agents,
        gr.Dropdown(choices=choices, value=selected),
        gr.Dropdown(choices=choices, value=selected),
        tasks,
        f"Loaded {len(agents)} agents and {len(tasks)} tasks.",
    )


# ============================================================
# Agent actions
# ============================================================

def create_agent_ui(
    name,
    description,
    mode,
    workflow,
    prompt,
    base_url,
    model,
    api_key_env,
    publish_api,
    publish_mcp,
):
    if not name or not name.strip():
        raise gr.Error("Enter an agent name.")

    try:
        request = CreateAgentRequest(
            name=name.strip(),
            description=description or "",
            execution_mode=mode,
            workflow=workflow or "mcp_search",
            system_prompt=prompt or "You are a helpful assistant.",
            base_url=base_url or "",
            model=model or "",
            api_key_env=api_key_env or "LLM_API_KEY",
            published_api=publish_api,
            published_mcp=publish_mcp,
        )

        agent = service.create_agent(request)
        choices = agent_choices()

        return (
            agent_rows(),
            gr.Dropdown(choices=choices, value=agent.id),
            gr.Dropdown(choices=choices, value=agent.id),
            f"Created agent **{agent.name}** (`{agent.id}`).",
        )

    except Exception as exc:
        raise gr.Error(f"Could not create agent: {exc}") from exc


# ============================================================
# Task actions
# ============================================================

async def run_agent_ui(agent_id, task, asynchronous):
    if not agent_id:
        raise gr.Error("Create or select an agent first.")

    if not task or not task.strip():
        raise gr.Error("Enter a task.")

    agent = service.get_agent(agent_id)

    if not agent:
        raise gr.Error("Agent not found. Refresh the agent list.")

    try:
        if asynchronous:
            record = task_manager.submit(agent, task.strip())

            return (
                f"Task queued: `{record.task_id}`",
                record.task_id,
                task_rows(),
            )

        result = await runtime.run(agent, task.strip())

        return (
            str(result),
            "",
            task_rows(),
        )

    except Exception as exc:
        raise gr.Error(f"Task failed: {exc}") from exc


def inspect_task(task_id):
    record = task_manager.get((task_id or "").strip())

    if not record:
        raise gr.Error("Task not found.")

    return record.model_dump()


# ============================================================
# Dataset Explorer helpers
# ============================================================

def normalise_value(value: Any):
    """Convert dataset values into values safe for Gradio and pandas."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return value


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve every column while making nested values displayable."""

    result = df.copy()

    for column in result.columns:
        result[column] = result[column].map(normalise_value)

    return result


def find_column(df: pd.DataFrame, candidates: list[str]):
    """Find a column without depending on exact capitalisation."""

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        match = column_lookup.get(candidate.lower())

        if match is not None:
            return match

    return None


def get_category_column(df: pd.DataFrame):
    return find_column(
        df,
        [
            "eval_group",
            "class_name",
            "category",
            "pack",
            "test_category",
            "test_type",
        ],
    )


def get_priority_column(df: pd.DataFrame):
    return find_column(
        df,
        [
            "priority",
            "pri",
            "test_priority",
        ],
    )


def get_severity_column(df: pd.DataFrame):
    return find_column(
        df,
        [
            "severity",
            "risk_severity",
            "test_severity",
        ],
    )


def get_agent_type_column(df: pd.DataFrame):
    return find_column(
        df,
        [
            "applies_to",
            "agent_type",
            "agent_types",
            "agent_types_supported",
        ],
    )


def checklist_columns(df: pd.DataFrame) -> list[str]:
    """Choose compact table columns while retaining the full dataset."""

    preferred = [
        "id",
        "test_id",
        "title",
        "priority",
        "severity",
        "applies_to",
        "agent_type",
        "eval_group",
        "class_name",
    ]

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    selected = []

    for candidate in preferred:
        column = column_lookup.get(candidate.lower())

        if column is not None and column not in selected:
            selected.append(column)

    if not selected:
        selected = list(df.columns[:min(5, len(df.columns))])

    return selected


def compact_checklist_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact table view."""

    if df is None or df.empty:
        return pd.DataFrame()

    return df[checklist_columns(df)].copy()


def make_filter_dropdown(
    df: pd.DataFrame,
    column: str | None,
    label: str,
):
    """Build a filter dropdown from actual dataset values."""

    if df is None or df.empty or column is None:
        return gr.Dropdown(
            choices=["All"],
            value="All",
            label=label,
        )

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = sorted(
        value
        for value in values.unique()
        if value and value.lower() != "nan"
    )

    return gr.Dropdown(
        choices=["All"] + values,
        value="All",
        label=label,
    )


def load_checklist_dataset(dataset_id, config_name):
    """
    Load a Hugging Face dataset.

    Supports a DatasetDict containing train/test/validation splits
    and datasets with a single default split.
    """

    dataset_id = (dataset_id or "").strip()
    config_name = (config_name or "").strip()

    if not dataset_id:
        raise gr.Error("Enter a Hugging Face dataset ID.")

    try:
        kwargs = {
            "path": dataset_id,
            "cache_dir": DATASET_CACHE_DIR,
        }

        if config_name:
            kwargs["name"] = config_name

        dataset = load_dataset(**kwargs)

        # If multiple splits exist, prefer train.
        if isinstance(dataset, DatasetDict):
            if not dataset:
                raise gr.Error("The dataset contains no splits.")

            if "train" in dataset:
                split_name = "train"
            else:
                split_name = next(iter(dataset.keys()))

            selected_dataset = dataset[split_name]

        else:
            split_name = "default"
            selected_dataset = dataset

        df = selected_dataset.to_pandas()

        if df.empty:
            raise gr.Error(
                "The dataset loaded successfully but contains no rows."
            )

        # Convert nested objects into readable values.
        df = normalise_dataframe(df)

        columns = list(df.columns)

        summary = (
            "## Dataset loaded successfully\n\n"
            f"- **Dataset:** `{dataset_id}`\n"
            f"- **Configuration:** `{config_name or 'default'}`\n"
            f"- **Split:** `{split_name}`\n"
            f"- **Rows:** {len(df):,}\n"
            f"- **Columns:** {len(columns)}\n\n"
            "All dataset columns are available in the selected-row details."
        )

        column_dropdown = gr.Dropdown(
            choices=["All"] + columns,
            value="All",
            label="Search in column",
        )

        priority_dropdown = make_filter_dropdown(
            df,
            get_priority_column(df),
            "Priority",
        )

        severity_dropdown = make_filter_dropdown(
            df,
            get_severity_column(df),
            "Severity",
        )

        category_dropdown = make_filter_dropdown(
            df,
            get_category_column(df),
            "Category",
        )

        agent_dropdown = make_filter_dropdown(
            df,
            get_agent_type_column(df),
            "Agent type",
        )

        # Show the entire schema in the summary.
        schema = "\n".join(f"- `{column}`" for column in columns)

        summary += f"\n\n### Available columns\n\n{schema}"

        status = f"Loaded {len(df):,} rows across {len(columns)} columns."

        return (
            df,
            df.copy(),
            compact_checklist_table(df),
            summary,
            column_dropdown,
            priority_dropdown,
            severity_dropdown,
            category_dropdown,
            agent_dropdown,
            status,
            {
                "message": "Select a row to inspect all columns.",
                "dataset": dataset_id,
                "split": split_name,
            },
        )

    except gr.Error:
        raise

    except Exception as exc:
        raise gr.Error(
            f"Could not load dataset '{dataset_id}': {exc}"
        ) from exc


def filter_checklist(
    df: pd.DataFrame,
    search_column: str,
    search_text: str,
    priority: str,
    severity: str,
    category: str,
    agent_type: str,
):
    """Filter the full dataset without dropping any columns."""

    if df is None or df.empty:
        return pd.DataFrame(), "Load a dataset first."

    result = df.copy()

    def apply_exact_filter(column, value):
        nonlocal result

        if (
            column is not None
            and value
            and value != "All"
            and column in result.columns
        ):
            result = result[
                result[column].astype(str).str.strip() == str(value)
            ]

    apply_exact_filter(get_priority_column(result), priority)
    apply_exact_filter(get_severity_column(result), severity)
    apply_exact_filter(get_category_column(result), category)
    apply_exact_filter(get_agent_type_column(result), agent_type)

    query = (search_text or "").strip().lower()

    if query:
        if search_column and search_column != "All":
            if search_column in result.columns:
                mask = (
                    result[search_column]
                    .astype(str)
                    .str.lower()
                    .str.contains(query, regex=False, na=False)
                )
                result = result[mask]

        else:
            mask = result.astype(str).apply(
                lambda column: column.str.lower().str.contains(
                    query,
                    regex=False,
                    na=False,
                )
            ).any(axis=1)

            result = result[mask]

    return (
        result,
        f"Showing {len(result):,} of {len(df):,} rows.",
    )


def update_checklist(
    df,
    search_column,
    search_text,
    priority,
    severity,
    category,
    agent_type,
):
    filtered, status = filter_checklist(
        df,
        search_column,
        search_text,
        priority,
        severity,
        category,
        agent_type,
    )

    return (
        filtered,
        compact_checklist_table(filtered),
        status,
        {
            "message": (
                "Select a row in the checklist table to see every field."
            )
        },
    )


def inspect_checklist_row(
    df: pd.DataFrame,
    evt: gr.SelectData,
):
    """Display all fields for the selected test."""

    if df is None or df.empty:
        return {"message": "Load a dataset and select a row."}

    index = evt.index

    # Gradio can return either a row index or a (row, column) tuple.
    if isinstance(index, (tuple, list)):
        row_index = index[0]
    else:
        row_index = index

    try:
        row_index = int(row_index)
    except (TypeError, ValueError):
        return {"message": "Select a valid row."}

    if row_index < 0 or row_index >= len(df):
        return {"message": "Selected row is outside the current results."}

    row = df.iloc[row_index].to_dict()

    return {
        "row_number": row_index + 1,
        "total_columns": len(df.columns),
        "columns": row,
    }


def export_checklist_csv(df: pd.DataFrame):
    """Export all columns from the filtered dataset."""

    if df is None or df.empty:
        raise gr.Error("There is no data to export.")

    output = (
        Path(tempfile.gettempdir())
        / "agentic-ai-checklist-filtered.csv"
    )

    df.to_csv(output, index=False, encoding="utf-8-sig")

    return str(output)


def clear_dataset_filters():
    return (
        gr.Dropdown(value="All"),
        "",
        gr.Dropdown(value="All"),
        gr.Dropdown(value="All"),
        gr.Dropdown(value="All"),
        gr.Dropdown(value="All"),
    )



# ============================================================
# Autonomous Workspace + LLM Providers (single-file extension)
# ============================================================
import re
import threading
import subprocess
import shutil
import uuid
from urllib.parse import urlsplit

_LAB_ROOT = Path(os.getenv("OHAMLAB_WORKSPACE_ROOT", "./data/agent_workspace")).resolve()
_LAB_ROOT.mkdir(parents=True, exist_ok=True)
_PROVIDERS_PATH = _LAB_ROOT / "providers.json"
_LAB_LOCK = threading.RLock()
_LAB_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LAB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_LAB_MAX_FILES = 25
_LAB_MAX_BYTES = 180_000


def _lab_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _lab_providers():
    with _LAB_LOCK:
        if not _PROVIDERS_PATH.exists():
            _lab_atomic(_PROVIDERS_PATH, {"default": "xing", "items": [{
                "id": "xing", "name": "Xing",
                "base_url": "https://rahul7star-xing4-0-29b-a4b.hf.space/v1",
                "model": "Xing", "api_key_env": "HF_TOKEN"}]})
        return json.loads(_PROVIDERS_PATH.read_text(encoding="utf-8"))


def _lab_provider_choices():
    data = _lab_providers()
    return [(p["name"] + (" (default)" if p["id"] == data["default"] else ""), p["id"])
            for p in data["items"]]


def _lab_provider_get(pid):
    data = _lab_providers()
    for p in data["items"]:
        if p["id"] == (pid or data["default"]):
            return p
    raise ValueError("Provider not found")


def _lab_validate_provider(name, url, model_name, env_name):
    name, url, model_name, env_name = [str(v or "").strip() for v in (name, url, model_name, env_name)]
    parsed = urlsplit(url)
    if not name or len(name) > 100 or not model_name or len(model_name) > 200:
        raise ValueError("Name and model are required")
    if (parsed.scheme not in ("https", "http") or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Enter a valid HTTP(S) API URL without embedded credentials")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote providers must use HTTPS")
    if not _LAB_ENV_PATTERN.fullmatch(env_name):
        raise ValueError("Enter an environment variable name, not an API key")
    return {"name": name, "base_url": url.rstrip("/"), "model": model_name, "api_key_env": env_name}


def _lab_provider_save(pid, name, url, model_name, env_name):
    values = _lab_validate_provider(name, url, model_name, env_name)
    with _LAB_LOCK:
        data = _lab_providers()
        if pid and pid != "__new__":
            target = next((p for p in data["items"] if p["id"] == pid), None)
            if target is None:
                raise ValueError("Unknown provider")
            target.update(values)
        else:
            target = {"id": uuid.uuid4().hex[:12], **values}
            data["items"].append(target)
        _lab_atomic(_PROVIDERS_PATH, data)
    return (gr.update(choices=_lab_provider_choices(), value=target["id"]),
            gr.update(choices=_lab_provider_choices(), value=target["id"]),
            f"Saved provider **{target['name']}**")


def _lab_provider_default(pid):
    _lab_provider_get(pid)
    with _LAB_LOCK:
        data = _lab_providers()
        data["default"] = pid
        _lab_atomic(_PROVIDERS_PATH, data)
    return gr.update(choices=_lab_provider_choices(), value=pid), "Default provider updated"


def _lab_provider_fields(pid):
    if pid == "__new__":
        return "", "", "", ""
    p = _lab_provider_get(pid)
    return p["name"], p["base_url"], p["model"], p["api_key_env"]


def _lab_client(provider):
    from openai import OpenAI
    key = os.getenv(provider["api_key_env"])
    if not key:
        raise ValueError(f"Set {provider['api_key_env']} in your Space secrets / environment")
    return OpenAI(base_url=provider["base_url"], api_key=key, timeout=60, max_retries=0)


def _lab_provider_test(pid):
    p = _lab_provider_get(pid)
    client = _lab_client(p)
    try:
        res = client.chat.completions.create(model=p["model"], messages=[
            {"role": "user", "content": "Reply with OK"}], max_tokens=32)
        return f"Connected to **{p['name']}**: {str(res.choices[0].message.content)[:160]}"
    except Exception as exc:
        return f"Connection failed: {type(exc).__name__}: {str(exc)[:300]}"
    finally:
        client.close()


def _lab_provider_apply(pid):
    p = _lab_provider_get(pid)
    return p["base_url"], p["model"], p["api_key_env"]


def _lab_task_path(tid):
    if not re.fullmatch(r"[a-f0-9]{32}", str(tid or "")):
        raise ValueError("Invalid task ID")
    return _LAB_ROOT / "tasks" / tid


def _lab_record(tid):
    return json.loads((_lab_task_path(tid) / "task.json").read_text(encoding="utf-8"))


def _lab_update(tid, **updates):
    with _LAB_LOCK:
        record = _lab_record(tid)
        record.update(updates)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        _lab_atomic(_lab_task_path(tid) / "task.json", record)
        return record


def _lab_safe_file(workspace, name):
    candidate = Path(str(name))
    if (candidate.is_absolute() or ".." in candidate.parts or not candidate.parts or
            any(part.startswith(".") for part in candidate.parts) or
            any(part in ("__pycache__", "node_modules") for part in candidate.parts)):
        raise ValueError(f"Unsafe file path: {name}")
    dest = (workspace / candidate).resolve()
    if not dest.is_relative_to(workspace.resolve()):
        raise ValueError("Path escapes workspace")
    return dest


def _lab_extract_json(text_value):
    text_value = str(text_value or "").strip()
    if text_value.startswith("```"):
        text_value = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_value, flags=re.I)
    return json.loads(text_value)


def _lab_generate(client, model_name, goal, previous, iteration):
    instructions = ("You are a Python coding agent. Return ONLY a JSON object with keys "
                    "files and summary. files is an array of objects with path and content. "
                    "Write a complete small Python project with unittest tests (test_*.py). "
                    "No external dependencies, no secrets, no network access. "
                    "Each response must include the complete desired content of each changed file. "
                    "Do not output markdown. Limit to 20 files and 120000 characters total.")
    user_prompt = f"GOAL:\n{goal}\nITERATION: {iteration}\nPREVIOUS TEST OUTPUT:\n{previous[-5000:]}"
    response = client.chat.completions.create(model=model_name,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": user_prompt}], max_tokens=5000)
    return _lab_extract_json(response.choices[0].message.content)


def _lab_run_tests(workspace):
    if not list(workspace.rglob("test_*.py")):
        return False, "No test_*.py files were generated; refusing to mark task as verified."
    # Deliberately fail closed: never execute LLM-generated code in the Gradio host.
    if not shutil.which("docker"):
        return False, "Docker sandbox unavailable. Configure an isolated worker; host execution is disabled."
    cmd = ["docker", "run", "--rm", "--network", "none", "--cpus", "1", "--memory", "512m",
           "--pids-limit", "96", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
           "--user", "65534:65534", "--mount", f"type=bind,src={workspace},dst=/work,readonly",
           "--workdir", "/work", "--env", "PYTHONDONTWRITEBYTECODE=1",
           os.getenv("OHAMLAB_SANDBOX_IMAGE", "python:3.12-slim"),
           "python", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py", "-v"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=False)
        return result.returncode == 0, (result.stdout + "\n" + result.stderr)[-10000:]
    except subprocess.TimeoutExpired:
        return False, "Sandbox tests timed out after 90 seconds"
    except Exception as exc:
        return False, f"Sandbox error: {exc}"


def _lab_worker(tid):
    workspace = _lab_task_path(tid) / "workspace"
    record = _lab_record(tid)
    client = None
    try:
        p = _lab_provider_get(record["provider_id"])
        client = _lab_client(p)
        _lab_update(tid, status="running")
        previous = ""
        for attempt in range(1, 4):
            current = _lab_record(tid)
            if current["status"] == "cancelled":
                return
            result = _lab_generate(client, p["model"], record["goal"], previous, attempt)
            files = result.get("files", [])
            if not isinstance(files, list) or not files or len(files) > _LAB_MAX_FILES:
                raise ValueError("Model did not return a valid bounded file list")
            total = sum(len(str(f.get("content", "")).encode("utf-8")) for f in files)
            if total > _LAB_MAX_BYTES:
                raise ValueError("Generated files exceed size limit")
            for f in files:
                if not isinstance(f, dict) or not isinstance(f.get("content"), str):
                    raise ValueError("Invalid generated file")
                path = _lab_safe_file(workspace, f.get("path", ""))
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    raise ValueError("Symlink file is not permitted")
                path.write_text(f["content"], encoding="utf-8")
            # In environments without Docker, generate files but do not execute
            # untrusted model-generated Python on the Gradio host.
            if not shutil.which("docker"):
                _lab_update(tid, attempts=attempt,
                            test_output="Tests not run: no isolated execution worker is configured.",
                            files=sorted(str(f.relative_to(workspace)) for f in workspace.rglob("*") if f.is_file()),
                            summary=str(result.get("summary", ""))[:1000],
                            status="generated_unverified")
                return
            passed, output = _lab_run_tests(workspace)
            previous = output
            _lab_update(tid, attempts=attempt, test_output=output,
                        files=sorted(str(f.relative_to(workspace)) for f in workspace.rglob("*") if f.is_file()),
                        summary=str(result.get("summary", ""))[:1000])
            if passed:
                if _lab_record(tid)["status"] == "cancelled":
                    return
                import hashlib
                hashes = {rel: hashlib.sha256(_lab_safe_file(workspace, rel).read_bytes()).hexdigest()
                          for rel in _lab_record(tid).get("files", [])}
                _lab_update(tid, status="awaiting_approval", verified_hashes=hashes)
                return
            if "sandbox unavailable" in output.lower():
                _lab_update(tid, status="blocked", error=output)
                return
        _lab_update(tid, status="tests_failed", error="Tests did not pass after three attempts")
    except Exception as exc:
        _lab_update(tid, status="failed", error=f"{type(exc).__name__}: {exc}"[:1200])
    finally:
        if client is not None:
            client.close()


def _lab_start(goal, provider_id):
    if not str(goal or "").strip():
        raise gr.Error("Enter a coding task")
    try:
        _lab_provider_get(provider_id)
        tid = uuid.uuid4().hex
        task_dir = _lab_task_path(tid)
        (task_dir / "workspace").mkdir(parents=True)
        _lab_atomic(task_dir / "task.json", {"id": tid, "goal": goal.strip()[:8000],
                    "provider_id": provider_id, "status": "queued", "attempts": 0,
                    "created_at": datetime.now(timezone.utc).isoformat()})
        threading.Thread(target=_lab_worker, args=(tid,), daemon=True).start()
        return tid, f"Started task `{tid}`. Refresh status to follow progress."
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def _lab_status(tid):
    try:
        record = _lab_record(tid)
        return json.dumps(record, indent=2, ensure_ascii=False)
    except Exception as exc:
        return f"Could not load task: {exc}"


def _lab_publish(tid, repo_name, visibility):
    # Only publish completed, verified runs, after an explicit UI click.
    record = _lab_record(tid)
    if record["status"] != "awaiting_approval":
        raise gr.Error("Task must pass tests and await approval before publishing")
    if not _LAB_REPO_PATTERN.fullmatch(str(repo_name or "")) or repo_name in (".", ".."):
        raise gr.Error("Enter a valid new GitHub repository name")
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise gr.Error("Set GITHUB_TOKEN as a server secret")
    import requests
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    api = "https://api.github.com"
    try:
        # Create a NEW repository only; never overwrite an existing repository.
        response = requests.post(api + "/user/repos", headers=headers, json={
            "name": repo_name, "private": visibility == "Private", "auto_init": True}, timeout=30)
        response.raise_for_status()
        repo = response.json()
        owner = repo["owner"]["login"]
        import base64
        workspace = _lab_task_path(tid) / "workspace"
        for rel in record.get("files", []):
            path = _lab_safe_file(workspace, rel)
            if not path.is_file() or path.is_symlink():
                raise ValueError("File changed since test execution; refusing to publish")
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("verified_hashes", {}).get(rel):
                raise ValueError("File changed after testing; refusing to publish")
            payload = {"message": f"Add {rel} from OhamLab task {tid[:8]}",
                       "content": base64.b64encode(path.read_bytes()).decode("ascii")}
            from urllib.parse import quote
            result = requests.put(f"{api}/repos/{owner}/{repo_name}/contents/{quote(rel, safe='/')}",
                                  headers=headers, json=payload, timeout=30)
            result.raise_for_status()
        _lab_update(tid, status="published", github_url=repo["html_url"])
        return f"Published to {repo['html_url']} ({len(record.get('files', []))} file commits)."
    except Exception as exc:
        # A repository may have been created even when uploading a later file fails.
        _lab_update(tid, status="publish_failed", error=f"GitHub publish failed: {exc}"[:1000])
        raise gr.Error(f"Publishing may be partial. Check GitHub before retrying: {exc}") from exc


def _lab_cancel(tid):
    record = _lab_record(tid)
    if record["status"] in ("queued", "running"):
        _lab_update(tid, status="cancelled")
        return "Cancellation requested. An active model call or sandbox test may finish first."
    return f"Task already {record['status']}"


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(
    title="OhamLab Agent Lab",
) as demo:

    gr.Markdown(
        """
        # OhamLab Agent Lab

        **Create agents · Execute tasks · Explore evaluation datasets**

        Build agents, run workflows, inspect task results, and browse
        agentic AI test checklists from Hugging Face.
        """
    )

    backend_status = gr.Markdown(
        f"**Task backend:** {task_manager.backend_status}"
    )

    with gr.Row():
        refresh_button = gr.Button(
            "Refresh all data",
            variant="primary",
        )

        data_status = gr.Markdown("Loading data...")

    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------

    with gr.Tab("Agents"):

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("### Create agent")

                name = gr.Textbox(
                    label="Name",
                    value="Researcher",
                )

                description = gr.Textbox(
                    label="Description",
                    value="Research and report generation",
                )

                mode = gr.Radio(
                    ["workflow", "llm"],
                    value="workflow",
                    label="Execution mode",
                )

                workflow = gr.Dropdown(
                    ["mcp_search", "mcp_research", "report", "echo"],
                    value="mcp_search",
                    label="Workflow",
                )

                prompt = gr.Textbox(
                    label="System prompt",
                    lines=4,
                    value="You are a helpful research assistant.",
                )

                provider_for_agent = gr.Dropdown(
                    label="Saved LLM provider (loads fields below)",
                    choices=_lab_provider_choices(),
                    value=_lab_providers()["default"],
                )

                base_url = gr.Textbox(
                    label="LLM base URL",
                    value=_lab_provider_get(None)["base_url"],
                )

                model = gr.Textbox(
                    label="LLM model",
                    value=_lab_provider_get(None)["model"],
                )

                api_key_env = gr.Textbox(
                    label="API key environment variable",
                    value=_lab_provider_get(None)["api_key_env"],
                )

                publish_api = gr.Checkbox(
                    value=True,
                    label="Publish through REST API",
                )

                publish_mcp = gr.Checkbox(
                    value=True,
                    label="Publish through MCP",
                )

                create_button = gr.Button(
                    "Create agent",
                    variant="primary",
                )

                create_status = gr.Markdown()

            with gr.Column(scale=2):

                gr.Markdown("### Saved agents")

                agents = gr.Dataframe(
                    headers=[
                        "ID",
                        "Name",
                        "Description",
                        "Mode",
                        "Workflow",
                        "REST",
                        "MCP",
                    ],
                    value=agent_rows(),
                    interactive=False,
                    wrap=True,
                    label="Registered agents",
                )

                selected_agent = gr.Dropdown(
                    choices=agent_choices(),
                    label="Selected agent",
                )

    # --------------------------------------------------------
    # Run task
    # --------------------------------------------------------

    with gr.Tab("LLM Providers"):
        gr.Markdown("### Saved OpenAI-compatible endpoints\nSecrets stay in server environment variables, not in this UI.")
        with gr.Row():
            with gr.Column():
                provider_pick = gr.Dropdown(label="Saved provider", choices=_lab_provider_choices(),
                                            value=_lab_providers()["default"])
                new_provider = gr.Button("New provider")
                provider_name = gr.Textbox(label="Provider name", value="Xing")
                provider_url = gr.Textbox(label="LLM base URL", value=_lab_provider_get(None)["base_url"])
                provider_model = gr.Textbox(label="LLM model", value=_lab_provider_get(None)["model"])
                provider_secret = gr.Textbox(label="API key environment variable", value=_lab_provider_get(None)["api_key_env"])
                with gr.Row():
                    provider_save = gr.Button("Save provider", variant="primary")
                    provider_test = gr.Button("Test connection")
                    provider_default = gr.Button("Set default")
                provider_message = gr.Markdown()
        provider_pick.change(_lab_provider_fields, [provider_pick],
                             [provider_name, provider_url, provider_model, provider_secret])
        new_provider.click(lambda: ("__new__", "", "", "", ""), outputs=[
            provider_pick, provider_name, provider_url, provider_model, provider_secret])
        provider_save.click(_lab_provider_save,
                            [provider_pick, provider_name, provider_url, provider_model, provider_secret],
                            [provider_pick, provider_for_agent, provider_message])
        provider_test.click(_lab_provider_test, [provider_pick], [provider_message])
        provider_default.click(_lab_provider_default, [provider_pick], [provider_pick, provider_message])

    with gr.Tab("Autonomous Workspace"):
        gr.Markdown("### Python coding agent\nGenerate Python project files with your selected LLM. If an isolated Docker runner is available, OhamLab also runs tests and retries fixes. Otherwise the task finishes as **generated_unverified**; no untrusted code runs on the app server, and GitHub publishing stays disabled until tests pass.")
        coding_provider = gr.Dropdown(label="Coding LLM provider", choices=_lab_provider_choices(),
                                     value=_lab_providers()["default"])
        coding_goal = gr.Textbox(label="Task", lines=5, placeholder="Build a Python expense tracker with unittest tests...")
        coding_start = gr.Button("Start coding task", variant="primary")
        coding_id = gr.Textbox(label="Task ID")
        coding_message = gr.Markdown()
        coding_refresh = gr.Button("Refresh task status")
        coding_result = gr.Code(label="Task status / test output", language="json")
        coding_cancel = gr.Button("Request cancellation")
        gr.Markdown("### Publish approved project to a NEW GitHub repository")
        github_repo = gr.Textbox(label="New repository name")
        github_visibility = gr.Radio(["Private", "Public"], value="Private", label="Visibility")
        github_approve = gr.Button("Approve and publish to GitHub", variant="secondary")
        github_result = gr.Markdown()
        coding_start.click(_lab_start, [coding_goal, coding_provider], [coding_id, coding_message])
        coding_refresh.click(_lab_status, [coding_id], [coding_result])
        coding_cancel.click(_lab_cancel, [coding_id], [coding_message])
        github_approve.click(_lab_publish, [coding_id, github_repo, github_visibility], [github_result])

    with gr.Tab("Run task"):

        gr.Markdown("## Execute an agent task")

        selected_agent_run = gr.Dropdown(
            choices=agent_choices(),
            label="Agent",
        )

        task_input = gr.Textbox(
            label="Task description",
            lines=5,
            placeholder="Describe what the agent should do...",
        )

        asynchronous = gr.Checkbox(
            value=True,
            label="Run as background task",
        )

        run_button = gr.Button(
            "Run task",
            variant="primary",
        )

        run_result = gr.Markdown()
        task_id_output = gr.Textbox(label="Task ID")
        run_task_result = gr.Textbox(label="Saved task result", lines=10, interactive=False)
        run_task_error = gr.Textbox(label="Saved task error", lines=2, interactive=False)
        run_task_refresh = gr.Button("Refresh task result")

        def refresh_run_task_result(task_id):
            record = task_manager.get((task_id or "").strip()) if task_id else None
            if record is None:
                return "", ""
            return format_task_result(record.result), str(record.error or "")

        run_task_refresh.click(
            refresh_run_task_result,
            inputs=[task_id_output],
            outputs=[run_task_result, run_task_error],
        )

        gr.Markdown("### Task history")

        current_tasks = gr.Dataframe(
            headers=[
                "Task ID",
                "Agent ID",
                "Status",
                "Created",
                "Result",
                "Report",
                "Error",
            ],
            value=task_rows(),
            interactive=False,
            wrap=True,
            label="Recent tasks",
        )

        refresh_tasks_button = gr.Button("Refresh task history")

    # --------------------------------------------------------
    # Task details
    # --------------------------------------------------------

    with gr.Tab("Task details"):

        gr.Markdown("## Inspect a task")

        task_id_input = gr.Textbox(
            label="Task ID",
            placeholder="Paste a task ID",
        )

        inspect_button = gr.Button(
            "Inspect task",
            variant="primary",
        )

        details = gr.JSON(label="Task details")
        task_details_result = gr.Textbox(
            label="Task result", lines=12, interactive=False,
        )
        task_details_error = gr.Textbox(
            label="Task error", lines=2, interactive=False,
        )

        def inspect_task_with_result(task_id):
            data = inspect_task(task_id)
            return (
                data,
                format_task_result(data.get("result")),
                str(data.get("error") or ""),
            )

        inspect_button.click(
            inspect_task_with_result,
            inputs=[task_id_input],
            outputs=[details, task_details_result, task_details_error],
        )

    # --------------------------------------------------------
    # Dataset Explorer
    # --------------------------------------------------------

    with gr.Tab("Dataset Explorer"):

        gr.Markdown(
            """
            # Agentic AI Test Checklist

            Load a Hugging Face dataset, browse test cases, search across
            columns, apply filters, inspect full records, and export results.

            The compact table is designed for browsing. Selecting a row
            displays **all fields available in that dataset record**.
            """
        )

        dataset_state = gr.State(pd.DataFrame())
        filtered_state = gr.State(pd.DataFrame())

        with gr.Row():

            dataset_id_input = gr.Textbox(
                label="Hugging Face dataset ID",
                value=DEFAULT_DATASET_ID,
                placeholder="username/dataset-name",
                scale=4,
            )

            config_input = gr.Textbox(
                label="Configuration (optional)",
                placeholder="For example: all",
                scale=2,
            )

            load_dataset_button = gr.Button(
                "Load dataset",
                variant="primary",
                scale=1,
            )

        dataset_status = gr.Markdown(
            "Enter a dataset ID and click **Load dataset**."
        )

        with gr.Accordion("Dataset schema", open=False):
            gr.Markdown(
                "The schema summary will list every column after loading."
            )

        # Search
        with gr.Row():

            search_column = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Search in column",
                scale=1,
            )

            search_text = gr.Textbox(
                label="Search",
                placeholder=(
                    "Search ID, title, scenario, steps, tags, "
                    "evidence, standards..."
                ),
                scale=3,
            )

        # Filters
        with gr.Row():

            priority_filter = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Priority",
            )

            severity_filter = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Severity",
            )

            category_filter = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Category / pack",
            )

            agent_type_filter = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Agent type",
            )

            clear_filters_button = gr.Button("Clear filters")

        results_status = gr.Markdown("No dataset loaded.")

        with gr.Row():

            with gr.Column(scale=3):

                gr.Markdown("### Checklist records")

                checklist_table = gr.Dataframe(
                    value=pd.DataFrame(),
                    interactive=False,
                    wrap=True,
                    max_height=600,
                    label="Select a row to inspect its full record",
                )

            with gr.Column(scale=2):

                gr.Markdown("### Complete record")

                selected_test_details = gr.JSON(
                    label="All columns",
                    value={
                        "message": (
                            "Load the dataset and select a checklist row."
                        )
                    },
                )

        with gr.Row():

            export_button = gr.Button(
                "Export filtered CSV",
                variant="secondary",
            )

            export_file = gr.File(
                label="Download exported dataset",
            )

        # --------------------------------------------
        # Load dataset
        # --------------------------------------------

        def load_dataset_ui(dataset_id, config_name):

            (
                full_df,
                filtered_df,
                preview,
                summary,
                column_dropdown,
                priority_dropdown,
                severity_dropdown,
                category_dropdown,
                agent_dropdown,
                status,
                details_placeholder,
            ) = load_checklist_dataset(
                dataset_id,
                config_name,
            )

            return (
                full_df,
                filtered_df,
                preview,
                summary,
                column_dropdown,
                priority_dropdown,
                severity_dropdown,
                category_dropdown,
                agent_dropdown,
                status,
                details_placeholder,
            )

        load_dataset_button.click(
            load_dataset_ui,
            inputs=[
                dataset_id_input,
                config_input,
            ],
            outputs=[
                dataset_state,
                filtered_state,
                checklist_table,
                dataset_status,
                search_column,
                priority_filter,
                severity_filter,
                category_filter,
                agent_type_filter,
                results_status,
                selected_test_details,
            ],
        )

        # --------------------------------------------
        # Search and filter
        # --------------------------------------------

        filter_inputs = [
            dataset_state,
            search_column,
            search_text,
            priority_filter,
            severity_filter,
            category_filter,
            agent_type_filter,
        ]

        filter_outputs = [
            filtered_state,
            checklist_table,
            results_status,
            selected_test_details,
        ]

        for component in (
            search_column,
            search_text,
            priority_filter,
            severity_filter,
            category_filter,
            agent_type_filter,
        ):
            component.change(
                update_checklist,
                inputs=filter_inputs,
                outputs=filter_outputs,
            )

        # --------------------------------------------
        # Inspect a selected record
        # --------------------------------------------

        checklist_table.select(
            inspect_checklist_row,
            inputs=[filtered_state],
            outputs=[selected_test_details],
        )

        # --------------------------------------------
        # Clear filters
        # --------------------------------------------

        clear_filters_button.click(
            clear_dataset_filters,
            outputs=[
                search_column,
                search_text,
                priority_filter,
                severity_filter,
                category_filter,
                agent_type_filter,
            ],
        )

        # --------------------------------------------
        # Export CSV
        # --------------------------------------------

        export_button.click(
            export_checklist_csv,
            inputs=[filtered_state],
            outputs=[export_file],
        )

    # ========================================================
    # Command Center: searchable task listing + inspector
    # ========================================================
    with gr.Tab("Command Center"):
        gr.Markdown("## Task Command Center\nLive tasks from the existing task manager / Hugging Face Bucket backend.")
        with gr.Row():
            cc_search = gr.Textbox(label="Search task ID, instructions, agent, or status", scale=4)
            cc_status = gr.Dropdown(["All", "queued", "pending", "running", "completed", "failed", "cancelled"], value="All", label="Status", scale=1)
            cc_refresh = gr.Button("Refresh", scale=1)
        cc_summary = gr.Markdown()
        cc_table = gr.Dataframe(headers=["Task ID", "Task", "Agent", "Status", "Created", "Updated", "Result", "Report", "Error"], interactive=False, label="Stored tasks", wrap=True, max_height=470)
        with gr.Row():
            cc_id = gr.Textbox(label="Inspect task ID", placeholder="Paste full task ID")
            cc_inspect = gr.Button("Open task", variant="primary")
        cc_details = gr.JSON(label="Complete task record")
        gr.Markdown("### Task result")
        cc_result_status = gr.Markdown("Select a task or enter its ID to view its result.")
        cc_result = gr.Textbox(
            label="Agent output / result",
            lines=14,
            max_lines=25,
            interactive=False,
        )
        cc_result_error = gr.Textbox(
            label="Task error (if any)",
            lines=2,
            interactive=False,
        )

        def command_center_inspect(task_id):
            task_id = (task_id or "").strip()
            if not task_id:
                return (
                    {"message": "Select a task or enter its ID."},
                    "Select a task or enter its ID to view its result.",
                    "",
                    "",
                )
            record = task_manager.get(task_id)
            if record is None:
                return (
                    {"error": "Task not found", "task_id": task_id},
                    f"**Task not found:** `{task_id}`",
                    "",
                    "",
                )
            data = record.model_dump(mode="json")
            result_text = format_task_result(data.get("result"))
            state = str(getattr(record, "status", "unknown"))
            report = getattr(record, "report_path", None)
            info = f"**Task:** `{task_id}` · **Status:** {state}"
            if report:
                info += f" · **Report path:** `{report}`"
            if not result_text and state.lower() not in ("completed", "failed", "cancelled"):
                result_text = "Task has not produced a result yet."
            return data, info, result_text, str(getattr(record, "error", "") or "")

        def command_center_select(rows, evt: gr.SelectData):
            index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            try:
                task_id = str(rows.iloc[int(index), 0] if hasattr(rows, "iloc") else rows[int(index)][0])
            except (IndexError, TypeError, ValueError):
                return "", {"message": "Select a valid task row."}, "", "", ""
            return (task_id, *command_center_inspect(task_id))

        gr.Markdown("### Submit a task")
        with gr.Row():
            cc_agent = gr.Dropdown(choices=agent_choices(), label="Assigned agent", scale=2)
            cc_async = gr.Checkbox(value=True, label="Background execution", scale=1)
        cc_instruction = gr.Textbox(label="Task instructions", lines=4, placeholder="Find public information about rahul7star... or write Python code...")
        cc_submit = gr.Button("Create and run task", variant="primary")
        cc_submit_result = gr.Markdown()
        cc_created_id = gr.Textbox(label="Created task ID")

        def command_center_rows(query="", status="All"):
            items = task_manager.list()
            q = (query or "").strip().casefold()
            output = []
            for t in items:
                instruction = str(getattr(t, "task", ""))
                agent_id = str(getattr(t, "agent_id", ""))
                state = str(getattr(t, "status", ""))
                if status != "All" and state.casefold() != status.casefold():
                    continue
                if q and q not in " ".join([str(t.task_id), instruction, agent_id, state, format_task_result(getattr(t, "result", None))]).casefold():
                    continue
                output.append([
                    str(t.task_id), instruction, agent_id, state,
                    str(getattr(t, "created_at", "")),
                    str(getattr(t, "updated_at", "")),
                    format_task_result(getattr(t, "result", None)),
                    str(getattr(t, "report_path", "")),
                    str(getattr(t, "error", "") or ""),
                ])
            return output, f"**Showing {len(output)} of {len(items)} stored tasks**"

        cc_search.change(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_status.change(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_refresh.click(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_inspect.click(
            command_center_inspect,
            [cc_id],
            [cc_details, cc_result_status, cc_result, cc_result_error],
        )
        cc_table.select(
            command_center_select,
            [cc_table],
            [cc_id, cc_details, cc_result_status, cc_result, cc_result_error],
        )
        cc_submit.click(run_agent_ui, [cc_agent, cc_instruction, cc_async],
                        [cc_submit_result, cc_created_id, current_tasks],
                        ).then(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_timer = gr.Timer(8)
        cc_timer.tick(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_timer.tick(
            command_center_inspect,
            [cc_id],
            [cc_details, cc_result_status, cc_result, cc_result_error],
        )
        demo.load(command_center_rows, outputs=[cc_table, cc_summary])

    # ========================================================
    # Agent Quality Lab: dataset-backed checklist inspection
    # This is a catalogue; it does NOT fabricate evaluation scores.
    # ========================================================
    with gr.Tab("Agent Quality Lab"):
        gr.Markdown("## Agent Quality Lab\nExplore applicable checklist cases. Automated evaluation requires dedicated case runners and evidence collection.")
        quality_df = gr.State(pd.DataFrame())
        with gr.Row():
            quality_dataset_id = gr.Textbox(value=DEFAULT_DATASET_ID, label="Checklist dataset", scale=4)
            quality_config = gr.Textbox(label="Dataset config (optional)", scale=1)
            quality_load = gr.Button("Load checklist", variant="primary")
        quality_info = gr.Markdown("Load the checklist to browse its cases.")
        quality_search = gr.Textbox(label="Search checklist", placeholder="Search test title, priority, severity, category...")
        quality_table = gr.Dataframe(label="Evaluation cases", interactive=False, max_height=500, wrap=True)
        quality_details = gr.JSON(label="Selected evaluation case")
        def quality_load_fn(dataset_id, config):
            df, _, compact, summary, *_ = load_checklist_dataset(dataset_id, config)
            return df, compact, summary
        def quality_filter_fn(df, q):
            if df is None or df.empty:
                return pd.DataFrame()
            if not (q or "").strip():
                return compact_checklist_table(df)
            mask = df.astype(str).apply(lambda c: c.str.contains(q, case=False, regex=False, na=False)).any(axis=1)
            return compact_checklist_table(df.loc[mask])
        quality_load.click(quality_load_fn, [quality_dataset_id, quality_config], [quality_df, quality_table, quality_info])
        quality_search.change(quality_filter_fn, [quality_df, quality_search], [quality_table])
        # Inspect the displayed, filtered rows rather than an unfiltered index.
        quality_table.select(inspect_checklist_row, [quality_table], [quality_details])

    # ========================================================
    # Shared event handlers
    # ========================================================

    provider_for_agent.change(_lab_provider_apply, [provider_for_agent],
                              [base_url, model, api_key_env])

    create_button.click(
        create_agent_ui,
        inputs=[
            name,
            description,
            mode,
            workflow,
            prompt,
            base_url,
            model,
            api_key_env,
            publish_api,
            publish_mcp,
        ],
        outputs=[
            agents,
            selected_agent,
            selected_agent_run,
            create_status,
        ],
    )

    refresh_button.click(
        refresh_all,
        outputs=[
            agents,
            selected_agent,
            selected_agent_run,
            current_tasks,
            data_status,
        ],
    )

    refresh_tasks_button.click(
        task_rows,
        outputs=[current_tasks],
    )

    run_button.click(
        run_agent_ui,
        inputs=[
            selected_agent_run,
            task_input,
            asynchronous,
        ],
        outputs=[
            run_result,
            task_id_output,
            current_tasks,
        ],
    )

    # Refresh task history every five seconds.
    timer = gr.Timer(5)

    timer.tick(
        task_rows,
        outputs=[current_tasks],
    )
    timer.tick(
        refresh_run_task_result,
        inputs=[task_id_output],
        outputs=[run_task_result, run_task_error],
    )


# ============================================================
# FastAPI + REST API + MCP + Gradio
# ============================================================

# Permit the public Hugging Face host while keeping MCP host validation enabled.
# Configure this before constructing the MCP ASGI application.
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "rahul7star-ohamlab-agentic.hf.space",
        "rahul7star-ohamlab-agentic.hf.space:*",
        "localhost:*",
        "127.0.0.1:*",
    ],
    allowed_origins=[
        "https://rahul7star-ohamlab-agentic.hf.space",
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


application = FastAPI(
    title="OhamLab Agent Lab- search is mcp research is llm",
    lifespan=lifespan,
)

install_auth_middleware(application)

application.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-OhamLab-Key",
    ],
)


# Root redirects to the Gradio UI.
@application.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(
        url="/ui/",
        status_code=307,
    )


# REST API is mounted separately from the UI.
application.mount("/api", api_app)

# Gradio UI.
application = gr.mount_gradio_app(
    application,
    demo,
    path="/ui",
    auth=(os.getenv("UIUSER", "admin"), os.environ["UIPASS"]),
    auth_message="Sign in to OhamLab Agent Lab",
)

# Mount MCP's ASGI app last at root. The SDK's streamable_http_app()
# already owns the internal /mcp route, so mounting it at /mcp
# would otherwise expose /mcp/mcp. Existing /api and /ui routes
# take precedence because they were registered first.
application.mount("/", mcp.streamable_http_app())


# ============================================================
# Start server
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        application,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )

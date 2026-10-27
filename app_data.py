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


def task_rows():
    return [
        [
            task.task_id,
            task.agent_id,
            task.status,
            str(task.created_at),
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
# Gradio UI
# ============================================================

with gr.Blocks(
    title="OhamLab Agent Lab",
    theme=gr.themes.Soft(),
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

                base_url = gr.Textbox(
                    label="LLM base URL",
                    value=os.getenv("LLM_BASE_URL", ""),
                )

                model = gr.Textbox(
                    label="LLM model",
                    value=os.getenv("LLM_MODEL", ""),
                )

                api_key_env = gr.Textbox(
                    label="API key environment variable",
                    value="LLM_API_KEY",
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

        gr.Markdown("### Task history")

        current_tasks = gr.Dataframe(
            headers=[
                "Task ID",
                "Agent ID",
                "Status",
                "Created",
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

        inspect_button.click(
            inspect_task,
            inputs=[task_id_input],
            outputs=[details],
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
        cc_table = gr.Dataframe(headers=["Task ID", "Task", "Agent", "Status", "Created", "Updated", "Report", "Error"], interactive=False, label="Stored tasks", wrap=True, max_height=470)
        with gr.Row():
            cc_id = gr.Textbox(label="Inspect task ID", placeholder="Paste full task ID")
            cc_inspect = gr.Button("Open task", variant="primary")
        cc_details = gr.JSON(label="Complete task record")
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
                if q and q not in " ".join([str(t.task_id), instruction, agent_id, state]).casefold():
                    continue
                output.append([str(t.task_id), instruction, agent_id, state,
                               str(getattr(t, "created_at", "")), str(getattr(t, "updated_at", "")),
                               str(getattr(t, "report_path", "")), str(getattr(t, "error", ""))])
            return output, f"**Showing {len(output)} of {len(items)} stored tasks**"

        cc_search.change(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_status.change(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_refresh.click(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_inspect.click(inspect_task, [cc_id], [cc_details])
        cc_submit.click(run_agent_ui, [cc_agent, cc_instruction, cc_async],
                        [cc_submit_result, cc_created_id, current_tasks],
                        ).then(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
        cc_timer = gr.Timer(8)
        cc_timer.tick(command_center_rows, [cc_search, cc_status], [cc_table, cc_summary])
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


# ============================================================
# FastAPI + REST API + MCP + Gradio
# ============================================================

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

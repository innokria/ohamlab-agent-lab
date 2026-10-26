
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from ohamlab_agent_lab.auth import install_auth_middleware
from ohamlab_agent_lab.api import app as api_app, service, task_manager, runtime
from ohamlab_agent_lab.mcp_server import mcp
from ohamlab_agent_lab.models import CreateAgentRequest


# --------------------------------------------------
# Initialise data
# --------------------------------------------------

def ensure_starter_agent():
    """Create one starter agent if persistent storage is empty."""
    try:
        existing = service.list_agents()
        if existing:
            return

        agent = service.create_agent(
            CreateAgentRequest(
                name="Researcher",
                description="Starter agent for research and report tasks.",
                execution_mode="workflow",
                workflow="report",
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


# --------------------------------------------------
# Data helpers
# --------------------------------------------------

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
            task.created_at,
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


# --------------------------------------------------
# Agent actions
# --------------------------------------------------

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
            workflow=workflow or "report",
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


# --------------------------------------------------
# Task actions
# --------------------------------------------------

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

        return result, "", task_rows()

    except Exception as exc:
        raise gr.Error(f"Task failed: {exc}") from exc


def inspect_task(task_id):
    record = task_manager.get((task_id or "").strip())

    if not record:
        raise gr.Error("Task not found.")

    return record.model_dump()


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

with gr.Blocks(
    title="OhamLab Agent Lab",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # OhamLab Agent Lab

        Create agents, execute tasks, and inspect task results.
        Agents and task records are loaded from the configured storage.
        """
    )

    backend_status = gr.Markdown(
        f"**Task backend:** {task_manager.backend_status}"
    )

    with gr.Row():
        refresh_button = gr.Button("Refresh all data", variant="primary")
        data_status = gr.Markdown("Loading data...")

    with gr.Tab("Agents"):
        with gr.Row():
            with gr.Column():
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
                    ["report", "echo"],
                    value="report",
                    label="Workflow",
                )
                prompt = gr.Textbox(
                    label="System prompt",
                    lines=3,
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

            with gr.Column():
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
                )

                selected_agent = gr.Dropdown(
                    choices=agent_choices(),
                    label="Selected agent",
                )

    with gr.Tab("Run task"):
        gr.Markdown("### Execute a task")

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

        run_button = gr.Button("Run task", variant="primary")

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
        )

        refresh_tasks_button = gr.Button("Refresh task history")

    with gr.Tab("Task details"):
        task_id_input = gr.Textbox(label="Task ID")
        inspect_button = gr.Button("Inspect task")
        details = gr.JSON(label="Task details")

        inspect_button.click(
            inspect_task,
            inputs=[task_id_input],
            outputs=[details],
        )

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


# --------------------------------------------------
# FastAPI + REST + MCP + Gradio
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


application = FastAPI(
    title="OhamLab Agent Lab",
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


@application.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(url="/ui", status_code=307)


application.mount("/mcp", mcp.streamable_http_app())

application = gr.mount_gradio_app(
    application,
    demo,
    path="/ui",
)

application.mount("/", api_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        application,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )

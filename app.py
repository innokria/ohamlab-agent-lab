from __future__ import annotations

import os
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from ohamlab_agent_lab.auth import install_auth_middleware

from ohamlab_agent_lab.api import app as api_app, service, task_manager, runtime, storage
from ohamlab_agent_lab.mcp_server import mcp


def agent_choices():
    return [
        (f"{agent.name} — {agent.id[:8]}", agent.id)
        for agent in service.list_agents()
    ]


def refresh_agents():
    return [
        [a.id, a.name, a.execution_mode, a.workflow, a.published_api, a.published_mcp]
        for a in service.list_agents()
    ], gr.Dropdown(choices=agent_choices())


def create_agent_ui(name, description, mode, workflow, prompt, base_url, model, api_key_env, publish_api, publish_mcp):
    from ohamlab_agent_lab.models import CreateAgentRequest
    if not name or not name.strip():
        raise gr.Error("Enter an agent name.")
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
    rows = [
        [a.id, a.name, a.execution_mode, a.workflow, a.published_api, a.published_mcp]
        for a in service.list_agents()
    ]
    return rows, gr.Dropdown(choices=agent_choices(), value=agent.id), f"Created `{agent.name}` (`{agent.id}`)."


async def run_agent_ui(agent_id, task, asynchronous):
    if not agent_id:
        raise gr.Error("Create or select an agent.")
    if not task or not task.strip():
        raise gr.Error("Enter a task.")
    agent = service.get_agent(agent_id)
    if not agent:
        raise gr.Error("Agent not found.")
    if asynchronous:
        record = task_manager.submit(agent, task.strip())
        return f"Queued task `{record.task_id}`.", record.task_id, task_rows()
    try:
        result = await runtime.run(agent, task.strip())
        return result, "", task_rows()
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def task_rows():
    return [
        [t.task_id, t.agent_id, t.status, t.created_at, t.report_path, t.error]
        for t in task_manager.list()[:100]
    ]


def inspect_task(task_id):
    record = task_manager.get((task_id or "").strip())
    if not record:
        raise gr.Error("Task not found.")
    return record.model_dump()


with gr.Blocks(title="OhamLab Agent Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# OhamLab Agent Lab\n"
        "Standalone agents with optional LLMs, a REST API, and an inbound MCP interface. "
        "This app does not require DSH."
    )
    gr.Markdown(f"**Task backend:** {task_manager.backend_status}")
    gr.Markdown("API documentation: `/docs` · Health: `/health` · MCP: `/mcp`")

    with gr.Tab("Create agent"):
        with gr.Row():
            with gr.Column():
                name = gr.Textbox(label="Name", placeholder="Researcher")
                description = gr.Textbox(label="Description")
                mode = gr.Radio(["workflow", "llm"], value="workflow", label="Execution mode")
                workflow = gr.Dropdown(["report", "echo"], value="report", label="Workflow")
                prompt = gr.Textbox(label="System prompt", lines=3)
                base_url = gr.Textbox(label="LLM base URL", value=os.getenv("LLM_BASE_URL", ""))
                model = gr.Textbox(label="LLM model", value=os.getenv("LLM_MODEL", ""))
                api_key_env = gr.Textbox(label="API key environment variable name", value="LLM_API_KEY")
                publish_api = gr.Checkbox(value=True, label="Publish through REST API")
                publish_mcp = gr.Checkbox(value=True, label="Publish through MCP")
                create_button = gr.Button("Create agent", variant="primary")
                create_status = gr.Markdown()
            with gr.Column():
                agents = gr.Dataframe(
                    headers=["ID", "Name", "Mode", "Workflow", "REST", "MCP"],
                    datatype=["str", "str", "str", "str", "bool", "bool"],
                    value=[
                        [a.id, a.name, a.execution_mode, a.workflow, a.published_api, a.published_mcp]
                        for a in service.list_agents()
                    ],
                    interactive=False,
                )
                refresh_button = gr.Button("Refresh")
                selected_agent = gr.Dropdown(choices=agent_choices(), label="Agent for task")

        create_button.click(
            create_agent_ui,
            inputs=[name, description, mode, workflow, prompt, base_url, model, api_key_env, publish_api, publish_mcp],
            outputs=[agents, selected_agent, create_status],
        )
        refresh_button.click(refresh_agents, outputs=[agents, selected_agent])

    with gr.Tab("Run task"):
        selected_agent_run = gr.Dropdown(choices=agent_choices(), label="Agent")
        task = gr.Textbox(label="Task", lines=5, placeholder="Describe the work for this agent.")
        asynchronous = gr.Checkbox(value=True, label="Run as background task")
        run_button = gr.Button("Run", variant="primary")
        run_result = gr.Markdown()
        task_id = gr.Textbox(label="Task ID")
        current_tasks = gr.Dataframe(
            headers=["Task ID", "Agent ID", "Status", "Created", "Report", "Error"],
            datatype=["str", "str", "str", "str", "str", "str"],
            value=task_rows(),
            interactive=False,
        )
        refresh_tasks = gr.Button("Refresh task list")
        run_button.click(run_agent_ui, inputs=[selected_agent_run, task, asynchronous], outputs=[run_result, task_id, current_tasks])
        refresh_tasks.click(task_rows, outputs=[current_tasks])

    with gr.Tab("Task details"):
        task_id_input = gr.Textbox(label="Task ID")
        inspect_button = gr.Button("Inspect")
        details = gr.JSON(label="Task record")
        inspect_button.click(inspect_task, inputs=[task_id_input], outputs=[details])

    timer = gr.Timer(5)
    timer.tick(task_rows, outputs=[current_tasks])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastMCP uses this lifespan to manage Streamable HTTP sessions.
    async with mcp.session_manager.run():
        yield


application = FastAPI(title="OhamLab Agent Lab", lifespan=lifespan)
install_auth_middleware(application)
application.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Mount specific routes before the catch-all REST app. The parent middleware
# protects both the mounted REST app and the separate MCP ASGI transport.
application.mount("/mcp", mcp.streamable_http_app())
application = gr.mount_gradio_app(application, demo, path="/ui")
application.mount("/", api_app)

# UI: /ui | REST API: /api/... | MCP: /mcp | Health: /health
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

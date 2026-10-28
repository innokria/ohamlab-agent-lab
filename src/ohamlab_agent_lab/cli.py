from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .agents import AgentStore
from .config import load_config
from .runtime import AgentRuntime


async def _run_one(runtime, store, agent, task):
    try:
        answer = await runtime.ask(
            task,
            system_prompt=agent.get("system_prompt"),
            allowed_tools=agent.get("tools") or None,
        )
        report = store.save_report(agent, task, answer)
        return {"agent": agent["name"], "id": agent["id"], "status": "completed",
                "answer": answer, "report": str(report)}
    except Exception as exc:
        report = store.save_report(agent, task, "", status="failed", error=str(exc))
        return {"agent": agent["name"], "id": agent["id"], "status": "failed",
                "error": str(exc), "report": str(report)}


async def _run(args):
    store = AgentStore(args.data_dir)

    if args.command == "create-agent":
        agent = store.create(args.name, args.role, args.system_prompt, args.tools)
        print(json.dumps(agent, indent=2))
        return 0

    if args.command == "clone-agent":
        agent = store.clone(args.source, args.name)
        print(json.dumps(agent, indent=2))
        return 0

    if args.command == "list-agents":
        print(json.dumps(store.list(), indent=2))
        return 0

    if args.command in {"run-agent", "run-swarm"}:
        config = load_config(args.config)
        runtime = AgentRuntime(config)
        if args.command == "run-agent":
            agent = store.get(args.agent)
            result = await _run_one(runtime, store, agent, args.task)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "completed" else 1

        agents = []
        for selector in args.agents:
            agents.append(store.get(selector))
        if not agents:
            print("No agents selected.", file=sys.stderr)
            return 2

        # Each agent gets the task independently. MCP/LLM provider capacity may limit concurrency.
        results = await asyncio.gather(
            *[_run_one(runtime, store, agent, args.task) for agent in agents]
        )
        summary = {
            "task": args.task,
            "agents_requested": len(agents),
            "completed": sum(r["status"] == "completed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "results": results,
        }
        summary_path = store.reports_dir / "swarm-summary.md"
        lines = ["# Agent Swarm Summary", "", f"**Task:** {args.task}", "",
                 f"- Agents: {len(agents)}", f"- Completed: {summary['completed']}",
                 f"- Failed: {summary['failed']}", "", "## Agent results", ""]
        for result in results:
            lines += [f"### {result['agent']} — {result['status']}", "",
                      f"- Report: `{result['report']}`", ""]
            if result.get("answer"):
                lines += [result["answer"], ""]
            if result.get("error"):
                lines += [f"Error: {result['error']}", ""]
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        summary["summary_report"] = str(summary_path)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["failed"] == 0 else 1

    config = load_config(args.config)
    if args.command == "check":
        print(json.dumps({
            "status": "config-valid",
            "mcp_transport": config.mcp.transport,
            "llm_provider": config.llm.provider,
            "model": config.llm.model,
            "secrets_printed": False,
        }, indent=2))
        return 0

    runtime = AgentRuntime(config)
    if args.command == "list-tools":
        print(json.dumps(await runtime.list_tools(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "ask":
        print(await runtime.ask(args.question))
        return 0
    return 2


def main():
    parser = argparse.ArgumentParser(prog="ohamlab-lab")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", default=".ohamlab", help="Persistent agents/reports directory")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start local UI, API and MCP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)

    sub.add_parser("check", help="Validate config without printing secrets")
    sub.add_parser("list-tools", help="Connect to MCP and list discovered tools")
    ask = sub.add_parser("ask", help="Ask the configured LLM; it may call MCP tools")
    ask.add_argument("question")

    create = sub.add_parser("create-agent", help="Create and persist an agent")
    create.add_argument("name")
    create.add_argument("--role", default="research assistant")
    create.add_argument("--system-prompt")
    create.add_argument("--tools", nargs="*", default=[], help="Optional MCP tool-name allowlist")

    clone = sub.add_parser("clone-agent", help="Clone an existing agent definition")
    clone.add_argument("source", help="Source agent ID or exact name")
    clone.add_argument("--name", help="Name for the clone")

    sub.add_parser("list-agents", help="List saved agents")

    run = sub.add_parser("run-agent", help="Assign a task to one agent and write a report")
    run.add_argument("agent", help="Agent ID or exact name")
    run.add_argument("task")
    swarm = sub.add_parser("run-swarm", help="Run one task concurrently across selected agents")
    swarm.add_argument("task")
    swarm.add_argument("--agents", nargs="+", required=True, help="Agent IDs or exact names")

    args = parser.parse_args()
    try:
        if args.command == "serve":
            import uvicorn
            uvicorn.run("app_data:application", host=args.host, port=args.port)
            return 0
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

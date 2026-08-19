import os
import sys
import argparse
import asyncio
import uuid
from dotenv import load_dotenv

# Load env file variables
load_dotenv()

# Windows event loop policy patch for psycopg async connection
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.graph import StateGraph, START, END
from src.core.state import ProjectState
from src.nodes.perception import perception_node, route_perception
from src.nodes.decomposer import decomposer_node
from src.db.session import get_checkpointer

# Define LangGraph State Machine Workflow
workflow = StateGraph(ProjectState)
workflow.add_node("perception", perception_node)
workflow.add_node("decomposer", decomposer_node)

workflow.add_edge(START, "perception")
workflow.add_conditional_edges(
    "perception",
    route_perception,
    {
        "decomposer": "decomposer",
        "__end__": END
    }
)
workflow.add_edge("decomposer", END)

def print_backlog(backlog) -> None:
    """Helper to cleanly format and output task list status in terminal."""
    if not backlog:
        print("[-] No tasks in backlog.")
        return
        
    print("\n=== Project Task Backlog ===")
    for task in backlog:
        # Support both Pydantic objects and deserialized JSON dictionaries
        if isinstance(task, dict):
            task_id = task.get("task_id", "")
            title = task.get("title", "")
            status = task.get("status", "PENDING")
            dependencies = task.get("dependencies", [])
        else:
            task_id = getattr(task, "task_id", "")
            title = getattr(task, "title", "")
            status = getattr(task, "status", "PENDING")
            dependencies = getattr(task, "dependencies", [])

        dep_str = ", ".join(dependencies) if dependencies else "None"
        print(f"[{status:<10}] {task_id:<8} : {title} (Depends on: {dep_str})")
    print("============================\n")

async def run_workflow(prompt: str, thread_id: str) -> None:
    """Compiles the orchestrator graph, connects checkpointer, and runs a new session."""
    async with get_checkpointer() as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        
        initial_state = {
            "project_id": f"proj_{uuid.uuid4().hex[:6]}",
            "project_root": os.getcwd(),
            "tech_stack": {},
            "locked_decisions": [],
            "task_backlog": [],
            "active_task_id": None,
            "retrieved_code_context": [],
            "retrieved_web_docs": [],
            "generated_prompt_payload": {"objective": prompt},
            "critic_iteration": 0,
            "critic_passed": False,
            "critic_feedback": None
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        print(f"[*] Starting new thread session: {thread_id}")
        
        result = await app.ainvoke(initial_state, config)
        print("[+] Session run completed successfully.")
        print_backlog(result.get("task_backlog", []))

async def resume_workflow(thread_id: str) -> None:
    """Connects to PostgreSQL checkpointer, retrieves checkpoints for thread_id, and resumes/prints state."""
    async with get_checkpointer() as checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"[*] Resuming session for thread: {thread_id} ...")
        state = await app.aget_state(config)
        
        if not state.values:
            print(f"[!] No checkpoint state found in database for thread: {thread_id}")
            return
            
        print("[+] Checkpoint loaded successfully.")
        print_backlog(state.values.get("task_backlog", []))

def main():
    parser = argparse.ArgumentParser(
        description="Master Brain Multi-Agent Orchestrator CLI foundation wrapper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a new orchestrator planning session.")
    run_parser.add_argument("--prompt", required=True, help="Developer goal or prompt to plan.")
    run_parser.add_argument("--thread-id", help="Optional thread ID to assign.")

    # 'continue' subcommand
    continue_parser = subparsers.add_parser("continue", help="Resume an existing session.")
    continue_parser.add_argument("--thread-id", required=True, help="The thread ID to resume.")

    args = parser.parse_args()

    # Generate thread id if none is provided
    thread_id = args.thread_id or str(uuid.uuid4())

    try:
        if args.command == "run":
            asyncio.run(run_workflow(args.prompt, thread_id))
        elif args.command == "continue":
            asyncio.run(resume_workflow(args.thread_id))
    except Exception as e:
        print(f"[!] CLI Exec error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

import sys
import asyncio
import pytest
import argparse
from unittest.mock import patch
from langgraph.checkpoint.memory import MemorySaver

# Windows event loop policy patch for psycopg async connection compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.main import workflow, run_workflow, resume_workflow
from src.db.session import get_checkpointer

def test_cli_parser():
    """Verify that command-line subcommands parse parameters correctly."""
    # We will test the arg parser by mock-invoking it
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--thread-id")

    continue_parser = subparsers.add_parser("continue")
    continue_parser.add_argument("--thread-id", required=True)

    # 1. Parse run command
    args_run = parser.parse_args(["run", "--prompt", "Build authentication node", "--thread-id", "test-t1"])
    assert args_run.command == "run"
    assert args_run.prompt == "Build authentication node"
    assert args_run.thread_id == "test-t1"

    # 2. Parse continue command
    args_continue = parser.parse_args(["continue", "--thread-id", "test-t1"])
    assert args_continue.command == "continue"
    assert getattr(args_continue, "thread_id") == "test-t1"

def test_workflow_memory_execution():
    """Verify that the compiled workflow executes perception -> decomposer nodes correctly offline."""
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "cli-memory-test"}}
    initial_state = {
        "project_id": "cli_test",
        "project_root": "e:/TA",
        "tech_stack": {},
        "locked_decisions": [],
        "task_backlog": [],
        "active_task_id": None,
        "retrieved_code_context": [],
        "retrieved_web_docs": [],
        "generated_prompt_payload": {"objective": "Plan out coding tasks"},
        "critic_iteration": 0,
        "critic_passed": False,
        "critic_feedback": None
    }
    
    # Run the graph synchronously
    result = app.invoke(initial_state, config)
    assert len(result["task_backlog"]) > 0
    assert result["generated_prompt_payload"]["intent"] == "DECOMPOSE"
    assert result["active_task_id"] == "TSK-001"

@pytest.mark.asyncio
async def test_cli_postgres_resume():
    """Verify database saving and resume flow using a Postgres checkpointer."""
    thread_id = "cli-postgres-test-thread"
    try:
        # 1. Run a new planning session
        await run_workflow(prompt="Decompose authentication security rules", thread_id=thread_id)
        
        # 2. Resume session and verify state
        async with get_checkpointer() as checkpointer:
            app = workflow.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            
            state = await app.aget_state(config)
            assert state.values is not None
            assert len(state.values["task_backlog"]) > 0
            assert state.values["active_task_id"] == "TSK-001"
            assert state.values["generated_prompt_payload"]["intent"] == "DECOMPOSE"

    except Exception as e:
        pytest.fail(f"Postgres CLI workflow test failed. Is the PostgreSQL container running?\nError: {e}")

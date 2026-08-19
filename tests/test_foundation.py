import sys
import asyncio
import pytest

# Configure event loop policy for Windows psycopg compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.memory import MemorySaver
from src.core.state import TaskItem, CodeReference, ProjectState
from src.core.config import settings
from src.core.graph import get_compiled_graph
from src.db.session import get_checkpointer

def test_state_schema():
    """Verify that state schema items initialize and validate correctly."""
    ref = CodeReference(
        file_path="src/main.py",
        symbol_name="main",
        start_line=1,
        end_line=10,
        code_snippet="def main(): pass"
    )
    assert ref.file_path == "src/main.py"
    assert ref.start_line == 1

    task = TaskItem(
        task_id="TSK-001",
        title="Database Setup",
        description="Verify checkpointer",
        status="PENDING",
        acceptance_criteria=["Verify connection"]
    )
    assert task.task_id == "TSK-001"
    assert task.status == "PENDING"

def test_memory_saver():
    """Verify StateGraph compilation and execution using in-memory saver (offline check)."""
    memory = MemorySaver()
    graph = get_compiled_graph(memory)
    
    # Initialize input state
    initial_state = {
        "project_id": "test_proj",
        "critic_iteration": 0
    }
    
    config = {"configurable": {"thread_id": "test_thread"}}
    
    # Run graph
    result = graph.invoke(initial_state, config)
    assert result["critic_iteration"] == 1
    assert result["active_task_id"] == "TSK-001-ACTIVE"

@pytest.mark.asyncio
async def test_postgres_checkpointer():
    """
    Verify PostgreSQL checkpointing by running the graph and recovering state.
    Fails with descriptive message if database is not reachable.
    """
    try:
        async with get_checkpointer() as checkpointer:
            graph = get_compiled_graph(checkpointer)
            
            config = {"configurable": {"thread_id": "thread-postgres-verify"}}
            
            # Initial Run
            initial_state = {
                "project_id": "db_test",
                "critic_iteration": 0
            }
            res = await graph.ainvoke(initial_state, config)
            assert res["critic_iteration"] == 1
            
            # Verify database checkpoint exists and state matches
            state = await graph.aget_state(config)
            assert state.values["critic_iteration"] == 1
            assert state.values["active_task_id"] == "TSK-001-ACTIVE"
            
    except Exception as e:
        pytest.fail(
            f"Failed to connect or test PostgreSQL checkpointer. "
            f"If Docker is not running, please start Docker Desktop and run "
            f"'docker compose up -d postgres' in your terminal.\n"
            f"Error details: {e}"
        )

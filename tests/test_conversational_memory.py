import os
import json
import pytest
from src.core.state import ProjectState
from src.core.graph import get_compiled_graph

RESEARCH_HISTORY_FILE = "docs/research_history.json"

@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown directories for tests."""
    yield
    # Cleanup research cache after tests run to maintain clean workspace
    if os.path.exists(RESEARCH_HISTORY_FILE):
        try:
            os.remove(RESEARCH_HISTORY_FILE)
        except Exception:
            pass

def test_chat_memory_remembers_past_turns():
    """Verify that chat history persists and compiles user and assistant turns sequentially."""
    app = get_compiled_graph()

    # Prepopulate history with prior turns
    initial_history = [
        {"role": "user", "content": "My database is PostgreSQL"},
        {"role": "assistant", "content": "Great, PostgreSQL is standard."}
    ]

    state = ProjectState(
        project_id="test_memory_thread",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "What database did I mention?"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=False,
        conversational_response=None,
        chat_history=initial_history
    )

    result = app.invoke(state)

    # 1. Assert intent classified as CHAT
    assert result["generated_prompt_payload"]["intent"] == "CHAT"

    # 2. Assert history was updated and has 4 total turns (2 initial + 2 new)
    updated_history = result["chat_history"]
    assert len(updated_history) == 4
    assert updated_history[2]["role"] == "user"
    assert updated_history[2]["content"] == "What database did I mention?"
    assert updated_history[3]["role"] == "assistant"

def test_chat_research_executes_and_caches():
    """Verify that RESEARCH intent runs web searches, writes JSON caches to disk, and feeds the chat response."""
    app = get_compiled_graph()

    state = ProjectState(
        project_id="test_research_thread",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "who is the president of america 2026"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=False,
        conversational_response=None,
        chat_history=[]
    )

    result = app.invoke(state)

    # 1. Assert intent classified as RESEARCH
    assert result["generated_prompt_payload"]["intent"] == "RESEARCH"

    # 2. Assert web docs context was gathered
    assert len(result["retrieved_web_docs"]) > 0

    # 3. Assert research history cache was written persistently to disk
    assert os.path.exists(RESEARCH_HISTORY_FILE)
    with open(RESEARCH_HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["query"] == "who is the president of america 2026"
    assert "results" in data[0]

    # 4. Assert conversational response shows search results
    assert result["conversational_response"] is not None
    assert "President" in result["conversational_response"]

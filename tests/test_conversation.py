import pytest
from src.core.state import ProjectState
from src.core.graph import get_compiled_graph

def test_chat_intent_routes_to_conversational_node():
    """Verify that greetings like 'hey' route to the conversational node and return assistant text replies."""
    app = get_compiled_graph()

    state = ProjectState(
        project_id="test_chat_greeting",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "hey"},  # Greeting prompt
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=False,
        conversational_response=None
    )

    result = app.invoke(state)

    # 1. Assert intent was classified as CHAT
    assert result["generated_prompt_payload"]["intent"] == "CHAT"

    # 2. Assert conversational response was populated with greeting reply
    assert result["conversational_response"] is not None
    assert "Hello! I am ResearchRealy" in result["conversational_response"]

    # 3. Assert no roadmap tasks or spec files were created
    assert len(result["task_backlog"]) == 0
    assert result["critic_passed"] is False

def test_chat_intent_handles_brainstorming():
    """Verify that general open-ended questions route to the conversational node and get assistance."""
    app = get_compiled_graph()

    state = ProjectState(
        project_id="test_chat_brainstorm",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "Suggest ideas for user session management"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=False,
        conversational_response=None
    )

    result = app.invoke(state)

    # 1. Assert intent was classified as CHAT
    assert result["generated_prompt_payload"]["intent"] == "CHAT"

    # 2. Assert conversational response was populated with general help reply
    assert result["conversational_response"] is not None
    assert "Suggest ideas" in result["conversational_response"]

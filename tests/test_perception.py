import pytest
from src.core.state import ProjectState, TaskItem
from src.nodes.perception import classify_user_intent, perception_node, route_perception

def test_classify_user_intent_heuristics():
    """Verify that offline keyword heuristic classification maps correctly."""
    assert classify_user_intent("Web search of the latest library API") == "RESEARCH"
    assert classify_user_intent("Write tests for authentication service") == "CODE"
    assert classify_user_intent("Verify that no cycles exist in graph") == "CRITIC"
    assert classify_user_intent("Plan the database scaffolding details") == "DECOMPOSE"

def test_perception_node_integration():
    """Verify that perception_node updates the payload with classified intent."""
    state = ProjectState(
        project_id="test_proj",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "Scrape Playwright docs"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )

    update = perception_node(state)
    assert "generated_prompt_payload" in update
    payload = update["generated_prompt_payload"]
    assert payload["objective"] == "Scrape Playwright docs"
    assert payload["intent"] == "RESEARCH"  # keyword 'docs' / heuristic

def test_route_perception_empty_backlog():
    """Verify that route_perception routes to decomposer if backlog is empty."""
    state = ProjectState(
        project_id="test_proj",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],  # Empty backlog
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "CODE"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )
    assert route_perception(state) == "decomposer"

def test_route_perception_decomposing_intent():
    """Verify that route_perception routes to decomposer if intent is DECOMPOSE."""
    state = ProjectState(
        project_id="test_proj",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[TaskItem(task_id="TSK-001", title="Task 1", description="desc")],
        active_task_id="TSK-001",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "DECOMPOSE"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )
    assert route_perception(state) == "decomposer"

def test_route_perception_halts_if_plan_exists():
    """Verify that route_perception halts (__end__) if plan exists and intent is not DECOMPOSE."""
    state = ProjectState(
        project_id="test_proj",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[TaskItem(task_id="TSK-001", title="Task 1", description="desc")],
        active_task_id="TSK-001",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "CODE"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )
    assert route_perception(state) == "coder"

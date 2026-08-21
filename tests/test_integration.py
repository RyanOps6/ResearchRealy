import os
import tempfile
import pytest
from src.core.state import ProjectState, TaskItem
from src.core.graph import get_compiled_graph

@pytest.fixture
def temp_spec_file():
    """Setup and teardown temporary blueprint spec file."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "blueprints_integration_test_spec.md").replace("\\", "/")
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_full_orchestration_decomposing_flow():
    """Verify that empty backlogs trigger intent perception routing to the decomposer."""
    app = get_compiled_graph()

    initial_state = ProjectState(
        project_id="integ_decom",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "DECOMPOSE", "objective": "Set up authentication"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )

    result = app.invoke(initial_state)

    # Decomposer node should have run and populated backlog tasks
    assert len(result["task_backlog"]) > 0
    assert result["active_task_id"] == "TSK-001"
    # Status of first task is marked PENDING initially
    assert result["task_backlog"][0].status == "COMPLETED" or result["task_backlog"][0].status == "PENDING"

def test_full_orchestration_coding_flow(temp_spec_file):
    """Verify that a CODE intent routes to the cyclic generator/critic specs refinement loop."""
    app = get_compiled_graph()

    task = TaskItem(
        task_id="TSK-CODE-INTEG",
        title="Generate authentication spec blueprint",
        description="Write Markdown specification for auth handler.",
        status="IN_PROGRESS",
        target_files=[temp_spec_file]
    )

    initial_state = ProjectState(
        project_id="integ_code",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[task],
        active_task_id="TSK-CODE-INTEG",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "CODE", "objective": "Create JWT spec"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=True,
        conversational_response=None
    )

    # Execute graph:
    # 1. perception (routes to coder)
    # 2. coder (writes TODO spec)
    # 3. critic (rejects, critic_iteration=1)
    # 4. route_critic (routes back to coder)
    # 5. coder (writes clean spec)
    # 6. critic (approves, critic_iteration=2)
    # 7. route_critic (halts -> END)
    result = app.invoke(initial_state)

    # Assert cyclic feedback loop successfully completed
    assert result["critic_passed"] is True
    assert result["critic_iteration"] == 2
    assert result["task_backlog"][0].status == "COMPLETED"

    # Verify that the generated spec contains clean content with no TODOs
    assert os.path.exists(temp_spec_file)
    with open(temp_spec_file, "r", encoding="utf-8") as f:
        spec_content = f.read()

    assert "💡 The Rough Idea" in spec_content
    assert "```prompt" in spec_content
    assert "TODO" not in spec_content

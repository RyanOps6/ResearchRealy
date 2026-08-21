import os
import tempfile
import pytest
from src.core.state import ProjectState, TaskItem
from src.core.graph import get_compiled_graph

@pytest.fixture
def temp_spec_path():
    """Setup and teardown a temporary file path for specifications."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "permission_gate_test_spec.md").replace("\\", "/")
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_permission_gate_blocks_writing(temp_spec_path):
    """Verify that when permission_granted is False, the graph blocks writing and asks for confirmation."""
    app = get_compiled_graph()

    task = TaskItem(
        task_id="TSK-GATE-BLOCK",
        title="Generate test spec",
        description="Write Markdown specification.",
        status="IN_PROGRESS",
        target_files=[temp_spec_path]
    )

    state = ProjectState(
        project_id="test_gate_block",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[task],
        active_task_id="TSK-GATE-BLOCK",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "CODE", "objective": "Create JWT spec"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=False,  # BLOCKED!
        conversational_response=None
    )

    result = app.invoke(state)

    # 1. Assert it populated the conversational response prompt
    assert result["conversational_response"] is not None
    assert "Would you like me to write the base specification files now?" in result["conversational_response"]

    # 2. Assert it did NOT execute the coder specification-writing loop
    assert result["critic_iteration"] == 0
    assert result["critic_passed"] is False

    # 3. Assert no file was written to disk
    assert not os.path.exists(temp_spec_path)

def test_permission_gate_allows_writing_on_approval(temp_spec_path):
    """Verify that when permission_granted is True, the graph runs the coder and writes specifications."""
    app = get_compiled_graph()

    task = TaskItem(
        task_id="TSK-GATE-ALLOW",
        title="Generate test spec",
        description="Write Markdown specification.",
        status="IN_PROGRESS",
        target_files=[temp_spec_path]
    )

    state = ProjectState(
        project_id="test_gate_allow",
        project_root="e:/TA",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[task],
        active_task_id="TSK-GATE-ALLOW",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"intent": "CODE", "objective": "Create JWT spec"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None,
        permission_granted=True,  # APPROVED!
        conversational_response=None
    )

    result = app.invoke(state)

    # 1. Assert it did NOT set a permission prompt
    assert result["conversational_response"] is None

    # 2. Assert it executed the coder/critic loop successfully
    assert result["critic_passed"] is True
    assert result["critic_iteration"] == 2

    # 3. Assert file was written to disk and has no placeholders
    assert os.path.exists(temp_spec_path)
    with open(temp_spec_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "💡 The Rough Idea" in content
    assert "```prompt" in content
    assert "TODO" not in content

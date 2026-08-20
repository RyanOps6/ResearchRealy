import os
import tempfile
import pytest
from src.core.state import ProjectState, TaskItem
from src.nodes.critic import (
    evaluate_task_output,
    run_heuristic_evaluation,
    critic_node,
    CriticEvaluation
)

def test_critic_evaluation_heuristics():
    """Verify that placeholder implementations are rejected, and clean code is approved."""
    
    # 1. Approved case
    clean_code = (
        "def compute_average(values):\n"
        "    if not values:\n"
        "        return 0\n"
        "    return sum(values) / len(values)\n"
    )
    res_pass = run_heuristic_evaluation(clean_code)
    assert res_pass.is_approved is True
    assert len(res_pass.issues) == 0

    # 2. Rejected case: empty snippet
    res_empty = run_heuristic_evaluation("")
    assert res_empty.is_approved is False
    assert "empty" in res_empty.feedback

    # 3. Rejected case: TODO marker
    todo_code = "def authenticate():\n    # TODO: implement authentication\n    pass"
    res_todo = run_heuristic_evaluation(todo_code)
    assert res_todo.is_approved is False
    assert any("TODO" in issue for issue in res_todo.issues)

    # 4. Rejected case: NotImplementedError stub
    nie_code = "def fetch_records():\n    raise NotImplementedError()\n"
    res_nie = run_heuristic_evaluation(nie_code)
    assert res_nie.is_approved is False
    assert any("NotImplementedError" in issue for issue in res_nie.issues)

def test_critic_node_integration():
    """Verify that critic_node reads target files, evaluates them, and updates graph state backlog."""
    
    # 1. Create a temporary target file representing code written by the agent
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as temp_file:
        temp_file.write("def calculate_sum(a, b):\n    return a + b\n")
        temp_path = temp_file.name

    try:
        # Standardize path slashes for consistency
        normalized_path = os.path.normpath(temp_path).replace("\\", "/")

        # 2. Initialize project state
        task = TaskItem(
            task_id="TSK-X",
            title="Create sum function",
            description="Write a Python function to calculate sum.",
            status="IN_PROGRESS",
            target_files=[normalized_path]
        )
        
        state = ProjectState(
            project_id="test_proj",
            project_root="",
            tech_stack={},
            locked_decisions=[],
            task_backlog=[task],
            active_task_id="TSK-X",
            retrieved_code_context=[],
            retrieved_web_docs=[],
            generated_prompt_payload=None,
            critic_iteration=0,
            critic_passed=False,
            critic_feedback=None
        )

        # 3. Execute the node
        updates = critic_node(state)

        # 4. Assert updates were set correctly
        assert updates["critic_iteration"] == 1
        assert updates["critic_passed"] is True
        assert updates["critic_feedback"] is None
        
        # Verify status in backlog was updated to COMPLETED
        updated_task = updates["task_backlog"][0]
        assert updated_task.status == "COMPLETED"

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

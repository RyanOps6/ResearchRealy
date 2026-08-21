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
    """Verify that placeholder specifications are rejected, and clean blueprints are approved."""
    
    # 1. Approved case
    clean_spec = (
        "# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler\n\n"
        "## 💡 The Rough Idea\n"
        "We want to create a secure token manager that issues temporary access keys.\n\n"
        "```prompt\n"
        "Act as an expert Python developer. Write the code for src/auth.py.\n"
        "```\n"
    )
    res_pass = run_heuristic_evaluation(clean_spec)
    assert res_pass.is_approved is True
    assert len(res_pass.issues) == 0

    # 2. Rejected case: empty specification
    res_empty = run_heuristic_evaluation("")
    assert res_empty.is_approved is False
    assert "empty" in res_empty.feedback

    # 3. Rejected case: missing structure (no Rough Idea paragraph)
    bad_structure_1 = (
        "# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler\n\n"
        "```prompt\n"
        "Write code for src/auth.py.\n"
        "```\n"
    )
    res_bad_1 = run_heuristic_evaluation(bad_structure_1)
    assert res_bad_1.is_approved is False
    assert any("Rough Idea" in issue for issue in res_bad_1.issues)

    # 4. Rejected case: missing structure (no Prompt Recipe block)
    bad_structure_2 = (
        "# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler\n\n"
        "## 💡 The Rough Idea\n"
        "We want to create a secure token manager that issues temporary access keys.\n"
    )
    res_bad_2 = run_heuristic_evaluation(bad_structure_2)
    assert res_bad_2.is_approved is False
    assert any("prompt" in issue for issue in res_bad_2.issues)

    # 5. Rejected case: TODO marker inside spec
    todo_spec = (
        "# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler\n\n"
        "## 💡 The Rough Idea\n"
        "We need to implement a JWT authentication handler. # TODO: explain logic details.\n\n"
        "```prompt\n"
        "Write code for src/auth.py.\n"
        "```\n"
    )
    res_todo = run_heuristic_evaluation(todo_spec)
    assert res_todo.is_approved is False
    assert any("TODO" in issue for issue in res_todo.issues)

def test_critic_node_integration():
    """Verify that critic_node reads target Markdown blueprints, evaluates them, and updates state."""
    
    # 1. Create a temporary target spec file representing markdown written by the generator
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(
            "# 📋 Copilot/ChatGPT Prompt Recipe: Sum Function\n\n"
            "## 💡 The Rough Idea\n"
            "We want to create a simple sum function.\n\n"
            "```prompt\n"
            "Write a Python function to calculate sum.\n"
            "```\n"
        )
        temp_path = temp_file.name

    try:
        normalized_path = os.path.normpath(temp_path).replace("\\", "/")

        # 2. Initialize project state
        task = TaskItem(
            task_id="TSK-X",
            title="Create sum blueprint",
            description="Write a specification for a sum function.",
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

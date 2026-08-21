import os
import tempfile
import pytest
from langgraph.graph import StateGraph, START, END
from src.core.state import ProjectState, TaskItem
from src.nodes.critic import critic_node
from src.core.graph_feedback import (
    coder_node,
    route_critic,
    compiled_feedback_graph
)

@pytest.fixture
def temp_task_file():
    """Setup and teardown a temporary markdown spec file for writing simulation."""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "simulated_target_spec.md").replace("\\", "/")
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_feedback_loop_repairs_and_passes(temp_task_file):
    """Verify that the graph cycles to repair stubbed specifications and terminates once passed."""
    task = TaskItem(
        task_id="TSK-FEEDBACK",
        title="Simulated Task",
        description="Generate a clean Markdown blueprint.",
        status="IN_PROGRESS",
        target_files=[temp_task_file]
    )
    
    state = ProjectState(
        project_id="test_feedback_pass",
        project_root="",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[task],
        active_task_id="TSK-FEEDBACK",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload=None,
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )

    # Invoke graph:
    # 1. coder_node (writes stub containing TODO)
    # 2. critic_node (evaluates -> fails on TODO, critic_iteration=1)
    # 3. route_critic (returns coder)
    # 4. coder_node (writes clean spec because iteration=1)
    # 5. critic_node (evaluates -> passes, critic_iteration=2)
    # 6. route_critic (returns __end__)
    final_state = compiled_feedback_graph.invoke(state)

    assert final_state["critic_passed"] is True
    assert final_state["critic_iteration"] == 2
    assert final_state["critic_feedback"] is None
    
    # Assert backlog task is completed
    assert final_state["task_backlog"][0].status == "COMPLETED"

def test_feedback_loop_max_retries_guard(temp_task_file):
    """Verify that the graph terminates on maximum iteration limits if spec validation keeps failing."""
    
    # We define a custom coder node that always writes stub specifications containing TODOs
    def bad_coder_node(state: ProjectState) -> dict:
        active_task_id = state.get("active_task_id")
        backlog = state.get("task_backlog", [])
        
        active_task = None
        for t in backlog:
            if t.task_id == active_task_id:
                active_task = t
                break
                
        if active_task:
            for path in active_task.target_files:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(
                        "# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler\n\n"
                        "## 💡 The Rough Idea\n"
                        "We need to implement a JWT authentication handler. # TODO: always broken.\n\n"
                        "```prompt\n"
                        "Write code for src/auth.py.\n"
                        "```\n"
                    )
        return {}

    # Compile a test workflow with the broken coder
    test_workflow = StateGraph(ProjectState)
    test_workflow.add_node("coder", bad_coder_node)
    test_workflow.add_node("critic", critic_node)
    test_workflow.add_edge(START, "coder")
    test_workflow.add_edge("coder", "critic")
    test_workflow.add_conditional_edges(
        "critic",
        route_critic,
        {
            "coder": "coder",
            "__end__": END
        }
    )
    test_graph = test_workflow.compile()

    task = TaskItem(
        task_id="TSK-RETRY-LIMIT",
        title="Simulated Failed Task",
        description="Generate spec.",
        status="IN_PROGRESS",
        target_files=[temp_task_file]
    )
    
    state = ProjectState(
        project_id="test_limit",
        project_root="",
        tech_stack={},
        locked_decisions=[],
        task_backlog=[task],
        active_task_id="TSK-RETRY-LIMIT",
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload=None,
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )

    final_state = test_graph.invoke(state)

    # Graph should exit on iteration limit guard (iteration = 3)
    assert final_state["critic_passed"] is False
    assert final_state["critic_iteration"] == 3
    assert final_state["task_backlog"][0].status == "IN_PROGRESS"

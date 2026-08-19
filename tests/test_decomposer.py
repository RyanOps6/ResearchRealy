import pytest
from src.core.state import TaskItem, ProjectState
from src.nodes.decomposer import validate_dependencies, decomposer_node

def test_validate_dependencies_valid_dag():
    """Verify that a valid Directed Acyclic Graph (DAG) passes validation."""
    tasks = [
        TaskItem(
            task_id="TSK-001",
            title="Task 1",
            description="First task",
            dependencies=[]
        ),
        TaskItem(
            task_id="TSK-002",
            title="Task 2",
            description="Second task",
            dependencies=["TSK-001"]
        ),
        TaskItem(
            task_id="TSK-003",
            title="Task 3",
            description="Third task",
            dependencies=["TSK-002"]
        )
    ]
    assert validate_dependencies(tasks) is True

def test_validate_dependencies_direct_cycle():
    """Verify that a direct circular dependency (A -> B -> A) is caught."""
    tasks = [
        TaskItem(
            task_id="TSK-001",
            title="Task 1",
            description="First task",
            dependencies=["TSK-002"]
        ),
        TaskItem(
            task_id="TSK-002",
            title="Task 2",
            description="Second task",
            dependencies=["TSK-001"]
        )
    ]
    assert validate_dependencies(tasks) is False

def test_validate_dependencies_circular_cycle():
    """Verify that an indirect circular dependency cycle (A -> B -> C -> A) is caught."""
    tasks = [
        TaskItem(
            task_id="TSK-001",
            title="Task 1",
            description="First task",
            dependencies=["TSK-003"]
        ),
        TaskItem(
            task_id="TSK-002",
            title="Task 2",
            description="Second task",
            dependencies=["TSK-001"]
        ),
        TaskItem(
            task_id="TSK-003",
            title="Task 3",
            description="Third task",
            dependencies=["TSK-002"]
        )
    ]
    assert validate_dependencies(tasks) is False

def test_validate_dependencies_duplicate_ids():
    """Verify that tasks with duplicate IDs fail validation."""
    tasks = [
        TaskItem(
            task_id="TSK-001",
            title="Task 1",
            description="First task",
            dependencies=[]
        ),
        TaskItem(
            task_id="TSK-001",
            title="Task 2",
            description="Duplicate ID task",
            dependencies=[]
        )
    ]
    assert validate_dependencies(tasks) is False

def test_decomposer_node_integration():
    """Verify that decomposer_node correctly populates task_backlog and active_task_id."""
    initial_state = ProjectState(
        project_id="test_project",
        project_root="e:/TA",
        tech_stack={"db": "PostgreSQL"},
        locked_decisions=[],
        task_backlog=[],
        active_task_id=None,
        retrieved_code_context=[],
        retrieved_web_docs=[],
        generated_prompt_payload={"objective": "Build a state machine"},
        critic_iteration=0,
        critic_passed=False,
        critic_feedback=None
    )

    update = decomposer_node(initial_state)
    
    assert "task_backlog" in update
    assert "active_task_id" in update
    
    backlog = update["task_backlog"]
    assert len(backlog) > 0
    assert backlog[0].task_id == "TSK-001"
    assert update["active_task_id"] == "TSK-001"
    
    # Assert the mock generated output passes our validation
    assert validate_dependencies(backlog) is True

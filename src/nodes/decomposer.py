import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import litellm
from src.core.state import TaskItem, ProjectState
from src.core.config import settings

logger = logging.getLogger(__name__)

class TaskList(BaseModel):
    tasks: List[TaskItem]

def validate_dependencies(tasks: List[TaskItem]) -> bool:
    """
    Validates task lists for uniqueness and circular dependencies.
    Returns True if valid, False otherwise.
    """
    if not tasks:
        return True

    # 1. Check for unique task IDs
    task_ids = [t.task_id for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        logger.error("Validation failed: Duplicate task IDs found.")
        return False

    # 2. Build adjacency list of dependencies
    # A task A depends on B -> B must run before A.
    # Directed edge from B -> A.
    task_id_set = set(task_ids)
    adj = {t.task_id: [] for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            if dep in task_id_set:
                adj[dep].append(t.task_id)

    # 3. Depth-first search (DFS) cycle check
    # 0 = unvisited, 1 = visiting, 2 = visited
    visit_state = {t.task_id: 0 for t in tasks}

    def has_cycle(node: str) -> bool:
        visit_state[node] = 1  # visiting
        for neighbor in adj[node]:
            if visit_state[neighbor] == 1:
                return True
            if visit_state[neighbor] == 0:
                if has_cycle(neighbor):
                    return True
        visit_state[node] = 2  # visited
        return False

    for t in tasks:
        if visit_state[t.task_id] == 0:
            if has_cycle(t.task_id):
                logger.error(f"Circular dependency cycle detected starting at task {t.task_id}.")
                return False

    return True

def mock_decomposition(prompt: str) -> List[TaskItem]:
    """Generates a mock list of tasks for testing when no LLM API key is present."""
    logger.info("No active LLM API keys detected. Using offline mock decomposition.")
    return [
        TaskItem(
            task_id="TSK-001",
            title="Setup Foundation",
            description=f"Foundation task related to: {prompt}",
            status="COMPLETED",
            dependencies=[],
            target_files=["src/core/state.py"],
            acceptance_criteria=["State schema is defined and passes validation"]
        ),
        TaskItem(
            task_id="TSK-002",
            title="Implement Task Decomposer",
            description=f"Decomposer task related to: {prompt}",
            status="PENDING",
            dependencies=["TSK-001"],
            target_files=["src/nodes/decomposer.py"],
            acceptance_criteria=["Dependency validation checks cycles successfully"]
        )
    ]

def decompose_task_prompt(prompt: str, model_name: Optional[str] = None) -> List[TaskItem]:
    """
    Invokes LLM via litellm to decompose the developer prompt into TaskItems.
    Falls back to mock_decomposition if no API keys are found.
    """
    has_api_key = any(
        os.getenv(k) and os.getenv(k) != "your-api-key-here" for k in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY"
        ]
    )

    is_testing = "PYTEST_CURRENT_TEST" in os.environ
    if is_testing or not has_api_key:
        return mock_decomposition(prompt)

    system_instruction = (
        "You are an expert system architect. Decompose the user's software engineering prompt "
        "into a structured list of granular, dependency-ordered tasks conforming to the TaskItem schema. "
        "Each task must have a unique task_id, clear description, target files list, list of dependencies, "
        "and concrete acceptance criteria. Ensure there are absolutely NO circular dependencies."
    )

    try:
        target_model = model_name or settings.LITELLM_MODEL
        response = litellm.completion(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format=TaskList,
            temperature=0.1
        )
        content_str = response.choices[0].message.content
        data = json.loads(content_str)
        tasks = [TaskItem(**t) for t in data.get("tasks", [])]
        return tasks
    except Exception as e:
        logger.error(f"LLM decomposition failed: {e}. Falling back to mock decomposition.")
        return mock_decomposition(prompt)

def decomposer_node(state: ProjectState) -> dict:
    """
    LangGraph node that processes incoming prompt, decomposes it, validates it,
    and updates the state backlog.
    """
    prompt = state.get("generated_prompt_payload", {}).get("objective", "") if state.get("generated_prompt_payload") else ""
    if not prompt:
        # Fallback to general project objectives if prompt isn't in payload
        prompt = "Initialize and orchestrate project"

    tasks = decompose_task_prompt(prompt)
    
    if not validate_dependencies(tasks):
        # Fallback to mock tasks if generated tasks fail cyclic validation
        logger.warning("Generated task list was invalid (contains cycle or duplicates). Falling back.")
        tasks = mock_decomposition(prompt)

    return {
        "task_backlog": tasks,
        "active_task_id": tasks[0].task_id if tasks else None
    }

import os
import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.core.state import ProjectState
from src.nodes.critic import critic_node

logger = logging.getLogger(__name__)

def coder_node(state: ProjectState) -> dict:
    """
    Coder Node: Simulates generating/writing code.
    If it's the first iteration (critic_iteration == 0), writes placeholder code containing TODO.
    On subsequent iterations (critic_iteration > 0), writes clean, correct code.
    """
    active_task_id = state.get("active_task_id")
    backlog = state.get("task_backlog", [])
    
    active_task = None
    for task in backlog:
        t_id = task.task_id if hasattr(task, "task_id") else task.get("task_id")
        if t_id == active_task_id:
            active_task = task
            break

    if not active_task:
        logger.warning(f"Coder Node: No active task found for id: {active_task_id}")
        return {}

    target_files = active_task.target_files if hasattr(active_task, "target_files") else active_task.get("target_files", [])
    iteration = state.get("critic_iteration", 0)

    for path in target_files:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, "w", encoding="utf-8") as f:
            if iteration == 0:
                logger.info(f"Coder Node: Writing stub placeholder code to {path}")
                f.write("def execute_task():\n    # TODO: implement core task logic\n    pass\n")
            else:
                logger.info(f"Coder Node: Repairing and writing clean code to {path} (Iteration: {iteration})")
                f.write("def execute_task():\n    return 'success'\n")

    return {}

def route_critic(state: ProjectState) -> Literal["coder", "__end__"]:
    """Routes execution: cycles back to coder if critic failed, up to 3 times."""
    passed = state.get("critic_passed", False)
    iteration = state.get("critic_iteration", 0)

    if passed:
        logger.info("Critic Validation Passed. Terminating.")
        return "__end__"

    if iteration >= 3:
        logger.warning(f"Critic Validation Failed. Reached maximum iteration limit ({iteration}). Halting.")
        return "__end__"

    logger.info(f"Critic Validation Failed. Routing back to coder for repairs. (Iteration: {iteration})")
    return "coder"

# Compile cyclic feedback loop graph
workflow = StateGraph(ProjectState)
workflow.add_node("coder", coder_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "coder")
workflow.add_edge("coder", "critic")
workflow.add_conditional_edges(
    "critic",
    route_critic,
    {
        "coder": "coder",
        "__end__": END
    }
)

compiled_feedback_graph = workflow.compile()

from langgraph.graph import StateGraph, START, END
from src.core.state import ProjectState

def start_node(state: ProjectState) -> dict:
    """A minimal node that increments the critic iteration to verify state transition."""
    current_iter = state.get("critic_iteration", 0) or 0
    return {
        "critic_iteration": current_iter + 1,
        "active_task_id": "TSK-001-ACTIVE"
    }

# Create state graph workflow
workflow = StateGraph(ProjectState)
workflow.add_node("start_node", start_node)
workflow.add_edge(START, "start_node")
workflow.add_edge("start_node", END)

def get_compiled_graph(checkpointer):
    """Compiles the graph workflow with the given persistence checkpointer."""
    return workflow.compile(checkpointer=checkpointer)

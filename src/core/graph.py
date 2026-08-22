from langgraph.graph import StateGraph, START, END
from src.core.state import ProjectState
from src.nodes.perception import perception_node, route_perception
from src.nodes.decomposer import decomposer_node
from src.core.graph_feedback import coder_node, route_critic
from src.nodes.critic import critic_node
from src.nodes.chat import conversational_node

# Compile unified orchestration workflow
workflow = StateGraph(ProjectState)

# Add all process nodes
workflow.add_node("perception", perception_node)
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("coder", coder_node)
workflow.add_node("critic", critic_node)
workflow.add_node("conversational", conversational_node)

# START -> perception router
workflow.add_edge(START, "perception")

# Conditional perception routes
workflow.add_conditional_edges(
    "perception",
    route_perception,
    {
        "decomposer": "decomposer",
        "coder": "coder",
        "critic": "critic",
        "conversational": "conversational",
        "__end__": END
    }
)

# After task decomposing completes, terminate flow
workflow.add_edge("decomposer", END)
workflow.add_edge("conversational", END)

# Spec Generator always feeds into Critic review
workflow.add_edge("coder", "critic")

# Critic conditional routing (max retries repair loop)
workflow.add_conditional_edges(
    "critic",
    route_critic,
    {
        "coder": "coder",
        "__end__": END
    }
)

def get_compiled_graph(checkpointer=None):
    """Compiles the unified graph workflow with the given persistence checkpointer."""
    return workflow.compile(checkpointer=checkpointer)

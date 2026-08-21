import os
import logging
from typing import Optional
import litellm
from src.core.state import ProjectState
from src.core.config import settings

logger = logging.getLogger(__name__)

def classify_user_intent(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Classifies the user prompt into DECOMPOSE, RESEARCH, CODE, or CRITIC.
    Falls back to heuristic analysis if no API key is present.
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
        # Heuristic offline classification
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["research", "web", "search", "lookup", "docs", "scrape"]):
            return "RESEARCH"
        elif any(w in p_lower for w in ["code", "implement", "write", "build", "create"]):
            return "CODE"
        elif any(w in p_lower for w in ["critic", "verify", "validate", "check"]):
            return "CRITIC"
        else:
            return "DECOMPOSE"

    system_instruction = (
        "You are an routing classifier node. Your job is to classify the user's intent "
        "into exactly one of the following words: DECOMPOSE, RESEARCH, CODE, CRITIC.\n"
        "- DECOMPOSE: If the request is about decomposing a project, planning tasks, or setting up a DAG.\n"
        "- RESEARCH: If the request is primarily asking to search technical documentation or libraries.\n"
        "- CODE: If the request is directly asking to write code, edit files, or build a specific component.\n"
        "- CRITIC: If the request is about validating, verifying invariants, or reviewing existing code.\n"
        "Reply with ONLY the uppercase classification word."
    )

    try:
        target_model = model_name or settings.LITELLM_MODEL
        response = litellm.completion(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        ans = response.choices[0].message.content.strip().upper()
        if ans in ["DECOMPOSE", "RESEARCH", "CODE", "CRITIC"]:
            return ans
        return "DECOMPOSE"
    except Exception as e:
        logger.error(f"Intent classification LLM call failed: {e}. Using heuristics.")
        # Revert to heuristics on error
        return classify_user_intent(prompt, model_name="")

def perception_node(state: ProjectState) -> dict:
    """
    Graph node that evaluates the input query intent and updates the state payload.
    """
    objective = ""
    if state.get("generated_prompt_payload"):
        objective = state["generated_prompt_payload"].get("objective", "")

    if not objective:
        objective = "Decompose project instructions"

    intent = classify_user_intent(objective)
    
    payload = dict(state.get("generated_prompt_payload") or {})
    payload["intent"] = intent
    payload["objective"] = objective

    return {
        "generated_prompt_payload": payload
    }

def route_perception(state: ProjectState) -> str:
    """
    Conditional routing function for LangGraph.
    Determines if execution goes to the decomposer or halts.
    """
    payload = state.get("generated_prompt_payload") or {}
    intent = payload.get("intent", "DECOMPOSE")
    backlog = state.get("task_backlog") or []

    # If the backlog is empty or intent is explicitly DECOMPOSE, route to decomposer
    if not backlog or intent == "DECOMPOSE":
        return "decomposer"
        
    return "__end__"

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

def classify_confirmation(prompt: str) -> bool:
    """
    Uses the live LLM to dynamically determine if the user is confirming/allowing file writing.
    Falls back to a keyword heuristic if offline/no key is present.
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
        return prompt.lower().strip() in ["yes", "y", "proceed", "go ahead", "do it", "sure", "ok", "do"]

    system_instruction = (
        "You are a routing classifier. Your job is to determine if the user's message "
        "expresses confirmation, approval, or consent to proceed (e.g., saying 'yes', 'do it', 'go ahead', 'sure', 'proceed', 'looks good').\n"
        "Reply with 'YES' if they approve, or 'NO' if they refuse, ask a question, or say something else.\n"
        "Reply with ONLY the uppercase word YES or NO."
    )

    try:
        response = litellm.completion(
            model=settings.LITELLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        ans = response.choices[0].message.content.strip().upper()
        return ans == "YES"
    except Exception as e:
        logger.warning(f"Confirmation classification failed: {e}. Falling back to keywords.")
        return prompt.lower().strip() in ["yes", "y", "proceed", "go ahead", "do it", "sure", "ok", "do"]

def perception_node(state: ProjectState) -> dict:
    """
    Graph node that evaluates the input query intent, checks permission status,
    and updates state payload / conversational response.
    """
    objective = ""
    if state.get("generated_prompt_payload"):
        objective = state["generated_prompt_payload"].get("objective", "")

    if not objective:
        objective = "Decompose project instructions"

    was_waiting = state.get("conversational_response") is not None
    permission_granted = state.get("permission_granted", False)
    conversational_response = state.get("conversational_response")

    payload = dict(state.get("generated_prompt_payload") or {})

    if was_waiting:
        # Check if the user confirmed the action dynamically via LLM
        if classify_confirmation(objective):
            permission_granted = True
            conversational_response = None
            # Objective remains the previous coding objective (do not overwrite)
        else:
            # User refused/changed prompt, treat as new request
            permission_granted = False
            conversational_response = None
            intent = classify_user_intent(objective)
            payload["intent"] = intent
            payload["objective"] = objective
    else:
        # New request (preserve permission_granted if it was already True)
        permission_granted = state.get("permission_granted", False)
        conversational_response = None
        intent = classify_user_intent(objective)
        payload["intent"] = intent
        payload["objective"] = objective

    # Check permission gating for CODE intent (file writing)
    intent = payload.get("intent", "DECOMPOSE")
    if intent == "CODE" and not permission_granted:
        conversational_response = (
            f"I have analyzed your request: '{payload.get('objective')}'.\n\n"
            f"I am ready to generate the detailed Markdown blueprint specification files for your project.\n"
            f"Before writing files to your disk, I want to confirm your motives.\n\n"
            f"Would you like me to write the base specification files now? Reply 'yes' or 'proceed' to confirm."
        )
    else:
        conversational_response = None

    return {
        "generated_prompt_payload": payload,
        "conversational_response": conversational_response,
        "permission_granted": permission_granted
    }

def route_perception(state: ProjectState) -> str:
    """
    Conditional routing function for LangGraph.
    Determines if execution goes to the decomposer, coder (if permission is granted), or halts.
    """
    payload = state.get("generated_prompt_payload") or {}
    intent = payload.get("intent", "DECOMPOSE")
    backlog = state.get("task_backlog") or []

    # If the backlog is empty or intent is explicitly DECOMPOSE, route to decomposer
    if not backlog or intent == "DECOMPOSE":
        return "decomposer"
        
    if intent == "CODE":
        # Only route to coder if permission is granted by the user
        if state.get("permission_granted", False):
            return "coder"
        else:
            return "__end__"
        
    if intent == "CRITIC":
        return "critic"
        
    return "__end__"

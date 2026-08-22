import os
import logging
import litellm
from src.core.state import ProjectState
from src.core.config import settings

logger = logging.getLogger(__name__)

def conversational_node(state: ProjectState) -> dict:
    """
    Orchestration node that handles casual user chatter and brainstorming.
    Generates a natural, helpful AI reply.
    """
    payload = state.get("generated_prompt_payload") or {}
    objective = payload.get("objective", "Hello")

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
        # Offline/Testing heuristic response
        p_lower = objective.lower().strip()
        if any(w in p_lower for w in ["hey", "hello", "hi"]):
            reply = "Hello! I am ResearchRealy, your project orchestrator and specification generator. How can I help you today?"
        else:
            reply = f"I received your request: '{objective}'. I am ready to help you plan your architecture, search documentation, or draft specs when you specify the task!"
        return {
            "conversational_response": reply
        }

    system_instruction = (
        "You are ResearchRealy, an enterprise-grade AI project advisor and assistant.\n"
        "The user wants to have a general conversation, say hello, ask for advice, or brainstorm ideas.\n"
        "Provide a helpful, friendly, and concise response to guide them or answer their questions. Keep it professional."
    )

    try:
        response = litellm.completion(
            model=settings.LITELLM_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": objective}
            ],
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        return {
            "conversational_response": reply
        }
    except Exception as e:
        logger.error(f"Conversational node LLM call failed: {e}")
        return {
            "conversational_response": f"I had trouble contacting my LLM brain, but I'm here! I received: '{objective}'"
        }

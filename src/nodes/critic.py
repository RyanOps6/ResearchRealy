import os
import logging
from typing import List, Optional
from pydantic import BaseModel
import litellm
from src.core.config import settings
from src.core.state import ProjectState, TaskItem

logger = logging.getLogger(__name__)

class CriticEvaluation(BaseModel):
    """Pydantic structure for critic output."""
    is_approved: bool
    feedback: str
    issues: List[str]

def evaluate_task_output(
    task_description: str,
    code_content: str,
    model_name: Optional[str] = None
) -> CriticEvaluation:
    """
    Evaluates code output using LLM with structured outputs.
    Falls back to offline heuristic review if API keys are missing.
    """
    has_api_key = any(
        os.getenv(k) and os.getenv(k) != "your-api-key-here" for k in [
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY"
        ]
    )

    if not has_api_key:
        return run_heuristic_evaluation(code_content)

    system_instruction = (
        "You are an expert code reviewer and quality critic. Your job is to verify "
        "if the proposed code satisfies the task requirements and contains no placeholders, "
        "syntactic errors, or bugs. Respond strictly with JSON conforming to the CriticEvaluation schema."
    )
    user_prompt = (
        f"Task Description:\n{task_description}\n\n"
        f"Generated Code:\n{code_content}\n"
    )

    try:
        target_model = model_name or settings.LITELLM_MODEL
        response = litellm.completion(
            model=target_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format=CriticEvaluation,
            temperature=0.1
        )
        import json
        content_str = response.choices[0].message.content
        data = json.loads(content_str)
        return CriticEvaluation(**data)
    except Exception as e:
        logger.error(f"LLM Critic call failed: {e}. Falling back to heuristic evaluation.")
        return run_heuristic_evaluation(code_content)

def run_heuristic_evaluation(code_content: str) -> CriticEvaluation:
    """Heuristic rule-based fallback evaluation for offline use."""
    if not code_content.strip():
        return CriticEvaluation(
            is_approved=False,
            feedback="The target code content is completely empty.",
            issues=["Empty file content"]
        )

    issues = []
    # Scan for common placeholder keywords
    if "todo" in code_content.lower():
        issues.append("Found pending 'TODO' marker in code.")
    if "notimplementederror" in code_content.lower():
        issues.append("Found un-implemented block raising 'NotImplementedError'.")
    if "pass" in code_content:
        # Check if pass is used as placeholder inside a block
        # Simple line checks to avoid false positives in comments
        lines = code_content.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "pass":
                # Check context to verify if it represents a stubbed function
                issues.append(f"Found stubbed 'pass' placeholder on line {i+1}.")
                break

    if issues:
        return CriticEvaluation(
            is_approved=False,
            feedback=f"Heuristic audit rejected the code due to placeholder symbols: {', '.join(issues)}",
            issues=issues
        )
    
    return CriticEvaluation(
        is_approved=True,
        feedback="Heuristic audit passed successfully.",
        issues=[]
    )

def critic_node(state: ProjectState) -> dict:
    """LangGraph node representing the anti-hallucination critic validation."""
    active_task_id = state.get("active_task_id")
    backlog = state.get("task_backlog", [])
    
    # 1. Locate active task item
    active_task = None
    for task in backlog:
        t_id = task.task_id if hasattr(task, "task_id") else task.get("task_id")
        if t_id == active_task_id:
            active_task = task
            break

    if not active_task:
        logger.warning(f"No active task found for id: {active_task_id}")
        return {
            "critic_passed": True,
            "critic_feedback": "Skipped. No active task found."
        }

    # 2. Read properties
    target_files = active_task.target_files if hasattr(active_task, "target_files") else active_task.get("target_files", [])
    description = active_task.description if hasattr(active_task, "description") else active_task.get("description", "")
    
    # 3. Read target files content from disk
    code_content = ""
    for path in target_files:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code_content += f"\n--- File: {path} ---\n" + f.read()
            except Exception as e:
                logger.error(f"Failed to read file {path} for validation: {e}")

    # 4. Evaluate
    eval_result = evaluate_task_output(description, code_content)
    new_iteration = state.get("critic_iteration", 0) + 1

    # 5. Update task backlog status
    updated_backlog = []
    for task in backlog:
        t_id = task.task_id if hasattr(task, "task_id") else task.get("task_id")
        if t_id == active_task_id:
            status = "COMPLETED" if eval_result.is_approved else "IN_PROGRESS"
            if hasattr(task, "status"):
                task.status = status
            else:
                task["status"] = status
        updated_backlog.append(task)

    return {
        "task_backlog": updated_backlog,
        "critic_passed": eval_result.is_approved,
        "critic_feedback": eval_result.feedback if not eval_result.is_approved else None,
        "critic_iteration": new_iteration
    }

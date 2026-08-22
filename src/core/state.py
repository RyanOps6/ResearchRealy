from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class CodeReference(BaseModel):
    file_path: str
    symbol_name: Optional[str] = None
    start_line: int
    end_line: int
    code_snippet: str

class TaskItem(BaseModel):
    task_id: str
    title: str
    description: str
    status: Literal["PENDING", "IN_PROGRESS", "BLOCKED", "COMPLETED"] = "PENDING"
    dependencies: List[str] = Field(default_factory=list)
    target_files: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)

class ProjectState(TypedDict):
    project_id: str
    project_root: str
    tech_stack: Dict[str, str]  # e.g. {"framework": "FastAPI", "db": "PostgreSQL 16"}
    locked_decisions: List[str]  # Invariants that cannot be violated
    task_backlog: List[TaskItem]
    active_task_id: Optional[str]
    retrieved_code_context: List[CodeReference]
    retrieved_web_docs: List[Dict[str, str]]
    generated_prompt_payload: Optional[Dict[str, Any]]
    critic_iteration: int
    critic_passed: bool
    critic_feedback: Optional[str]
    permission_granted: bool
    conversational_response: Optional[str]
    chat_history: List[Dict[str, str]]

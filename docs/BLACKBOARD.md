# Project Blackboard & Inter-Agent Coordination

## 1. Active File Locks & Assignments
| File Path | Locked By (Agent ID / Task ID) | Action / Purpose | Status |
| :--- | :--- | :--- | :--- |
| `src/auth/service.py` | `Agent-Backend (TSK-002)` | Implementing JWT token rotation | **LOCKED** |

---

## 2. Change Event Stream (Append-Only Log)
> *Every agent MUST append its change summary here immediately after making edits.*

### [2026-08-19 11:34] — Task: TSK-004 (CLI Interface & Resume Engine)
- **Agent:** `Antigravity (Lead Architect & Core Engineer)`
- **Files Created:**
  - `src/main.py` (Implemented CLI runner, compiled workflow linking START ➔ perception ➔ route_perception ➔ decomposer ➔ END, loaded .env config, and handled Windows SelectorEventLoop policy)
  - `tests/test_cli.py` (Implemented unit tests for argument parsing, MemorySaver dry runs, and PostgreSQL workflow run & resumption)
- **Exported Symbols / Interfaces:**
  - `main()` -> argparse CLI entry point
  - `run_workflow(prompt, thread_id)` -> executes new orchestrator workflow
  - `resume_workflow(thread_id)` -> resumes session state from database
- **Important Notes:**
  - Modified `src/nodes/decomposer.py` and `src/nodes/perception.py` to recognize and ignore placeholder credentials (`"your-api-key-here"`) in `.env` to prevent test failures offline.

### [2026-08-19 09:48] — Task: TSK-003 (Perception Node / Intent Router)
- **Agent:** `Antigravity (Lead Architect & Core Engineer)`
- **Files Created:**
  - `src/nodes/perception.py` (Implemented user intent classification with litellm and heuristic search fallback, and LangGraph routing function `route_perception`)
  - `tests/test_perception.py` (Implemented unit tests for classification mappings, integration with ProjectState, and conditional edge routing outcomes)
- **Exported Symbols / Interfaces:**
  - `perception_node(state)` -> update prompt payload in ProjectState
  - `classify_user_intent(prompt)` -> classified uppercase string
  - `route_perception(state)` -> conditional edge routing name ("decomposer" or "__end__")

### [2026-08-19 09:15] — Task: TSK-002 (Task Decomposer Node)
- **Agent:** `Antigravity (Lead Architect & Core Engineer)`
- **Files Created:**
  - `src/nodes/decomposer.py` (Implemented LLM task planner using litellm and DFS DAG cycle detection validation)
  - `tests/test_decomposer.py` (Implemented unit tests for valid DAGs, cyclic dependencies, duplicates, and state graph node integration)
- **Exported Symbols / Interfaces:**
  - `decomposer_node(state)` -> LangGraph state dictionary update
  - `validate_dependencies(tasks)` -> boolean (checks for duplicates and cycles)
- **Important Notes:**
  - Added `litellm` dependency to `requirements.txt` and installed it in the virtual environment.

### [2026-08-19 08:30] — Task: TSK-001 (Core State & Persistence Engine)
- **Agent:** `Antigravity (Lead Architect & Core Engineer)`
- **Files Created:**
  - `src/core/state.py` (Defined CodeReference, TaskItem, and ProjectState TypedDict schemas)
  - `src/core/config.py` (Created settings model using Pydantic Settings for validating POSTGRES_URI)
  - `src/db/session.py` (Implemented get_checkpointer async context manager for AsyncPostgresSaver connection pool & migrations setup)
  - `src/core/graph.py` (Set up minimal compiled LangGraph with postgres checkpointer)
  - `tests/test_foundation.py` (Implemented state schema validation, offline MemorySaver tests, and Windows-asyncio-compatible PostgreSQL integration tests)
- **Exported Symbols / Interfaces:**
  - `get_checkpointer()` -> yields `AsyncPostgresSaver`
  - `get_compiled_graph(checkpointer)` -> compiled LangGraph executable
- **Important Notes:**
  - Standard Windows ProactorEventLoop is incompatible with psycopg. In `test_foundation.py`, SelectorEventLoopPolicy is set to bypass this constraint on Windows systems.

### [2026-08-17 19:30] — Task: Pre-setup PostgreSQL connection sample
- **Agent:** `Agent-DataEngine`
- **Files Modified:**
  - `src/db/session.py` (Created async engine using `asyncpg`)
  - `src/core/config.py` (Added `POSTGRES_URI` environment validation)
- **Exported Symbols / Interfaces:**
  - `get_db_session()` -> yields `AsyncSession`
- **Important Notes for Next Agent:**
  - Database pool size is capped at 20 connections. Always use `async with get_db_session() as session:`.

---

## 3. Global Shared State & Discovered Gotchas
- **Known Issue:** Redis client version requires `decode_responses=True` when initializing the connection pool in `src/core/redis.py`.
- **API Spec Lock:** All API route endpoints must return snake_case JSON schemas via Pydantic models.
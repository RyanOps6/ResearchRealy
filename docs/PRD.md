# Product Requirements Document (PRD): Enterprise Multi-Agent Project Orchestrator ("Master Brain")

---

## 1. Executive Summary & Problem Definition

The **Master Brain AI Orchestrator** is an enterprise-grade, state-driven multi-agent system designed to act as the primary architect, task decomposer, and quality gate for complex software engineering projects.

### The Failure of Standard Chat-Based AI

* **Context Truncation & Forgetting:** Large language models in continuous chat threads exceed their practical attention windows, leading to context eviction. Early architectural constraints, database schemas, and configuration decisions are dropped from active memory.
* **Attention Dilution & Hallucination:** Outdated, intermediate, or incorrect code snippets in a long transcript pollute the context. The model begins inventing non-existent package methods, deprecated syntax, and imaginary file paths.
* **Lack of Ground-Truth Grounding:** Standard models guess codebase structures rather than parsing actual disk files, and rely on static training cutoffs rather than live package documentation.

### The Master Brain Solution

This system replaces linear chat transcripts with an **event-driven state graph**, **tri-tier persistent memory**, **AST-driven hybrid Code RAG**, **live web intelligence**, and a **dual-pass Critic-Verifier loop** to guarantee zero context loss and zero-regression prompt specification generation.

---

## 2. Core Key Performance Indicators (KPIs)

| Metric | Target SLA | Measurement & Verification Method |
| --- | --- | --- |
| **Hallucination / Factuality Error Rate** | $\le 0.1\%$ | Critic-node validation of prompt recipes against local AST structures and scraped online API schemas. |
| **Context Retention Across Sessions** | $100\%$ ($0\%$ drift) | Graph state checkpointing stored in PostgreSQL tables (`checkpoints`, `writes`). |
| **Code Retrieval Accuracy (NDCG@10)** | $\ge 0.94$ | Hybrid search: Tree-sitter AST chunks + BM25 keyword matching + Dense embeddings via Qdrant. |
| **External Documentation Recency** | Real-time ($\le 24$h) | Tavily / Exa search integration with version-pinned documentation scrapers. |
| **Source Code Protection / Safety** | $100\%$ | Guaranteeing zero lines of unauthorized code are written directly to production source files. |
| **Downstream Schema Adherence** | $100\%$ | Pydantic validation on all task delegation blueprints and prompt recipe outputs. |

---

## 3. High-Level System Architecture

```
                               ┌─────────────────────────────────────────┐
                               │         Developer / CLI Client          │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       LangGraph Orchestration Core                                      │
│                                                                                                        │
│   ┌──────────────────────┐        ┌─────────────────────────┐        ┌─────────────────────────────┐   │
│   │   Perception Node    │ ─────► │  Task Decomposer Node   │ ─────► │   Prompt Generator Node     │   │
│   │   (Intent Router)    │        │  (Milestone / DAG Spec) │        │   (Downstream Delegator)    │   │
│   └──────────┬───────────┘        └────────────┬────────────┘        └──────────────┬──────────────┘   │
│              │                                 │                                    │                  │
│              ▼                                 ▼                                    ▼                  │
│   ┌──────────────────────┐        ┌─────────────────────────┐        ┌─────────────────────────────┐   │
│   │  Web Research Tool   │        │   Codebase RAG Tool     │        │   Critic / Verifier Node    │   │
│   │  (Tavily / Exa API)  │        │ (Tree-sitter + Qdrant)  │        │ (AST Check + Invariant Gate)│   │
│   └──────────────────────┘        └─────────────────────────┘        └──────────────┬──────────────┘   │
│                                                                                     │                  │
│                                    ◄── Feedback Loop (Auto-Fix on Failure) ─────────┘                  │
└───────────────────────────────────────────┬────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌───────────────────────────────────────────────────────────────────────────┐ ┌──────────────────────────┐
│                         Tri-Tier Persistent Storage                       │ │  Downstream Coding AI    │
│ ┌──────────────────────┬─────────────────────────┬──────────────────────┐ │ │ (Target: Claude Code,   │
│ │    Working Memory    │     Episodic Memory     │    Semantic Memory   │ │ │  Cursor, Windsurf,     │
│ │ (Redis Task Buffer)  │ (Postgres Checkpointer) │ (Qdrant Vector DB)   │ │ │  Aider, Copilot)         │
│ └──────────────────────┴─────────────────────────┴──────────────────────┘ │ └──────────────────────────┘
└───────────────────────────────────────────────────────────────────────────┘

```

---

## 4. Subsystem Specifications

### 4.1 LangGraph State Machine & Schema Definition

The state machine is deterministic, strongly typed, and persists every transition to PostgreSQL using `AsyncPostgresSaver`.

```python
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
    tech_stack: Dict[str, str] # e.g. {"framework": "FastAPI", "db": "PostgreSQL 16", "orm": "SQLAlchemy 2.0"}
    locked_decisions: List[str] # Invariants that cannot be violated
    task_backlog: List[TaskItem]
    active_task_id: Optional[str]
    retrieved_code_context: List[CodeReference]
    retrieved_web_docs: List[Dict[str, str]]
    generated_prompt_payload: Optional[Dict[str, Any]]
    critic_iteration: int
    critic_passed: bool
    critic_feedback: Optional[str]

```

---

### 4.2 Tri-Tier Memory Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. WORKING MEMORY (Redis 7.x)                                                          │
│    - Active session ID, ephemeral intermediate scratchpad, current node execution trace│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. EPISODIC MEMORY (PostgreSQL 16 + AsyncPostgresSaver + JSONB)                        │
│    - Checkpoints: Every graph execution snapshot is stored immutably.                 │
│    - Architectural Decision Records (ADRs): "Why we chose Redis over Memcached".      │
│    - Bug Resolution Log: Stack traces, root causes, and verified fix patches.          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. SEMANTIC & KNOWLEDGE MEMORY (Qdrant Vector DB)                                      │
│    - AST-indexed local codebase embeddings + external framework documentation.         │
└────────────────────────────────────────────────────────────────────────────────────────┘

```

#### Checkpointing Implementation:

```python
import os
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/agent_master_db")

async def init_checkpointer():
    """Initializes the PostgreSQL connection and runs automatic migrations."""
    checkpointer = AsyncPostgresSaver.from_conn_string(DB_URI)
    await checkpointer.setup()  # Provisions checkpoints, checkpoint_blobs, and writes tables
    return checkpointer

```

---

### 4.3 Code-Aware Hybrid RAG Engine (Tree-sitter + Qdrant)

Standard character splitting breaks code logic. The RAG pipeline parses code using Abstract Syntax Trees (AST):

```
  Source Code (.py / .ts / .go / .rs)
                 │
                 ▼
     [ Tree-sitter AST Parser ]
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
Class & Method Blocks     Imports & Type Signatures
  │                             │
  └──────────────┬──────────────┘
                 ▼
       [ Structural Metadata ] (file_path, parent_class, line_range, symbols)
                 │
  ┌──────────────┴──────────────┐
  ▼                             ▼
Dense Vectors (Qdrant)       BM25 Sparse Lexical Index
  │                             │
  └──────────────┴──────────────┘
                 ▼
    [ Reciprocal Rank Fusion (RRF) ]
                 │
                 ▼
    High-Precision Code Context

```

* **AST Parsing:** Tree-sitter extracts complete function blocks, class declarations, docstrings, and imports. Large functions ($>1000$ tokens) are recursively subdivided by logical code blocks (e.g., control flows, loops).
* **Hybrid Search Formula:**

$$\text{RRF\_Score}(d) = \frac{1}{60 + \text{Rank}_{\text{Dense}}(d)} + \frac{1}{60 + \text{Rank}_{\text{BM25}}(d)}$$


* **File Watcher:** Incremental updates via `watchfiles` compute SHA-256 hashes per file and update only modified AST nodes in Qdrant.

---

### 4.4 Real-Time Web Intelligence Layer

To eliminate outdated syntax:

1. **Search Tool:** Integration with Tavily Search API / Exa AI configured to query technical documentation domains (`docs.*`, `[github.com/*/releases](https://github.com/*/releases)`, `pypi.org`, `npmjs.com`).
2. **Version Pinning Guard:** Every third-party library reference must match the version defined in `tech_stack`. If an agent proposes a method, the search tool verifies its deprecation status in the official documentation.
3. **Headless Scraper:** Playwright fallback for single-page documentation applications (SPAs) where static HTML scraping yields empty content.

---

### 4.5 Anti-Hallucination Critic-Verifier Engine

Before output is finalized, it must pass a 4-point verification check:

```
[ Generator Draft Payload ]
             │
             ▼
┌────────────────────────────────────────────────────────────┐
│                    Critic / Verifier Node                  │
│                                                            │
│ 1. File & Symbol Gate: Do referenced files & symbols exist?│
│ 2. Invariant Gate: Does draft violate locked decisions?    │
│ 3. Deprecation Gate: Are imports valid in target version?  │
│ 4. Scope Gate: Does output address the assigned task ID?   │
└─────────────┬──────────────────────────────────────────────┘
              │
      ┌───────┴────────┐
      │                │
  [ Failed ]       [ Passed ]
      │                │
      ▼                ▼
Trigger Feedback    Approve & Emit
Loop (Max 3x)       Output Payload

```

If validation fails, the Critic generates structured feedback detailing the missing sections or placeholder text to trigger a spec-repair loop.

---

### 4.6 Downstream Delegation Schema (Target Agent Output)

The final output is a structured Markdown blueprint document designed for copying and pasting directly into downstream coding AIs (like Cursor, Claude Code, Windsurf, or Copilot Chat):

# 📋 Copilot/ChatGPT Prompt Recipe: JWT Authentication Handler

> **Target File:** src/auth.py
> **Dependencies:** PyJWT==2.8.0

## 💡 The Rough Idea
We want to create a secure token manager that issues temporary access keys for user sessions. When a user logs in, we take their unique user ID, wrap it in a payload dictionary with a 60-minute expiration time, and sign it using a secure secret key and the HS256 algorithm. The function should return this signature string directly so that our backend can attach it to response headers. We must also include error catching to intercept any encryption failures and raise standard unauthorized HTTP exceptions.

---

Copy and paste the prompt block below into your AI coding assistant:

```prompt
Act as an expert Python developer. Write the code for src/auth.py based on the following requirements:

1. Import:
   - import jwt
   - from datetime import datetime, timezone

2. Implement a function:
   - Name: generate_access_token
   - Parameters: user_id: str, expires_in_minutes: int = 60
   - Returns: str (the encoded JWT token)

3. Logic:
   - Create a payload dictionary containing sub, exp, and iat keys.
   - Encode the payload using jwt.encode() with your secret key, using algorithm "HS256".

4. Error Handling:
   - Wrap the encoding in a try-except block and raise HTTP exception on failure.
```

---

## 5. Phased Vertical-Slice Implementation & Manual Testing Matrix

Every phase delivers a runnable, testable slice of the architecture:

| Phase | Core Deliverable | Technologies | How to Manually Verify (Test Protocol) |
| --- | --- | --- | --- |
| **Phase 1: Foundation** | Persistent State Graph + Task Decomposer + CLI | `langgraph`, `langgraph-checkpoint-postgres`, `psycopg[binary]`, `pydantic` | **Test 1:** Run task prompt via CLI. Kill the terminal mid-execution. Restart and run `"continue"`. Verify it recovers exact state from Postgres tables (`checkpoints`, `writes`). |
| **Phase 2: Code RAG** | AST Chunker + Hybrid Inverted/Vector Search | `tree-sitter`, `tree-sitter-languages`, `qdrant-client`, `rank-bm25` | **Test 2:** Inject a custom buggy function into a local file. Ask the agent to resolve it. Verify it retrieves exact AST line ranges from disk rather than inventing signatures. |
| **Phase 3: Web Search** | Real-Time Docs Search & Version Resolver | `tavily-python`, `playwright`, `httpx` | **Test 3:** Prompt for syntax on a library that deprecated functions in recent versions. Verify it queries Tavily and outputs latest syntax. |
| **Phase 4: Critic Loop** | Anti-Hallucination Validation Gate & Feedback | `langgraph` (cyclic edges), `pydantic` | **Test 4:** Set a locked rule (e.g., *"No raw SQL"*). Prompt it to generate raw SQL. Verify the Critic catches the violation and forces regeneration. |
| **Phase 5: Production** | Observability, Tracing, Security Sandbox | `langfuse`, `fastapi`, `redis`, `docker` | **Test 5:** View full execution traces, latency, and token costs in the Langfuse dashboard. Verify path traversal security checks block `../../` file reads. |

---

## 6. Infrastructure & Deployment Specification

### Docker Compose Stack (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: master_brain_postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgrespassword
      POSTGRES_DB: agent_master_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:v1.10.0
    container_name: master_brain_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    container_name: master_brain_redis
    ports:
      - "6379:6379"

volumes:
  postgres_data:
  qdrant_data:

```

---

## 7. Security, Sandboxing & Observability

1. **Path Traversal Sandboxing:** All file access is validated against `os.path.realpath(project_root)`. Any operation attempting to read or write outside the repository root throws a hard security exception.
2. **Secret Scrubbing:** Regex filters intercept all state serializations and vector payloads to mask API keys, SSH keys, passwords, and `.env` variables before storing them in PostgreSQL or Qdrant.
3. **Observability & Tracing:** Integrated via **Langfuse** callbacks to trace every LLM request, tool execution latency, vector search similarity score, and Critic retry count.
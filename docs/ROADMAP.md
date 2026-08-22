# Master Brain Project Roadmap

This document outlines the phased, dependency-ordered engineering roadmap for the Enterprise Multi-Agent Project Orchestrator ("Master Brain"). 
We follow a strict **"Test-As-You-Build / Walking Skeleton"** philosophy, ensuring every single phase yields a runnable, verifiable slice of the system.

---

## Phase 1: Foundation (State Graph, Persistence, and CLI)
Deliverable: A persistent state machine using LangGraph and PostgreSQL checkpointer, exposed via a CLI.

### [x] TSK-001: Core State & Persistence Engine
- **Target Files:**
  - [NEW] [`docker-compose.yml`](file:///e:/TA/docker-compose.yml)
  - [NEW] [`requirements.txt`](file:///e:/TA/requirements.txt)
  - [NEW] [`src/core/config.py`](file:///e:/TA/src/core/config.py)
  - [NEW] [`src/core/state.py`](file:///e:/TA/src/core/state.py)
  - [NEW] [`src/db/session.py`](file:///e:/TA/src/db/session.py)
  - [NEW] [`src/core/graph.py`](file:///e:/TA/src/core/graph.py)
  - [NEW] [`tests/test_foundation.py`](file:///e:/TA/tests/test_foundation.py)
- **Acceptance Criteria:**
  - Define `ProjectState`, `TaskItem`, and `CodeReference` schemas.
  - Establish connection to PostgreSQL via `AsyncPostgresSaver.from_conn_string`.
  - Compile and run a minimal LangGraph using the checkpointer.
  - Verify that interrupting and resuming execution restores exact state from database.
- **Verification Commands:**
  - Start database: `docker compose up -d postgres`
  - Run pytest: `pytest tests/test_foundation.py -v`

### [x] TSK-002: Task Decomposer Node
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/nodes/decomposer.py`](file:///e:/TA/src/nodes/decomposer.py)
  - [NEW] [`tests/test_decomposer.py`](file:///e:/TA/tests/test_decomposer.py)
- **Acceptance Criteria:**
  - Implement LLM prompt structured to output a detailed DAG of tasks conforming to `TaskItem` schemas.
  - Support validation of dependencies to ensure no cycles are created.
- **Verification Commands:**
  - `pytest tests/test_decomposer.py -v`

### [x] TSK-003: Perception Node (Intent Router)
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/nodes/perception.py`](file:///e:/TA/src/nodes/perception.py)
  - [NEW] [`tests/test_perception.py`](file:///e:/TA/tests/test_perception.py)
- **Acceptance Criteria:**
  - Router parsing incoming developer intents (e.g., Code RAG search, Web Research, validation critic request) and directing the LangGraph execution flow.
- **Verification Commands:**
  - `pytest tests/test_perception.py -v`

### [x] TSK-004: CLI Interface & Session Resume Engine
- **Dependency:** `TSK-002`, `TSK-003`
- **Target Files:**
  - [NEW] [`src/main.py`](file:///e:/TA/src/main.py)
  - [NEW] [`tests/test_cli.py`](file:///e:/TA/tests/test_cli.py)
- **Acceptance Criteria:**
  - Build executable CLI using `argparse` or `typer`.
  - Support command: `python -m src.main run --prompt "Build login endpoint"`
  - Support command: `python -m src.main continue --thread-id <id>` (recovers exact state after session interruption).
- **Verification Commands:**
  - Run CLI: `python -m src.main run --prompt "Verify CLI foundation works"`
  - Run pytest: `pytest tests/test_cli.py -v`

---

## Phase 2: AST-Driven Code RAG Engine
Deliverable: Local repository indexing and hybrid search (dense embeddings + sparse lexical matching) powered by Qdrant.

### [x] TSK-005: Tree-sitter AST Parser
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/rag/ast_parser.py`](file:///e:/TA/src/rag/ast_parser.py)
  - [NEW] [`tests/test_ast.py`](file:///e:/TA/tests/test_ast.py)
- **Acceptance Criteria:**
  - Parse Python source code into AST. Extract class names, method signatures, line ranges, and imports.
  - Chunk functions/classes while maintaining parent metadata contexts.
- **Verification Commands:**
  - `pytest tests/test_ast.py -v`

### [x] TSK-006: Dual Indexing (Dense & Sparse Indexes)
- **Dependency:** `TSK-005`
- **Target Files:**
  - [NEW] [`src/rag/indexer.py`](file:///e:/TA/src/rag/indexer.py)
  - [NEW] [`tests/test_indexer.py`](file:///e:/TA/tests/test_indexer.py)
- **Acceptance Criteria:**
  - Connect to Qdrant vector database container.
  - Index code chunks as dense vectors (via sentence-transformers/openai).
  - Index code chunks in a BM25 sparse index (`rank-bm25`).
- **Verification Commands:**
  - Start Qdrant: `docker compose up -d qdrant`
  - `pytest tests/test_indexer.py -v`

### [x] TSK-007: Hybrid Search (RRF Retrieval)
- **Dependency:** `TSK-006`
- **Target Files:**
  - [NEW] [`src/rag/searcher.py`](file:///e:/TA/src/rag/searcher.py)
  - [NEW] [`tests/test_searcher.py`](file:///e:/TA/tests/test_searcher.py)
- **Acceptance Criteria:**
  - Implement Reciprocal Rank Fusion (RRF) formula combining dense and BM25 results.
  - Apply cross-encoder re-ranking model to filter the top retrieved context results.
- **Verification Commands:**
  - `pytest tests/test_searcher.py -v`

### [x] TSK-008: Codebase Watcher for Incremental Updates
- **Dependency:** `TSK-007`
- **Target Files:**
  - [NEW] [`src/rag/watcher.py`](file:///e:/TA/src/rag/watcher.py)
  - [NEW] [`tests/test_watcher.py`](file:///e:/TA/tests/test_watcher.py)
- **Acceptance Criteria:**
  - Track codebase changes using `watchfiles`.
  - Calculate SHA-256 hashes of files and trigger incremental AST re-indexing on change.
- **Verification Commands:**
  - `pytest tests/test_watcher.py -v`

---

## Phase 3: Web Research & Scraper Layer
Deliverable: Live technical documentation lookup and headless SPA scraping to ensure package version safety.

### [x] TSK-009: Tavily/Exa Documentation Resolver & Version Pinning
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/web/search.py`](file:///e:/TA/src/web/search.py)
  - [NEW] [`tests/test_web_search.py`](file:///e:/TA/tests/test_web_search.py)
- **Acceptance Criteria:**
  - Integrate Tavily / Exa search API.
  - Filter queries to strict technical domains (`docs.*`, releases, PyPI, npm).
  - Verify if returned libraries match exact version definitions in `tech_stack`.
- **Verification Commands:**
  - `pytest tests/test_web_search.py -v`

### [x] TSK-010: Headless Scraper for SPA Docs
- **Dependency:** `TSK-009`
- **Target Files:**
  - [NEW] [`src/web/scraper.py`](file:///e:/TA/src/web/scraper.py)
  - [NEW] [`tests/test_scraper.py`](file:///e:/TA/tests/test_scraper.py)
- **Acceptance Criteria:**
  - Spin up Playwright headless browser to scrape dynamic documentation content where standard HTTP requests return empty pages.
- **Verification Commands:**
  - `pytest tests/test_scraper.py -v`

---

## Phase 4: Anti-Hallucination Critic-Verifier Engine
Deliverable: A cyclic feedback loop validating code drafts against rules and database constraints before finalizing output.

### [x] TSK-011: Critic Validation Node
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/nodes/critic.py`](file:///e:/TA/src/nodes/critic.py)
  - [NEW] [`tests/test_critic.py`](file:///e:/TA/tests/test_critic.py)
- **Acceptance Criteria:**
  - Parse generated prompt specification drafts.
  - Implement a 4-point verification check: structural spec layout (Rough Idea & Prompt Recipe), target file AST consistency, version pinning, and placeholder checks.
- **Verification Commands:**
  - `pytest tests/test_critic.py -v`

### [x] TSK-012: Cyclic Graph Feedback Loop
- **Dependency:** `TSK-004`, `TSK-011`
- **Target Files:**
  - [NEW] [`src/core/graph_feedback.py`](file:///e:/TA/src/core/graph_feedback.py)
  - [NEW] [`tests/test_feedback.py`](file:///e:/TA/tests/test_feedback.py)
- **Acceptance Criteria:**
  - Configure LangGraph cyclic edges to route back to Spec Generator Node on Critic validation failures (max 3 retries).
- **Verification Commands:**
  - `pytest tests/test_feedback.py -v`

---

## Phase 5: Productionization, Security & Observability
Deliverable: Fully sandboxed execution, regex secret scrubbers, and live tracing dashboard integration.

### [x] TSK-013: Sandboxing, Secret Scrubbing & Tracing
- **Dependency:** `TSK-001`
- **Target Files:**
  - [NEW] [`src/core/security.py`](file:///e:/TA/src/core/security.py)
  - [NEW] [`src/core/tracing.py`](file:///e:/TA/src/core/tracing.py)
  - [NEW] [`tests/test_security.py`](file:///e:/TA/tests/test_security.py)
- **Acceptance Criteria:**
  - Validate all local file accesses against `project_root` real path (throwing SecurityException on path traversal attacks).
  - Use regex filters to scrub API keys, SSH tokens, and database passwords from logs/checkpoints.
  - Register Langfuse callback handler for execution observability.
- **Verification Commands:**
  - `pytest tests/test_security.py -v`

### [x] TSK-014: Full Orchestration Compose Stack
- **Dependency:** `TSK-008`, `TSK-010`, `TSK-012`, `TSK-013`
- **Target Files:**
  - [NEW] [`tests/test_integration.py`](file:///e:/TA/tests/test_integration.py)
- **Acceptance Criteria:**
  - Run the entire multi-agent orchestrator stack containing all perception, decomposer, research, RAG, critic, and database Saver services.
- **Verification Commands:**
  - `docker compose up -d`
  - `pytest tests/test_integration.py -v`

### [x] TSK-015: Interactive Conversational Permission Gate
- **Dependency:** `TSK-014`
- **Target Files:**
  - [MODIFY] [`src/core/state.py`](file:///e:/TA/src/core/state.py)
  - [MODIFY] [`src/nodes/perception.py`](file:///e:/TA/src/nodes/perception.py)
  - [MODIFY] [`src/main.py`](file:///e:/TA/src/main.py)
  - [MODIFY] [`chat.py`](file:///e:/TA/chat.py)
  - [NEW] [`tests/test_permission_gate.py`](file:///e:/TA/tests/test_permission_gate.py)
- **Acceptance Criteria:**
  - Block file writing operations (`CODE` intent) if `permission_granted` is `False`.
  - Prompt user conversationally with a permission request to write base files to disk.
  - Automatically evaluate user input and grant permission when `"yes"` or `"proceed"` is detected.
- **Verification Commands:**
  - `pytest tests/test_permission_gate.py -v`

### [x] TSK-016: Conversational Chat Node (CHAT Intent)
- **Dependency:** `TSK-015`
- **Target Files:**
  - [MODIFY] [`src/nodes/perception.py`](file:///e:/TA/src/nodes/perception.py)
  - [NEW] [`src/nodes/chat.py`](file:///e:/TA/src/nodes/chat.py)
  - [MODIFY] [`src/core/graph.py`](file:///e:/TA/src/core/graph.py)
  - [NEW] [`tests/test_conversation.py`](file:///e:/TA/tests/test_conversation.py)
- **Acceptance Criteria:**
  - Implement a dedicated `"CHAT"` intent that intercepts greetings, open-ended brainstorming, and advice requests.
  - Route execution to a new `"conversational"` node when `CHAT` intent is classified, bypassing task plans and file writes.
  - Formulate natural conversational assistant replies inside the state machine and output them to the client.
- **Verification Commands:**
  - `pytest tests/test_conversation.py -v`

### [x] TSK-017: Conversational Memory & Search Cache
- **Dependency:** `TSK-016`
- **Target Files:**
  - [MODIFY] [`src/core/state.py`](file:///e:/TA/src/core/state.py)
  - [MODIFY] [`chat.py`](file:///e:/TA/chat.py)
  - [MODIFY] [`src/nodes/perception.py`](file:///e:/TA/src/nodes/perception.py)
  - [NEW] [`src/nodes/research_node.py`](file:///e:/TA/src/nodes/research_node.py)
  - [MODIFY] [`src/nodes/chat.py`](file:///e:/TA/src/nodes/chat.py)
  - [MODIFY] [`src/core/graph.py`](file:///e:/TA/src/core/graph.py)
  - [NEW] [`tests/test_conversational_memory.py`](file:///e:/TA/tests/test_conversational_memory.py)
- **Acceptance Criteria:**
  - Track multi-turn conversational message threads inside `chat_history` database checkpointer state.
  - Dynamically evaluate permission approvals using the live LLM, removing all hardcoded confirmation checks from the client.
  - Route research queries (`RESEARCH` intent) to a new `research_node` that queries the web and caches JSON logs persistently to `docs/research_history.json`.
  - Synthesize and output the research findings directly within the assistant's conversational replies.
- **Verification Commands:**
  - `pytest tests/test_conversational_memory.py -v`

# Master Brain AI Orchestrator: Starting Point

This document provides a high-level overview of the system we are building, its core architecture, and the protocol for manual verification.

---

## 1. What We Are Creating & How It Works
The **Master Brain AI Orchestrator** is a state-driven, multi-agent system designed to orchestrate and delegate software engineering tasks to downstream coding agents (like Claude Code, Aider, or Cursor) without context loss or hallucination.

Instead of a simple chatbot that easily forgets details, the system uses a strongly typed state machine (built with **LangGraph**) and a tri-tier memory architecture to execute tasks:

1. **Perception (Intent Router):** Parses your request to determine if it requires research, codebase editing, or validation.
2. **Task Decomposer:** Automatically splits complex requests into a dependency-ordered list (DAG) of sub-tasks.
3. **AST-Driven Code RAG & Web Search:** Gathers local code snippets using a smart parser (Tree-sitter) and queries real-time web documents (Tavily) to verify libraries.
4. **Anti-Hallucination Critic:** Automatically validates the proposed solution against your coding rules and database constraints before printing anything.
5. **Structured Output:** Outputs the final instructions as a precise, structured JSON payload ready to be processed by coding tools.

---

## 2. The First Phase: Foundation (`TSK-001` through `TSK-004`)
The goal of **Phase 1: Foundation** is to establish the "skeleton" of the orchestrator—the state schemas, database saving mechanism (persistence), and a CLI wrapper. 

In this phase, we build:
- **`TSK-001` (Core State & Persistence):** Setup the PostgreSQL saver to checkpoint every state transition.
- **`TSK-002` (Decomposer Node):** Implement the logic that breaks down a prompt into tasks.
- **`TSK-003` (Perception Router):** Route flows based on intent.
- **`TSK-004` (CLI Interface):** The command-line program to talk to the orchestrator.

---

## 3. How to Manually Test Phase 1
You will be able to test this layer directly from your terminal to verify that it is tracking and saving state reliably:

### Step A: Run a Task
You start a session by providing a prompt:
```bash
python -m src.main run --prompt "Create a new FastAPI login endpoint"
```
The system will start planning the tasks, saving the state after each step in PostgreSQL.

### Step B: Interrupt Execution (Simulating a crash/cancel)
You can forcefully close the terminal or hit `Ctrl+C` in the middle of execution.

### Step C: Resume from Checkpoint
You run the CLI again, telling it to resume:
```bash
python -m src.main continue --thread-id <thread_id>
```
The orchestrator will connect to PostgreSQL, load the exact checkpoint from where you stopped, and print the remaining steps/questions. This guarantees that no context is ever lost.

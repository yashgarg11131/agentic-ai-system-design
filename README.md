# Agentic AI System for Multi-Step Decision Workflows

A production-grade, modular multi-agent AI system built with Python and FastAPI.
Demonstrates AI system design, modular architecture, and product thinking.

---

## 🔄 System Flow

1. User sends input via API
2. Orchestrator receives request
3. ClassifierAgent determines intent (question / analysis / summarisation / action)
4. Memory module retrieves prior session context
5. TaskAgent (or SummariserAgent) processes the task
6. Memory module stores the new turn
7. Evaluation module scores the output
8. Final structured response is returned

---

## Why This Project Matters

Traditional LLM-based systems struggle with multi-step reasoning, context retention, and complex decision workflows.
This project demonstrates how to build a **reliable, scalable, auditable** agentic system where:

- Every decision is **traceable** (full step-by-step execution log in every response)
- Quality is **measured** (multi-dimensional evaluation, not a single opaque score)
- Context **persists** across turns (session memory with TTL and eviction)
- Components are **independently replaceable** (swap any agent or module without touching others)

---

## Problem

Traditional LLM-based systems struggle with multi-step reasoning, context retention, and complex decision workflows. This leads to low accuracy, inconsistent responses, and poor reliability in production environments.

## Users

- Internal AI teams building automation workflows
- End users interacting with AI-driven systems
- Businesses requiring reliable AI decision-making

---

## Architecture

```
User Input (HTTP POST)
        │
        ▼
┌───────────────────┐
│    API Gateway    │  FastAPI + request instrumentation middleware
│   (api/main.py)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────────┐
│                      Orchestrator                         │
│  1. ClassifierAgent  → intent + routing hint              │
│  2. MemoryStore      → retrieve session context           │
│  3. Primary Agent    → TaskAgent | SummariserAgent        │
│  4. Evaluator        → multi-dimensional quality score    │
│  5. MemoryStore      → persist turns                      │
└───────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  WorkflowResult    │  { input, steps, output, evaluation_score }
└────────────────────┘
```

### Modules

| Module | File | Responsibility |
|---|---|---|
| Config | `config.py` | Centralised, frozen settings |
| Logger | `utils/logger.py` | Structured, cached named loggers |
| LLM Simulator | `utils/llm_simulator.py` | Drop-in LLM mock (no API key needed) |
| Memory | `memory/memory_store.py` | Thread-safe, TTL-aware session store |
| BaseAgent | `agents/base_agent.py` | Abstract contract; handles validation, timing, errors |
| ClassifierAgent | `agents/classifier_agent.py` | Intent classification + routing hints |
| TaskAgent | `agents/task_agent.py` | 5-stage task execution pipeline |
| SummariserAgent | `agents/summariser_agent.py` | Text condensation with compression metrics |
| Evaluator | `evaluation/evaluator.py` | Relevance · Completeness · Coherence · Confidence |
| Orchestrator | `orchestrator/orchestrator.py` | Full workflow coordination |
| Schemas | `api/schemas.py` | Pydantic request/response contracts |
| Routes | `api/routes.py` | FastAPI route handlers |
| App | `api/main.py` | App factory, middleware, lifecycle hooks |

---

## Project Structure

```
agentic-ai-system-design/
├── config.py                        # Frozen app/memory/eval settings
├── requirements.txt
│
├── utils/
│   ├── logger.py                    # Structured logging (cached loggers)
│   └── llm_simulator.py             # Deterministic LLM mock
│
├── memory/
│   └── memory_store.py              # Session store (thread-safe, TTL-evicting)
│
├── agents/
│   ├── base_agent.py                # Abstract base — Strategy pattern
│   ├── classifier_agent.py          # Intent classification
│   ├── task_agent.py                # Core task execution (5-stage pipeline)
│   └── summariser_agent.py          # Summarisation specialist
│
├── evaluation/
│   └── evaluator.py                 # Multi-dimensional quality scorer
│
├── orchestrator/
│   └── orchestrator.py              # Workflow coordinator
│
└── api/
    ├── schemas.py                   # Pydantic models (request + response)
    ├── routes.py                    # HTTP handlers
    └── main.py                      # FastAPI app factory + middleware
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd agentic-ai-system-design

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn api.main:app --reload --port 8000
```

The API is now live at **http://localhost:8000**

- Interactive docs (Swagger UI): http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

---

## API Reference

### `POST /api/v1/process` — Run workflow

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Analyse the trade-offs of multi-agent AI systems.",
    "session_id": "demo-session-001"
  }'
```

**Response:**
```json
{
  "input": "Analyse the trade-offs of multi-agent AI systems.",
  "steps": [
    { "agent": "ClassifierAgent", "output": "Classified intent: 'analysis'...", "confidence": 0.78 },
    { "agent": "TaskAgent",       "output": "Synthesised answer...",            "confidence": 0.81 },
    { "agent": "Evaluator",       "output": "Score 81% — Good",                "score": 0.812    }
  ],
  "output": "Analysing multi-agent AI systems: ...",
  "evaluation_score": {
    "score": 0.812,
    "grade": "Good",
    "explanation": "Score 81% — primary drag is coherence (72%).",
    "dimensions": {
      "relevance": 0.84,
      "completeness": 0.88,
      "coherence": 0.72,
      "confidence": 0.81
    },
    "passed": true
  },
  "_meta": {
    "request_id": "a3f2...",
    "session_id": "demo-session-001",
    "intent": "analysis",
    "total_latency_ms": 185.4,
    "total_tokens": 312,
    "success": true
  }
}
```

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/system/health` | Liveness probe |
| `GET` | `/api/v1/system/memory/stats` | Session store stats |
| `GET` | `/api/v1/system/agents/stats` | Per-agent call counts + token usage |
| `GET` | `/api/v1/sessions/{id}` | Retrieve session history |
| `DELETE` | `/api/v1/sessions/{id}` | Delete a session |
| `GET` | `/api/v1/sessions` | List all active sessions |

---

## Key Product Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | Multi-agent + orchestrator | Modularity, testability, independent scaling |
| Agent interface | Abstract base class | Uniform contract; orchestrator is agent-agnostic |
| Memory | In-memory dict + TTL | Zero-dependency for local dev; interface is storage-agnostic |
| Evaluation | 4-dimension weighted score | Richer signal than a single number; each dimension is swappable |
| LLM | Simulated | Zero cost, zero external deps; real API swap is one-class change |
| Error handling | Degraded WorkflowResult | API never crashes; every error surfaces in the response envelope |
| Extension point | `orchestrator.register_agent()` | New agents without touching existing code |

---

## Extending the System

**Add a new agent:**
```python
from agents.base_agent import BaseAgent, AgentResult

class MyNewAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MyNewAgent", description="Does X.")

    def _run(self, input_text, context):
        return AgentResult(agent_name=self.name, success=True, output="...")

# Register at runtime — no orchestrator changes needed
from orchestrator.orchestrator import orchestrator
orchestrator.register_agent(MyNewAgent())
```

**Swap in a real LLM:**
Replace `LLMSimulator.complete()` in `utils/llm_simulator.py` with an actual API call.
No other file needs to change.

---

## System Design Explained

### The Core Idea

Think of it like a company with a manager and specialists.

When a customer (user) sends a request, the **manager (Orchestrator)** doesn't try to do everything himself. He figures out what's needed, hands it to the right **specialist (Agent)**, checks the work quality, and files it away for next time.

### How One Request Flows

```
You send:  "Analyse the trade-offs of multi-agent AI systems."

1. CLASSIFY   → "This is an analysis request"          (ClassifierAgent)
2. REMEMBER   → "Last time this user asked about..."   (MemoryStore)
3. EXECUTE    → Run the 5-stage task pipeline          (TaskAgent)
4. EVALUATE   → "Good answer, but coherence is weak"   (Evaluator)
5. STORE      → Save this exchange for future turns    (MemoryStore)
6. RESPOND    → Return { input, steps, output, evaluation_score }
```

Every one of those steps is a separate module. They don't know about each other — only the Orchestrator does.

### Why Split Into Agents?

The monolith alternative: one big function that receives input, calls an LLM, returns output. Simple — but it breaks badly:

| Problem | Monolith | Multi-Agent |
|---|---|---|
| Bad output? | Can't tell why | Evaluator tells you which dimension failed |
| Add a new capability | Edit the one big function | Register a new agent, nothing else changes |
| Test summarisation alone | Can't — it's entangled | Call `SummariserAgent.process()` directly |
| Swap GPT-4 for Claude | Find and edit multiple places | Change one method in `llm_simulator.py` |

### Key Trade-offs

**1. In-memory store vs. a real database**
- Chose: Python dict + TTL
- Why: Zero dependencies, runs locally instantly
- Cost: Data dies when the server restarts
- Upgrade path: The `MemoryStore` interface is storage-agnostic — swap the dict for Redis by changing one class, zero call-site changes

**2. Simulated LLM vs. real API**
- Chose: Keyword-driven simulator
- Why: Anyone can run the system without API keys or costs
- Cost: Outputs aren't actually intelligent
- Upgrade path: Replace `LLMSimulator.complete()` — nothing else needs to change

**3. Heuristic evaluator vs. a model-based scorer**
- Chose: Regex + length ratios for 4 dimensions
- Why: Fast, deterministic, no external dependency
- Cost: Can miss nuance (a confident but wrong answer scores well)
- Upgrade path: Each dimension scorer is an isolated static method — replace one at a time with embedding similarity or an LLM judge

**4. Sequential agents vs. parallel**
- Chose: Sequential (classify → execute → evaluate)
- Why: Simpler, easier to reason about, each step can use the previous step's output
- Cost: Latency adds up — 3 agent hops = 3× the time
- Upgrade path: `OrchestratorConfig.enable_parallel_agents` flag is already in the config, intentionally left as a future lever

**5. Single primary agent per request vs. fan-out**
- Chose: Route to one agent (ClassifierAgent picks it)
- Why: Avoids result-merging complexity; predictable, auditable
- Cost: Can't run a "summarise AND classify" request efficiently
- Upgrade path: The `_dispatch()` method already accepts a `routing_hint` list — extend it to run multiple agents and merge

### Why Each Design Pattern Was Chosen

**Abstract BaseAgent (Strategy pattern)**
The Orchestrator calls `agent.process()` without knowing if it's a `TaskAgent` or a `SummariserAgent`. Drop in a new agent type and the entire rest of the system works unchanged.

**Module-level singletons**
`orchestrator`, `memory_store`, `evaluator` are created once at import time. No dependency injection framework needed — simple, predictable, testable by replacing the module-level variable in tests.

**Frozen config dataclasses**
All settings live in one place and can't be mutated at runtime. No "who changed this flag?" debugging. Environment-specific overrides are explicit, not scattered.

**App factory pattern (`create_app()`)**
The FastAPI app is built by a function, not at module level. Tests can call `create_app()` to get a fresh, isolated instance with no shared state between test cases.

### What This System Is NOT

- Not a real AI system — the LLM is simulated
- Not horizontally scalable as-is — in-memory state doesn't survive multiple server instances
- Not secured for production — CORS is open, no auth layer

These are deliberate simplifications to keep the architecture visible. The interfaces are designed so each gap can be filled independently without restructuring anything.

---

## Challenges

- Latency due to multi-agent calls
- Cost optimisation for LLM usage
- Managing context across agents

## Future Improvements

- Replace LLM simulator with real model API (Anthropic / OpenAI)
- Parallel agent execution for independent sub-tasks
- Persistent memory backend (Redis / PostgreSQL)
- Fine-tuned evaluation model replacing heuristics
- Real-time streaming responses (SSE / WebSocket)

---

## Tech Stack

Python 3.11+ · FastAPI · Pydantic v2 · Uvicorn

## Certification

Certified in Product Management and Agentic AI — IIT Patna (Vishlesan iHub x Masai)

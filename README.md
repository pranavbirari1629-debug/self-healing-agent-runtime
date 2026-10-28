# Self-Healing Agent Runtime (SHAR)

A lightweight, deterministic Python runtime providing resilient task execution, automated failure classification, self-healing recovery orchestration, post-execution output verification, permission-controlled tool execution, and bounded agent coordination.

SHAR is designed with **zero runtime dependencies**, relying strictly on the Python Standard Library to ensure deterministic execution, interview-defensible safety boundaries, and high testability.

---

## Architecture Overview

SHAR implements a multi-layer pipeline architecture where each component has distinct, isolated responsibilities:

```
[ User Goal / Host Application ]
               │
               ▼
   [ Agent Coordination Layer ]  ◄── (Agent, ScriptedPlanner, AgentAction, AgentStep, AgentResult)
               │
               ▼
    [ Tool Integration Layer ]   ◄── (ToolOrchestratorBridge, ToolRegistry, ToolSpec, ToolParameter)
               │
               ▼
   [ Recovery Orchestration ]    ◄── (RecoveryOrchestrator, RecoveryPolicy, RecoveryMemory)
               │
       ┌───────┴───────┐
       ▼               ▼
 [ TaskRunner ]  [ OutputVerifier ]
```

### Component Responsibilities

1. **Task Execution (`Task`, `TaskRunner`)**:
   * Encapsulates units of work with arguments, execution-time validators, deadlines (`timeout_seconds`), and idempotency declarations.
   * Dispatches timed tasks to an internal `ThreadPoolExecutor`.
   * **Safety Boundary**: Accurately accounts for the fact that CPython worker threads cannot be forcibly killed externally upon timeout.
2. **Failure Detection & Diagnosis (`FailureDetector`, `DiagnosisEngine`)**:
   * Deterministically classifies execution failures into `TRANSIENT`, `PERMANENT`, or `UNKNOWN` based on exception registries.
   * Diagnoses structured root causes without external LLM or network dependencies.
3. **Recovery Policy (`RecoveryPolicy`)**:
   * Pure decision engine evaluating 5 safety gates: failure category, idempotency confirmation, timeout retry authorization, retry caps, and cumulative recovery budgets.
   * Authorizes exponential backoff delays with optional bounded jitter.
   * Refuses retries or fallback transitions on timed-out tasks unless `safe_to_retry_on_timeout` was explicitly declared.
4. **Recovery Orchestrator (`RecoveryOrchestrator`)**:
   * Loop controller coordinating execution attempts, backoff sleeps, bounded fallback delegation (`max_fallback_depth`), and attempt history tracking.
   * Enforces wall-clock recovery budgets across all attempts and fallback tasks.
5. **Output Verification & Memory (`OutputVerifier`, `RecoveryMemory`)**:
   * Decouples raw execution success from semantic output correctness. Enforces strict boolean verification invariants.
   * Maintains a bounded, thread-safe FIFO memory capturing anonymized recovery metadata for statistical observation without influencing policy decisions.
6. **Tool Integration (`ToolRegistry`, `ToolOrchestratorBridge`)**:
   * Defines typed parameters (`ToolParameter`) with strict validation (including a guard preventing `bool` from passing as `int`).
   * Enforces caller permissions immediately before execution at the bridge boundary.
   * Validates fallback tool permissions independently to prevent privilege escalation.
7. **Agent Coordination (`Agent`, `ScriptedPlanner`)**:
   * Coordinates multi-step goal execution through the tool bridge.
   * Enforces strict step limits (`max_steps`) and tool-call caps (`max_tool_calls`) prior to invocation.
   * Keeps tool-level verification distinct from overall goal-level verification.

---

## Current Status & Scope

* **Implemented Phases**:
  * Phase 1: Core runtime foundation (`Task`, `TaskRunner`, structured JSON logging).
  * Phase 2: Failure detection and diagnosis engine.
  * Phase 3: Recovery policy engine and recovery orchestration.
  * Phase 4: Output verification and observational recovery memory.
  * Phase 5: Deterministic evaluation suite and 3-baseline benchmark harness.
  * Phase 6, Step 1: Permission-aware tool integration (`ToolSpec`, `ToolRegistry`, `ToolOrchestratorBridge`).
  * Phase 6, Step 2: Bounded agent coordination layer (`Agent`, `ScriptedPlanner`, `AgentResult`).
* **Design Boundary Clarification**:
  * SHAR currently provides a deterministic `ScriptedPlanner` supporting static action sequences and rule-based decision callables for predictable testing and controlled workflows.
  * Autonomous LLM-based planning and external model providers are **not implemented** in this version. The `AgentPlanner` protocol is decoupled to allow future planner plugins without modifying runtime or recovery code.
  * Benchmark results reflect deterministic project test workloads and do not constitute independent real-world operational guarantees.

---

## Project Structure

```text
self-healing-agent-runtime/
├── benchmarks/
│   ├── results/                  # Generated benchmark output (git-ignored)
│   └── run_benchmark.py          # Standalone 11-workload benchmark harness
├── src/
│   ├── main.py                   # Sample application entry point
│   └── shar/                     # Core runtime package
│       ├── __init__.py           # Public API exports
│       ├── agent.py              # Agent coordinator and ScriptedPlanner
│       ├── detector.py           # Failure classification
│       ├── diagnosis.py          # Root cause analysis
│       ├── logger.py             # Structured JSON logger
│       ├── memory.py             # Observational recovery memory
│       ├── models.py             # Core dataclasses and enums
│       ├── orchestrator.py       # RecoveryOrchestrator loop
│       ├── policy.py             # Gated recovery decision engine
│       ├── runner.py             # TaskRunner and thread pool management
│       ├── tools.py              # ToolParameter, ToolSpec, Bridge
│       └── verifier.py           # OutputVerifier
├── tests/                        # Pytest suite (270 unit & scenario tests)
├── .gitignore
├── pyproject.toml                # Project metadata and pytest configuration
└── README.md
```

---

## Prerequisites & Setup

### Prerequisites
* Python `>= 3.11` (Tested on Python 3.14.3 on Windows).
* Virtual environment (`venv`).

### Setup
Clone the repository and set up a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install development dependencies (pytest)
pip install pytest>=8.0
```

---

## Running Tests & Benchmarks

### Run the Pytest Suite
Execute all 270 unit, scenario, and integration tests:

```bash
# Run complete test suite quietly
.venv\Scripts\pytest -q

# Run specific agent and tool test suites with verbose output
.venv\Scripts\pytest tests/test_agent.py tests/test_tools.py -v
```

### Run the Benchmark Harness
SHAR includes a deterministic benchmark evaluating 11 representative workloads across three distinct operational baselines:
* **Baseline A (Unassisted)**: Single-attempt execution via `TaskRunner` with recovery disabled.
* **Baseline B (Self-Healing)**: Full recovery orchestration (retries, backoff, fallbacks, verifiers) with memory disabled.
* **Baseline C (SHAR + Memory)**: Full recovery orchestration with observational `RecoveryMemory` enabled.

```bash
.venv\Scripts\python benchmarks/run_benchmark.py
```

*Note: Under identical workloads, Baselines B and C produce 100% identical execution and verification verdicts, demonstrating that `RecoveryMemory` is purely observational and does not alter policy decisions.*

---

## Usage Examples

### 1. Basic Task Execution with Self-Healing Recovery

```python
from shar import RecoveryOrchestrator, Task

attempts = 0

def flaky_network_call(endpoint: str) -> dict:
    global attempts
    attempts += 1
    if attempts < 2:
        raise ConnectionError("Temporary upstream reset")
    return {"status": "ok", "endpoint": endpoint}

# Configure an idempotent task with retry and verification boundaries
task = Task(
    name="fetch_user_data",
    callable=flaky_network_call,
    kwargs={"endpoint": "/api/v1/users"},
    is_idempotent=True,
    max_retries=2,
    backoff_base_delay=0.01,
    verifier=lambda out: isinstance(out, dict) and out.get("status") == "ok",
)

with RecoveryOrchestrator() as orchestrator:
    result = orchestrator.execute(task)

print(f"Status: {result.status.value}")       # recovered_via_retry
print(f"Total attempts: {result.total_attempts}") # 2
print(f"Verified: {result.is_verified}")       # True
print(f"Output: {result.final_result.output}") # {'status': 'ok', 'endpoint': '/api/v1/users'}
```

### 2. Multi-Step Agent Workflow with Tool Registry

```python
from shar import (
    Agent,
    AgentAction,
    AgentActionType,
    ScriptedPlanner,
    ToolOrchestratorBridge,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
)

# 1. Register tools
registry = ToolRegistry()
registry.register(ToolSpec(
    name="get_reading",
    description="Fetches raw sensor reading",
    callable=lambda sensor_id: 45.0,
    parameters=(ToolParameter(name="sensor_id", type=str),),
    is_idempotent=True,
))
registry.register(ToolSpec(
    name="calibrate",
    description="Calibrates reading with verified multiplier",
    callable=lambda raw_val, offset: (raw_val + offset) * 1.1,
    parameters=(
        ToolParameter(name="raw_val", type=float),
        ToolParameter(name="offset", type=float),
    ),
    is_idempotent=True,
    verifier=lambda out: isinstance(out, float) and out > 0.0,
))

# 2. Define deterministic planning sequence
def plan_chain(goal, tools, history):
    if len(history) == 0:
        return AgentAction(AgentActionType.TOOL_CALL, tool_name="get_reading", arguments={"sensor_id": "sensor_A"})
    if len(history) == 1:
        raw = history[0].tool_result.output
        return AgentAction(AgentActionType.TOOL_CALL, tool_name="calibrate", arguments={"raw_val": raw, "offset": 5.0})
    if len(history) == 2:
        calibrated = history[1].tool_result.output
        return AgentAction(AgentActionType.FINISH, output={"calibrated_reading": calibrated})
    return AgentAction(AgentActionType.FAIL)

# 3. Execute via Agent
bridge = ToolOrchestratorBridge(registry)
agent = Agent(bridge=bridge, planner=ScriptedPlanner(plan_chain), max_steps=5, max_tool_calls=5)

result = agent.run("Calibrate sensor A reading")

print(f"Agent Status: {result.status.value}")   # completed
print(f"Total Steps: {result.total_steps}")     # 3
print(f"Tool Calls: {result.total_tool_calls}") # 2
print(f"Final Output: {result.output}")         # {'calibrated_reading': 55.0}
```

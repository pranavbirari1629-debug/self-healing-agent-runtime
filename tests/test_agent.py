"""Deterministic unit and integration tests for Phase 6, Step 2 Agent Layer in shar.agent."""

import time
import pytest

from shar import (
    Agent,
    AgentAction,
    AgentActionType,
    AgentPlanner,
    AgentResult,
    AgentStatus,
    AgentStep,
    RecoveryOrchestrator,
    RecoveryStatus,
    ScriptedPlanner,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolOrchestratorBridge,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Fixtures and Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def fast_orchestrator() -> RecoveryOrchestrator:
    """Orchestrator with instant sleep for fast tests."""
    return RecoveryOrchestrator(sleep_fn=lambda _: None)


@pytest.fixture
def basic_bridge(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator) -> ToolOrchestratorBridge:
    return ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)


# ---------------------------------------------------------------------------
# 1. Immediate FINISH without tool calls
# ---------------------------------------------------------------------------

def test_agent_immediate_finish(basic_bridge: ToolOrchestratorBridge):
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FINISH, output={"answer": 42}, thought="Already solved"),
    ])
    agent = Agent(bridge=basic_bridge, planner=planner, max_steps=5, max_tool_calls=5)

    result = agent.run(goal="Answer immediately")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.is_verified is False  # No goal verifier
    assert result.output == {"answer": 42}
    assert result.total_steps == 1
    assert result.total_tool_calls == 0
    assert len(result.steps) == 1
    assert result.steps[0].action.action_type == AgentActionType.FINISH
    assert result.steps[0].tool_result is None


# ---------------------------------------------------------------------------
# 2. Successful single-tool execution followed by FINISH
# ---------------------------------------------------------------------------

def test_agent_single_tool_then_finish(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="double_num",
        description="Doubles an integer",
        callable=lambda n: n * 2,
        parameters=(ToolParameter(name="n", type=int),),
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="double_num", arguments={"n": 21}),
        AgentAction(action_type=AgentActionType.FINISH, output=42, thought="Finished calculation"),
    ])
    agent = Agent(bridge=bridge, planner=planner, max_steps=5, max_tool_calls=5)

    result = agent.run(goal="Double 21")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.output == 42
    assert result.total_steps == 2
    assert result.total_tool_calls == 1
    assert len(result.steps) == 2
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.output == 42


# ---------------------------------------------------------------------------
# 3. Multi-tool chaining with output passed between tools
# ---------------------------------------------------------------------------

def test_agent_multi_tool_chaining(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="fetch_base",
        description="Fetches base number",
        callable=lambda: 100,
        is_idempotent=True,
    ))
    empty_registry.register(ToolSpec(
        name="apply_discount",
        description="Subtracts discount",
        callable=lambda amount, discount: amount - discount,
        parameters=(
            ToolParameter(name="amount", type=int),
            ToolParameter(name="discount", type=int),
        ),
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    def dynamic_logic(goal, tools, history):
        if len(history) == 0:
            return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="fetch_base")
        if len(history) == 1:
            base = history[0].tool_result.output
            return AgentAction(
                action_type=AgentActionType.TOOL_CALL,
                tool_name="apply_discount",
                arguments={"amount": base, "discount": 15},
            )
        if len(history) == 2:
            final_val = history[1].tool_result.output
            return AgentAction(action_type=AgentActionType.FINISH, output=final_val)
        return AgentAction(action_type=AgentActionType.FAIL, thought="Unexpected step")

    planner = ScriptedPlanner(dynamic_logic)
    agent = Agent(bridge=bridge, planner=planner, max_steps=5, max_tool_calls=5)

    result = agent.run(goal="Compute discounted price")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.output == 85
    assert result.total_steps == 3
    assert result.total_tool_calls == 2


# ---------------------------------------------------------------------------
# 4. Correct correlation ID propagation
# ---------------------------------------------------------------------------

def test_agent_correlation_id_propagation(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="ping",
        description="Echo ping",
        callable=lambda: "pong",
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="ping"),
        AgentAction(action_type=AgentActionType.FINISH, output="done"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    custom_cid = "trace-custom-correlation-id-777"
    result = agent.run(goal="Ping server", correlation_id=custom_cid)

    assert result.correlation_id == custom_cid
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.correlation_id == custom_cid
    assert result.steps[0].tool_result.recovery_result.correlation_id == custom_cid


# ---------------------------------------------------------------------------
# 5. Generated correlation ID when omitted
# ---------------------------------------------------------------------------

def test_agent_generated_correlation_id_when_omitted(basic_bridge: ToolOrchestratorBridge):
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FINISH, output="ok"),
    ])
    agent = Agent(bridge=basic_bridge, planner=planner)

    result = agent.run(goal="Quick finish")

    assert result.correlation_id != ""
    assert isinstance(result.correlation_id, str)
    assert len(result.correlation_id) >= 16


# ---------------------------------------------------------------------------
# 6. Planner FAIL action
# ---------------------------------------------------------------------------

def test_agent_planner_fail_action(basic_bridge: ToolOrchestratorBridge):
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FAIL, thought="No viable strategy available"),
    ])
    agent = Agent(bridge=basic_bridge, planner=planner)

    result = agent.run(goal="Impossible goal")

    assert result.status == AgentStatus.FAILED
    assert result.is_success is False
    assert result.is_verified is False
    assert result.error == "No viable strategy available"
    assert result.total_steps == 1
    assert result.total_tool_calls == 0


# ---------------------------------------------------------------------------
# 7. Planner exception containment
# ---------------------------------------------------------------------------

def test_agent_planner_exception_contained(basic_bridge: ToolOrchestratorBridge):
    def crashing_planner(goal, tools, history):
        raise ZeroDivisionError("Math error inside planner deliberation")

    agent = Agent(bridge=basic_bridge, planner=ScriptedPlanner(crashing_planner))

    result = agent.run(goal="Crash test")

    assert result.status == AgentStatus.PLANNING_ERROR
    assert result.is_success is False
    assert result.is_verified is False
    assert "ZeroDivisionError" in result.error
    assert "Math error inside planner deliberation" in result.error
    assert result.total_steps == 0


# ---------------------------------------------------------------------------
# 8. Invalid planner return value
# ---------------------------------------------------------------------------

def test_agent_invalid_planner_return_value(basic_bridge: ToolOrchestratorBridge):
    class DirectBadPlanner:
        def plan(self, goal, tools, history):
            return "not an AgentAction instance"  # Invalid return type

    agent = Agent(bridge=basic_bridge, planner=DirectBadPlanner())

    result = agent.run(goal="Type test")

    assert result.status == AgentStatus.PLANNING_ERROR
    assert result.is_success is False
    assert "Planner returned invalid type 'str'" in result.error
    assert result.total_steps == 0


# ---------------------------------------------------------------------------
# 9. Exhausted ScriptedPlanner
# ---------------------------------------------------------------------------

def test_agent_exhausted_scripted_planner(basic_bridge: ToolOrchestratorBridge):
    # Empty action sequence
    planner = ScriptedPlanner([])
    agent = Agent(bridge=basic_bridge, planner=planner)

    result = agent.run(goal="Empty sequence")

    assert result.status == AgentStatus.FAILED
    assert result.is_success is False
    assert "exhausted" in result.error.lower()
    assert result.total_steps == 1


# ---------------------------------------------------------------------------
# 10. Step limit enforcement
# ---------------------------------------------------------------------------

def test_agent_step_limit_enforced(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="noop",
        description="No-op",
        callable=lambda: None,
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    # Infinite tool loop
    def infinite_loop(goal, tools, history):
        return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="noop")

    planner = ScriptedPlanner(infinite_loop)
    agent = Agent(bridge=bridge, planner=planner, max_steps=3, max_tool_calls=10)

    result = agent.run(goal="Infinite loop goal")

    assert result.status == AgentStatus.STEP_LIMIT_EXCEEDED
    assert result.is_success is False
    assert result.is_verified is False
    assert result.total_steps == 3
    assert result.total_tool_calls == 3
    assert len(result.steps) == 3
    assert "Step limit of 3 reached" in result.error


# ---------------------------------------------------------------------------
# 11. Tool-call limit enforcement
# ---------------------------------------------------------------------------

def test_agent_tool_call_limit_enforced(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="action",
        description="Action",
        callable=lambda: "done",
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    # Request tool calls continuously
    def loop_tool(goal, tools, history):
        return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="action")

    planner = ScriptedPlanner(loop_tool)
    agent = Agent(bridge=bridge, planner=planner, max_steps=10, max_tool_calls=2)

    result = agent.run(goal="Tool call limit goal")

    assert result.status == AgentStatus.TOOL_CALL_LIMIT_EXCEEDED
    assert result.is_success is False
    assert result.is_verified is False
    assert result.total_tool_calls == 2  # Exactly 2 calls executed; the 3rd was blocked BEFORE execution!
    assert "Tool-call limit of 2 reached" in result.error
    assert len(result.steps) == 3
    assert result.steps[2].tool_result is None  # 3rd step was stopped before tool execution


# ---------------------------------------------------------------------------
# 12. Unknown tool handling
# ---------------------------------------------------------------------------

def test_agent_unknown_tool_handling(basic_bridge: ToolOrchestratorBridge):
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="non_existent_tool"),
        AgentAction(action_type=AgentActionType.FAIL, thought="Cannot continue without tool"),
    ])
    agent = Agent(bridge=basic_bridge, planner=planner)

    result = agent.run(goal="Use missing tool")

    assert result.status == AgentStatus.FAILED
    assert result.total_steps == 2
    assert result.total_tool_calls == 1
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.TOOL_NOT_FOUND


# ---------------------------------------------------------------------------
# 13. Unauthorized tool handling with confirmation callable was not invoked
# ---------------------------------------------------------------------------

def test_agent_unauthorized_tool_handling(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    call_count = 0

    def secure_action():
        nonlocal call_count
        call_count += 1
        return "secret"

    empty_registry.register(ToolSpec(
        name="secure_tool",
        description="Requires admin",
        callable=secure_action,
        required_permissions=frozenset({"admin"}),
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="secure_tool"),
        AgentAction(action_type=AgentActionType.FAIL, thought="Permission denied"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Access secure tool", caller_permissions=frozenset({"user"}))

    assert result.status == AgentStatus.FAILED
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.PERMISSION_DENIED
    assert call_count == 0  # Callable was NEVER invoked!


# ---------------------------------------------------------------------------
# 14. Invalid arguments with confirmation callable was not invoked
# ---------------------------------------------------------------------------

def test_agent_invalid_arguments_handling(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    call_count = 0

    def calculate(n: int):
        nonlocal call_count
        call_count += 1
        return n * 10

    empty_registry.register(ToolSpec(
        name="calc",
        description="Calculation",
        callable=calculate,
        parameters=(ToolParameter(name="n", type=int),),
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    # Pass bool True for int parameter n (rejected by bool-as-int guard)
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="calc", arguments={"n": True}),
        AgentAction(action_type=AgentActionType.FAIL, thought="Invalid argument rejected"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Calculate")

    assert result.status == AgentStatus.FAILED
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert "expected int, got bool" in result.steps[0].tool_result.error
    assert call_count == 0  # Callable was NEVER invoked!


# ---------------------------------------------------------------------------
# 15. Successful recovery retry handled by the existing bridge
# ---------------------------------------------------------------------------

def test_agent_successful_recovery_retry(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    attempts = 0

    def flaky_service():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Temporary disconnect")
        return "data_ok"

    empty_registry.register(ToolSpec(
        name="flaky_tool",
        description="Flaky network call",
        callable=flaky_service,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.001,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="flaky_tool"),
        AgentAction(action_type=AgentActionType.FINISH, output="recovered_data"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Fetch from flaky service")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert attempts == 2
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.SUCCESS
    assert result.steps[0].tool_result.recovery_result.status == RecoveryStatus.RECOVERED_VIA_RETRY


# ---------------------------------------------------------------------------
# 16. Successful fallback handled by the existing bridge
# ---------------------------------------------------------------------------

def test_agent_successful_fallback(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="cached_data",
        description="Fallback cache",
        callable=lambda: "cached_payload",
        is_idempotent=True,
    ))
    empty_registry.register(ToolSpec(
        name="primary_data",
        description="Primary source (fails permanently)",
        callable=lambda: (_ for _ in ()).throw(ValueError("Invalid primary source")),
        is_idempotent=True,
        fallback_tool_name="cached_data",
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="primary_data"),
        AgentAction(action_type=AgentActionType.FINISH, output="fallback_used"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Fetch data with fallback")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.SUCCESS
    assert result.steps[0].tool_result.recovery_result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK


# ---------------------------------------------------------------------------
# 17. Terminal tool failure followed by a planner-selected alternative
# ---------------------------------------------------------------------------

def test_agent_pivots_after_tool_failure(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="failing_tool",
        description="Fails permanently without fallback",
        callable=lambda: (_ for _ in ()).throw(ValueError("Irrecoverable error")),
        is_idempotent=True,
    ))
    empty_registry.register(ToolSpec(
        name="alternative_tool",
        description="Working alternate",
        callable=lambda: "alt_success",
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    def pivot_logic(goal, tools, history):
        if len(history) == 0:
            return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="failing_tool")
        if len(history) == 1:
            assert history[0].tool_result.status == ToolExecutionStatus.EXECUTION_FAILED
            # Planner observes failure and pivots to alternative tool!
            return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="alternative_tool")
        if len(history) == 2:
            return AgentAction(action_type=AgentActionType.FINISH, output=history[1].tool_result.output)
        return AgentAction(action_type=AgentActionType.FAIL)

    planner = ScriptedPlanner(pivot_logic)
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Try primary, pivot on failure")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.output == "alt_success"
    assert result.total_steps == 3
    assert result.total_tool_calls == 2
    assert result.steps[0].tool_result.status == ToolExecutionStatus.EXECUTION_FAILED
    assert result.steps[1].tool_result.status == ToolExecutionStatus.SUCCESS


# ---------------------------------------------------------------------------
# 18. Timeout recorded without an external retry loop
# ---------------------------------------------------------------------------

def test_agent_handles_timeout_without_retry_loop(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    def slow_task():
        time.sleep(0.15)
        return "late"

    empty_registry.register(ToolSpec(
        name="slow_tool",
        description="Times out",
        callable=slow_task,
        timeout_seconds=0.02,
        safe_to_retry_on_timeout=False,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="slow_tool"),
        AgentAction(action_type=AgentActionType.FAIL, thought="Tool timed out; stopping"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Timeout test")

    assert result.status == AgentStatus.FAILED
    assert result.total_steps == 2
    assert result.total_tool_calls == 1  # Only 1 tool call performed; no loop
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.status == ToolExecutionStatus.TIMEOUT


# ---------------------------------------------------------------------------
# 19. Unverified tool output does not become a verified goal result
# ---------------------------------------------------------------------------

def test_unverified_tool_output_does_not_verify_goal(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="unverified_calc",
        description="No verifier configured",
        callable=lambda: 100,
        verifier=None,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="unverified_calc"),
        AgentAction(action_type=AgentActionType.FINISH, output=100),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    result = agent.run(goal="Unverified test")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.is_verified is False  # Must NOT be marked verified
    assert result.steps[0].tool_result.verification_status == VerificationStatus.UNVERIFIED


# ---------------------------------------------------------------------------
# 20. Verified terminal tool output does not automatically mark overall goal verified
# ---------------------------------------------------------------------------

def test_verified_terminal_tool_does_not_auto_verify_goal(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="verified_calc",
        description="Tool with passing verifier",
        callable=lambda: 200,
        verifier=lambda out: out == 200,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="verified_calc"),
        AgentAction(action_type=AgentActionType.FINISH, output=200),
    ])
    # Agent has NO goal_verifier configured
    agent = Agent(bridge=bridge, planner=planner, goal_verifier=None)

    result = agent.run(goal="Goal verification separation test")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.steps[0].tool_result.is_verified is True  # Tool itself was verified
    assert result.is_verified is False  # Overall goal is NOT verified without explicit goal verifier!


# ---------------------------------------------------------------------------
# 21. Failed tool verification does not become verified goal completion
# ---------------------------------------------------------------------------

def test_failed_tool_verification_does_not_verify_goal(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    empty_registry.register(ToolSpec(
        name="bad_verification_tool",
        description="Tool whose verifier fails",
        callable=lambda: 50,
        verifier=lambda out: out == 999,  # Fails!
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="bad_verification_tool"),
        AgentAction(action_type=AgentActionType.FINISH, output=50),
    ])
    agent = Agent(bridge=bridge, planner=planner, goal_verifier=lambda out: out == 999)

    result = agent.run(goal="Failed verifier test")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.steps[0].tool_result.verification_status == VerificationStatus.FAILED
    assert result.steps[0].tool_result.is_verified is False
    assert result.is_verified is False


# ---------------------------------------------------------------------------
# 22. Planner cannot escalate permissions
# ---------------------------------------------------------------------------

def test_planner_cannot_escalate_permissions(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    call_count = 0

    def root_only_exec():
        nonlocal call_count
        call_count += 1
        return "root_executed"

    empty_registry.register(ToolSpec(
        name="root_tool",
        description="Superuser tool",
        callable=root_only_exec,
        required_permissions=frozenset({"root"}),
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    # Planner tries to call root_tool
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="root_tool"),
        AgentAction(action_type=AgentActionType.FAIL, thought="Denied"),
    ])
    agent = Agent(bridge=bridge, planner=planner)

    # Caller only has "guest"
    result = agent.run(goal="Try escalation", caller_permissions=frozenset({"guest"}))

    assert result.status == AgentStatus.FAILED
    assert result.steps[0].tool_result.status == ToolExecutionStatus.PERMISSION_DENIED
    assert call_count == 0  # Forbidden tool was NEVER called


# ---------------------------------------------------------------------------
# 23. Malformed tool output is handled safely
# ---------------------------------------------------------------------------

def test_malformed_tool_output_handled_safely(empty_registry: ToolRegistry, fast_orchestrator: RecoveryOrchestrator):
    # Tool returning unexpected types (e.g. None or object)
    empty_registry.register(ToolSpec(
        name="odd_tool",
        description="Returns object()",
        callable=lambda: object(),
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry=empty_registry, orchestrator=fast_orchestrator)

    def robust_planner(goal, tools, history):
        if len(history) == 0:
            return AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="odd_tool")
        output = history[0].tool_result.output
        # Safe type inspection
        if isinstance(output, dict):
            return AgentAction(action_type=AgentActionType.FINISH, output=output.get("val"))
        return AgentAction(action_type=AgentActionType.FINISH, output="handled_non_dict")

    agent = Agent(bridge=bridge, planner=ScriptedPlanner(robust_planner))
    result = agent.run(goal="Robust output handling")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.output == "handled_non_dict"


# ---------------------------------------------------------------------------
# 24. History and counters remain consistent on every terminal path
# ---------------------------------------------------------------------------

def test_history_and_counters_consistency(basic_bridge: ToolOrchestratorBridge):
    # Test on COMPLETED path
    p1 = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FINISH, output="ok"),
    ])
    r1 = Agent(basic_bridge, p1).run("Goal 1")
    assert r1.total_steps == len(r1.steps) == 1
    assert r1.total_tool_calls == 0
    assert r1.total_elapsed_ms >= 0.0

    # Test on FAILED path
    p2 = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FAIL, thought="Failed deliberately"),
    ])
    r2 = Agent(basic_bridge, p2).run("Goal 2")
    assert r2.total_steps == len(r2.steps) == 1
    assert r2.total_tool_calls == 0

    # Test on PLANNING_ERROR path
    def crasher(g, t, h):
        raise RuntimeError("boom")
    r3 = Agent(basic_bridge, ScriptedPlanner(crasher)).run("Goal 3")
    assert r3.total_steps == len(r3.steps) == 0
    assert r3.total_tool_calls == 0


# ---------------------------------------------------------------------------
# 25. Public API exports are correct
# ---------------------------------------------------------------------------

def test_public_api_exports():
    import shar

    for symbol in (
        "AgentActionType",
        "AgentStatus",
        "AgentAction",
        "AgentStep",
        "AgentResult",
        "AgentPlanner",
        "ScriptedPlanner",
        "Agent",
    ):
        assert hasattr(shar, symbol), f"shar is missing public export '{symbol}'"


# ---------------------------------------------------------------------------
# 26. Data Model Validation & Immutability Tests
# ---------------------------------------------------------------------------

def test_agent_action_validation():
    with pytest.raises(TypeError, match="action_type must be an AgentActionType"):
        AgentAction(action_type="invalid_type")

    with pytest.raises(ValueError, match="non-empty tool_name"):
        AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="")

    # Defensive copy of arguments dict
    orig_dict = {"a": 1}
    action = AgentAction(action_type=AgentActionType.TOOL_CALL, tool_name="tool", arguments=orig_dict)
    orig_dict["a"] = 999
    assert action.arguments["a"] == 1  # Action arguments unchanged!


def test_agent_result_invariants():
    # Unsuccessful cannot be verified
    with pytest.raises(ValueError, match="cannot be marked is_verified=True"):
        AgentResult(
            goal="Test",
            status=AgentStatus.FAILED,
            is_success=False,
            is_verified=True,
        )

    # Non-completed cannot have is_success=True
    with pytest.raises(ValueError, match="cannot have is_success=True"):
        AgentResult(
            goal="Test",
            status=AgentStatus.STEP_LIMIT_EXCEEDED,
            is_success=True,
        )


def test_agent_goal_verifier_integration(basic_bridge: ToolOrchestratorBridge):
    planner = ScriptedPlanner([
        AgentAction(action_type=AgentActionType.FINISH, output={"verified_field": "pass"}),
    ])
    # Agent with explicit goal_verifier
    agent = Agent(
        bridge=basic_bridge,
        planner=planner,
        goal_verifier=lambda out: isinstance(out, dict) and out.get("verified_field") == "pass",
    )
    result = agent.run("Goal with verifier")

    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.is_verified is True

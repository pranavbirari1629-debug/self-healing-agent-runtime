"""Deterministic evaluation test suite for the Self-Healing AI Agent Runtime (shar).

Implements the 15 reproducible evaluation and failure-injection scenarios identified
in the Phase 5 benchmarking audit.

Scenarios Covered:
 1. First-attempt success.
 2. Transient failure followed by successful retry.
 3. Transient failure followed by retry exhaustion.
 4. Permanent failure with no inappropriate retry.
 5. Non-idempotent task failure with no retry.
 6. Timeout when safe_to_retry_on_timeout is False.
 7. Timeout when safe_to_retry_on_timeout is True (bounded worker thread).
 8. Recovery-budget exhaustion.
 9. Successful execution with VERIFIED output.
10. Successful execution with UNVERIFIED output.
11. Successful execution with FAILED verification.
12. Successful execution with INCONCLUSIVE verification.
13. Fallback success.
14. Fallback failure.
15. Memory-recording failure without corruption of the recovery result.
"""

from __future__ import annotations

import threading
import time
from typing import Any
import pytest

from shar.detector import FailureDetector
from shar.diagnosis import DiagnosisEngine
from shar.memory import RecoveryMemory
from shar.models import (
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskOutcome,
    VerificationStatus,
)
from shar.orchestrator import RecoveryOrchestrator
from shar.policy import RecoveryPolicy
from shar.runner import TaskRunner
from shar.verifier import OutputVerifier


class MockClock:
    """Deterministic monotonic clock for simulation in tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.current_time = start

    def time(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class MockSleeper:
    """Mock sleep function that records requested sleep delays without real waiting."""

    def __init__(self, clock: MockClock | None = None) -> None:
        self.delays: list[float] = []
        self.clock = clock

    def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)


@pytest.fixture
def eval_env():
    """Provides a clean orchestrator configured with mock clock, mock sleeper, and memory."""
    clock = MockClock(start=100.0)
    sleeper = MockSleeper(clock=clock)
    runner = TaskRunner()
    memory = RecoveryMemory()
    orchestrator = RecoveryOrchestrator(
        runner=runner,
        sleep_fn=sleeper.sleep,
        time_fn=clock.time,
        memory=memory,
    )
    yield orchestrator, clock, sleeper, runner, memory
    orchestrator.close()
    runner.close()


# ===========================================================================
# Scenario 1: First-attempt success
# ===========================================================================

def test_scenario_1_first_attempt_success(eval_env):
    """Scenario 1: Ordinary first-attempt success must not record recovery memory."""
    orchestrator, _, sleeper, _, memory = eval_env

    task = Task("scenario_1_task", callable=lambda: {"status": "ok", "items": [1, 2]})
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.total_attempts == 1
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.UNVERIFIED
    assert result.is_verified is False
    assert len(sleeper.delays) == 0

    # Invariant: Ordinary first-attempt success must not fabricate recovery history
    assert len(memory) == 0


# ===========================================================================
# Scenario 2: Transient failure followed by successful retry
# ===========================================================================

def test_scenario_2_transient_failure_recovered_via_retry(eval_env):
    """Scenario 2: Transient failure recovered on retry records accurate history."""
    orchestrator, _, sleeper, _, memory = eval_env

    attempt_count = 0
    def flaky_work():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ConnectionError("Transient network glitch")
        return {"data": 42}

    task = Task(
        name="scenario_2_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.1,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.total_attempts == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].task_result.is_success is False
    assert result.attempts[1].task_result.is_success is True
    assert len(sleeper.delays) == 1
    assert sleeper.delays[0] == 0.1

    # Invariant: Exactly one recovery entry recorded attributing retry success
    assert len(memory) == 1
    entries = memory.find_similar("scenario_2_task", "network_connectivity")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.task_name == "scenario_2_task"
    assert entry.recovery_action == RecoveryAction.RETRY
    assert entry.recovered_execution is True
    assert entry.terminal_status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert entry.attempt_count == 2
    assert entry.fallback_task_name is None


# ===========================================================================
# Scenario 3: Transient failure followed by retry exhaustion
# ===========================================================================

def test_scenario_3_transient_failure_retry_exhaustion(eval_env):
    """Scenario 3: Transient failure exceeding max_retries reaches FAILED_RETRIES_EXHAUSTED."""
    orchestrator, _, sleeper, _, memory = eval_env

    attempt_count = 0
    def persistent_failure():
        nonlocal attempt_count
        attempt_count += 1
        raise ConnectionError("Host permanently unreachable")

    task = Task(
        name="scenario_3_task",
        callable=persistent_failure,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.1,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_RETRIES_EXHAUSTED
    assert result.is_success is False
    assert result.is_verified is False
    assert result.total_attempts == 3  # Initial attempt + 2 retries
    assert attempt_count == 3
    assert len(sleeper.delays) == 2

    # Invariant: Exactly one entry recorded for terminal recovery failure
    assert len(memory) == 1
    entry = memory.find_similar("scenario_3_task", "network_connectivity")[0]
    assert entry.recovered_execution is False
    assert entry.terminal_status == RecoveryStatus.FAILED_RETRIES_EXHAUSTED
    assert entry.attempt_count == 3


# ===========================================================================
# Scenario 4: Permanent failure with no inappropriate retry
# ===========================================================================

def test_scenario_4_permanent_failure_no_retry(eval_env):
    """Scenario 4: Permanent exception (TypeError) stops immediately without retry."""
    orchestrator, _, sleeper, _, memory = eval_env

    attempt_count = 0
    def logic_bug():
        nonlocal attempt_count
        attempt_count += 1
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")

    task = Task(
        name="scenario_4_task",
        callable=logic_bug,
        is_idempotent=True,
        max_retries=3,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_PERMANENT
    assert result.is_success is False
    assert result.total_attempts == 1
    assert attempt_count == 1
    assert len(sleeper.delays) == 0

    # Invariant: No retry was attempted, so no recovery memory entry is recorded
    assert len(memory) == 0


# ===========================================================================
# Scenario 5: Non-idempotent task failure with no retry
# ===========================================================================

def test_scenario_5_non_idempotent_task_failure(eval_env):
    """Scenario 5: Non-idempotent task failure stops immediately even if transient."""
    orchestrator, _, sleeper, _, memory = eval_env

    attempt_count = 0
    def non_idempotent_charge():
        nonlocal attempt_count
        attempt_count += 1
        raise ConnectionError("Socket reset during credit card capture")

    task = Task(
        name="scenario_5_task",
        callable=non_idempotent_charge,
        is_idempotent=False,
        max_retries=3,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_NON_IDEMPOTENT
    assert result.is_success is False
    assert result.total_attempts == 1
    assert attempt_count == 1
    assert len(sleeper.delays) == 0

    # Invariant: Non-idempotent tasks must never retry; no recovery memory recorded
    assert len(memory) == 0


# ===========================================================================
# Scenario 6: Timeout when safe_to_retry_on_timeout is False
# ===========================================================================

def test_scenario_6_unsafe_timeout(eval_env):
    """Scenario 6: Unsafe timeout stops immediately because worker threads cannot be killed."""
    orchestrator, _, sleeper, _, memory = eval_env
    finish_event = threading.Event()

    def slow_worker():
        finish_event.wait(timeout=0.3)
        return "late_result"

    task = Task(
        name="scenario_6_task",
        callable=slow_worker,
        timeout_seconds=0.01,
        safe_to_retry_on_timeout=False,
        max_retries=2,
    )

    try:
        result = orchestrator.execute(task)

        assert result.status == RecoveryStatus.FAILED_TIMEOUT_UNSAFE
        assert result.is_success is False
        assert result.total_attempts == 1
        assert result.final_result.failure_type == FailureType.TIMEOUT
        assert len(sleeper.delays) == 0

        # Invariant: Unsafe timeout stops immediately; no recovery attempted
        assert len(memory) == 0
    finally:
        finish_event.set()


# ===========================================================================
# Scenario 7: Timeout when safe_to_retry_on_timeout is True
# ===========================================================================

def test_scenario_7_safe_timeout_retry_recovery():
    """Scenario 7: Safe timeout retry recovers when caller explicitly authorizes timeout retry.

    Honest Worker-Thread Treatment:
    The worker thread from attempt 1 cannot be killed, but uses an Event so it
    terminates cleanly after the test without leaking resources.
    """
    runner = TaskRunner(max_workers=4)
    memory = RecoveryMemory()
    orchestrator = RecoveryOrchestrator(runner=runner, memory=memory)

    attempt_1_event = threading.Event()
    call_count = 0

    def timeout_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Attempt 1 times out
            attempt_1_event.wait(timeout=0.4)
            return "stale_output"
        # Attempt 2 completes immediately
        return "attempt_2_success"

    task = Task(
        name="scenario_7_task",
        callable=timeout_then_succeed,
        timeout_seconds=0.02,
        safe_to_retry_on_timeout=True,  # Explicitly authorized!
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.01,
    )

    try:
        result = orchestrator.execute(task)

        assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
        assert result.is_success is True
        assert result.total_attempts == 2
        assert result.final_result.output == "attempt_2_success"

        # Invariant: Attempt 1 had TIMEOUT failure, Attempt 2 succeeded
        assert result.attempts[0].task_result.failure_type == FailureType.TIMEOUT
        assert result.attempts[1].task_result.outcome == TaskOutcome.SUCCESS

        # Invariant: Memory recorded 1 entry documenting timeout recovery
        assert len(memory) == 1
        entry = memory.find_similar("scenario_7_task", "operation_timeout")[0]
        assert entry.failure_type == FailureType.TIMEOUT
        assert entry.recovered_execution is True
        assert entry.terminal_status == RecoveryStatus.RECOVERED_VIA_RETRY
    finally:
        # Guarantee worker thread is unblocked in all exit paths (success, assertion error, exception)
        attempt_1_event.set()
        orchestrator.close()
        # TaskRunner.close() calls shutdown(wait=False); in this test cleanup we access
        # runner._executor.shutdown(wait=True) to synchronously join the unblocked worker thread.
        runner._executor.shutdown(wait=True)


# ===========================================================================
# Scenario 8: Recovery-budget exhaustion
# ===========================================================================

def test_scenario_8_recovery_budget_exhaustion(eval_env):
    """Scenario 8: Wall-clock budget bounds total recovery time and blocks next retry."""
    orchestrator, clock, sleeper, _, memory = eval_env

    def slow_failing_work():
        clock.advance(0.6)  # Takes 0.6s out of 1.0s budget
        raise ConnectionError("Transient failure")

    task = Task(
        name="scenario_8_task",
        callable=slow_failing_work,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=1.0,
        backoff_base_delay=0.5,  # 0.6s + 0.5s = 1.1s > 1.0s budget
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.is_success is False
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0  # Pre-sleep check prevented sleep

    # Invariant: Terminal budget exhaustion after active recovery decision is recorded
    assert len(memory) == 1
    entry = memory.find_similar("scenario_8_task", "network_connectivity")[0]
    assert entry.task_name == "scenario_8_task"
    assert entry.terminal_status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert entry.recovered_execution is False
    assert entry.attempt_count == 1
    assert entry.recovery_action == RecoveryAction.RETRY
    assert entry.root_cause == "network_connectivity"
    assert entry.fallback_task_name is None


def test_scenario_8_contrasting_budget_exhausted_before_recovery_decision(eval_env):
    """Contrasting Test: When attempt 1 consumes the entire recovery budget before
    any retry or backoff is proposed, no recovery memory entry is recorded.
    """
    orchestrator, clock, sleeper, _, memory = eval_env

    def initial_attempt_exceeds_budget():
        clock.advance(1.5)  # 1.5s > 1.0s budget during attempt 1
        raise ConnectionError("Timeout during long initial attempt")

    task = Task(
        name="scenario_8_contrasting_task",
        callable=initial_attempt_exceeds_budget,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=1.0,
        backoff_base_delay=0.1,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.is_success is False
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0

    # Invariant: No retry was ever formulated or proposed because the budget was already
    # exhausted before retry evaluation; therefore zero memory entries are recorded.
    assert len(memory) == 0


# ===========================================================================
# Scenario 9: Successful execution with VERIFIED output
# ===========================================================================

def test_scenario_9_successful_execution_verified(eval_env):
    """Scenario 9: Execution succeeds and satisfies semantic output verifier."""
    orchestrator, _, _, _, memory = eval_env

    task = Task(
        name="scenario_9_task",
        callable=lambda: {"auth": True, "token": "abc-123"},
        verifier=lambda out: isinstance(out, dict) and out.get("auth") is True,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is True
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.verification.error is None
    assert len(memory) == 0


# ===========================================================================
# Scenario 10: Successful execution with UNVERIFIED output
# ===========================================================================

def test_scenario_10_successful_execution_unverified(eval_env):
    """Scenario 10: Execution succeeds without a configured verifier (UNVERIFIED)."""
    orchestrator, _, _, _, memory = eval_env

    task = Task(
        name="scenario_10_task",
        callable=lambda: "completed_output",
        verifier=None,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification.status == VerificationStatus.UNVERIFIED
    assert len(memory) == 0


# ===========================================================================
# Scenario 11: Successful execution with FAILED verification
# ===========================================================================

def test_scenario_11_successful_execution_failed_verification(eval_env):
    """Scenario 11: Execution succeeds, but semantic verifier returns False."""
    orchestrator, _, _, _, memory = eval_env

    task = Task(
        name="scenario_11_task",
        callable=lambda: {"status": "declined"},
        verifier=lambda out: out.get("status") == "approved",
    )
    result = orchestrator.execute(task)

    # Invariant: Execution completed, but output verification failed
    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification.status == VerificationStatus.FAILED
    assert result.total_attempts == 1
    assert len(memory) == 0


# ===========================================================================
# Scenario 12: Successful execution with INCONCLUSIVE verification
# ===========================================================================

def test_scenario_12_successful_execution_inconclusive_verification(eval_env):
    """Scenario 12: Execution succeeds, but verifier raises an unexpected exception."""
    orchestrator, _, _, _, memory = eval_env

    def crashing_verifier(out: Any) -> bool:
        raise KeyError("missing_expected_verification_field")

    task = Task(
        name="scenario_12_task",
        callable=lambda: {"irrelevant": 1},
        verifier=crashing_verifier,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification.status == VerificationStatus.INCONCLUSIVE
    assert "missing_expected_verification_field" in (result.verification.error or "")
    assert result.total_attempts == 1
    assert len(memory) == 0


# ===========================================================================
# Scenario 13: Fallback success
# ===========================================================================

def test_scenario_13_fallback_success(eval_env):
    """Scenario 13: Permanent primary failure successfully recovers via designated fallback."""
    orchestrator, _, _, _, memory = eval_env

    fallback_called = False
    def fallback_work():
        nonlocal fallback_called
        fallback_called = True
        return "fallback_ok_payload"

    fallback_task = Task(
        name="scenario_13_fallback",
        callable=fallback_work,
        verifier=lambda out: out == "fallback_ok_payload",
    )
    primary_task = Task(
        name="scenario_13_primary",
        callable=lambda: (_ for _ in ()).throw(TypeError("Primary logic error")),
        fallback=fallback_task,
    )
    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    assert result.is_success is True
    assert result.is_verified is True
    assert result.total_attempts == 2
    assert fallback_called is True

    # Invariant: Primary task failure attributed with fallback task name
    assert len(memory) == 1
    entry = memory.find_similar("scenario_13_primary", "type_mismatch")[0]
    assert entry.task_name == "scenario_13_primary"
    assert entry.fallback_task_name == "scenario_13_fallback"
    assert entry.recovery_action == RecoveryAction.FALLBACK
    assert entry.recovered_execution is True
    assert entry.verification_status == VerificationStatus.VERIFIED


# ===========================================================================
# Scenario 14: Fallback failure
# ===========================================================================

def test_scenario_14_fallback_failure(eval_env):
    """Scenario 14: Primary failure followed by fallback failure escalates without infinite chaining."""
    orchestrator, _, _, _, memory = eval_env

    fallback_task = Task(
        name="scenario_14_fallback",
        callable=lambda: (_ for _ in ()).throw(ConnectionError("Fallback service unreachable")),
    )
    primary_task = Task(
        name="scenario_14_primary",
        callable=lambda: (_ for _ in ()).throw(TypeError("Primary crash")),
        fallback=fallback_task,
    )
    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.ESCALATED
    assert result.is_success is False
    assert result.is_verified is False
    assert result.total_attempts == 2

    # Invariant: Both primary and fallback failed; terminal escalation recorded
    assert len(memory) == 1
    entry = memory.find_similar("scenario_14_primary", "type_mismatch")[0]
    assert entry.task_name == "scenario_14_primary"
    assert entry.fallback_task_name == "scenario_14_fallback"
    assert entry.recovered_execution is False
    assert entry.terminal_status == RecoveryStatus.ESCALATED


# ===========================================================================
# Scenario 15: Memory-recording failure without result corruption
# ===========================================================================

def test_scenario_15_memory_recording_failure_does_not_corrupt_result(eval_env):
    """Scenario 15: Memory.remember exception is swallowed and logged without affecting RecoveryResult."""
    orchestrator, _, _, _, _ = eval_env

    class BrokenMemory(RecoveryMemory):
        def remember(self, entry):
            raise IOError("Simulated storage write fault")

    orchestrator._memory = BrokenMemory()

    attempt_count = 0
    def flaky_work():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise ConnectionError("Transient reset")
        return "success_after_glitch"

    task = Task("scenario_15_task", callable=flaky_work, is_idempotent=True, max_retries=2)
    result = orchestrator.execute(task)

    # Invariant: Memory failure does not prevent return or corrupt RecoveryResult
    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.final_result.output == "success_after_glitch"
    assert result.total_attempts == 2


# ===========================================================================
# Metric & Outcome Distinction Invariants
# ===========================================================================

def test_metric_distinction_execution_vs_verified_success(eval_env):
    """Verifies that execution success count is strictly an upper bound on verified success count.

    Demonstrates:
    - Execution Success Count >= Verified Success Count.
    - UNVERIFIED, FAILED, and INCONCLUSIVE verification outcomes are NEVER counted as verified successes.
    """
    orchestrator, _, _, _, _ = eval_env

    tasks = [
        # 1. Verified success (Execution=True, Verified=True)
        Task("t_ver", lambda: 1, verifier=lambda x: x == 1),
        # 2. Unverified success (Execution=True, Verified=False)
        Task("t_unver", lambda: 1, verifier=None),
        # 3. Failed verification (Execution=True, Verified=False)
        Task("t_fail_ver", lambda: 1, verifier=lambda x: x == 99),
        # 4. Inconclusive verification (Execution=True, Verified=False)
        Task("t_inconcl", lambda: 1, verifier=lambda x: (_ for _ in ()).throw(ValueError("boom"))),
        # 5. Failed execution (Execution=False, Verified=False)
        Task("t_fail_exec", lambda: (_ for _ in ()).throw(KeyError("missing"))),
    ]

    results: list[RecoveryResult] = [orchestrator.execute(t) for t in tasks]

    execution_successes = sum(1 for r in results if r.is_success)
    verified_successes = sum(1 for r in results if r.is_verified)

    assert execution_successes == 4  # Tasks 1, 2, 3, 4 executed cleanly
    assert verified_successes == 1   # ONLY Task 1 is verified

    # Mathematical Invariant
    assert verified_successes <= execution_successes


def test_metric_distinction_recovery_success_vs_verified_recovery(eval_env):
    """Verifies that recovering execution is distinct from verified recovery.

    Demonstrates:
    - A task that recovers execution but fails verification counts as a Recovery Success,
      but MUST NOT count as a Verified Recovery.
    """
    orchestrator, _, _, _, _ = eval_env

    # Task A: Recovers execution and PASSES verification
    cnt_a = 0
    def work_a():
        nonlocal cnt_a
        cnt_a += 1
        if cnt_a == 1:
            raise ConnectionError("err")
        return {"code": 200}
    task_a = Task("rec_ver", work_a, is_idempotent=True, max_retries=2, verifier=lambda o: o.get("code") == 200)

    # Task B: Recovers execution but FAILS verification
    cnt_b = 0
    def work_b():
        nonlocal cnt_b
        cnt_b += 1
        if cnt_b == 1:
            raise ConnectionError("err")
        return {"code": 500}
    task_b = Task("rec_unver", work_b, is_idempotent=True, max_retries=2, verifier=lambda o: o.get("code") == 200)

    res_a = orchestrator.execute(task_a)
    res_b = orchestrator.execute(task_b)

    # Both tasks successfully recovered execution!
    assert res_a.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert res_b.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert res_a.is_success is True
    assert res_b.is_success is True

    # But only Task A achieved verified recovery
    assert res_a.is_verified is True
    assert res_b.is_verified is False
    assert res_b.verification.status == VerificationStatus.FAILED

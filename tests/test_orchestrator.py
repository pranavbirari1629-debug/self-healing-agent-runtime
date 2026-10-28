"""Unit tests for RecoveryOrchestrator in shar.orchestrator."""

import dataclasses
import time
from typing import Any
import pytest

from shar.detector import FailureDetector
from shar.diagnosis import DiagnosisEngine
from shar.models import (
    ConfidenceLevel,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskOutcome,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)
from shar.memory import RecoveryMemory, RecoveryMemoryEntry
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
def test_env():
    """Provides an orchestrator configured with mock clock and sleep function."""
    clock = MockClock(start=100.0)
    sleeper = MockSleeper(clock=clock)
    runner = TaskRunner()
    orchestrator = RecoveryOrchestrator(
        runner=runner,
        sleep_fn=sleeper.sleep,
        time_fn=clock.time,
    )
    yield orchestrator, clock, sleeper, runner
    orchestrator.close()
    runner.close()


# ---- 1. Immediate success ----

def test_immediate_success(test_env):
    """Verifies that a task succeeding on the first attempt completes with SUCCEEDED_FIRST_ATTEMPT."""
    orchestrator, _, sleeper, _ = test_env
    task = Task("simple_task", lambda: "ok")

    result = orchestrator.execute(task)

    assert result.is_success is True
    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.total_attempts == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].attempt_number == 1
    assert result.attempts[0].task_result.output == "ok"
    assert result.final_result.output == "ok"
    assert len(sleeper.delays) == 0  # No sleep invoked


# ---- 2. Transient failure followed by successful retry ----

def test_transient_failure_recovered_via_retry(test_env):
    """Verifies that an idempotent task failing once then succeeding is recovered via retry."""
    orchestrator, _, sleeper, _ = test_env
    call_count = 0

    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Connection lost")
        return "data_received"

    task = Task(
        name="flaky_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.1,
    )

    result = orchestrator.execute(task)

    assert result.is_success is True
    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.total_attempts == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].attempt_number == 1
    assert result.attempts[0].task_result.is_success is False
    assert result.attempts[1].attempt_number == 2
    assert result.attempts[1].task_result.is_success is True
    assert result.final_result.output == "data_received"
    assert sleeper.delays == [0.1]


# ---- 3. Permanent failure without retry ----

def test_permanent_failure_without_retry(test_env):
    """Verifies that permanent errors (TypeError) immediately stop with FAILED_PERMANENT."""
    orchestrator, _, sleeper, _ = test_env

    def bug_work():
        raise TypeError("unsupported operand")

    task = Task("bug_task", callable=bug_work, is_idempotent=True, max_retries=3)

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_PERMANENT
    assert result.total_attempts == 1
    assert len(result.attempts) == 1
    assert len(sleeper.delays) == 0


# ---- 4. Unknown failure without retry ----

def test_unknown_failure_without_retry(test_env):
    """Verifies that custom unrecognized exceptions stop with FAILED_UNKNOWN."""
    orchestrator, _, sleeper, _ = test_env

    class CustomUnknownException(Exception):
        pass

    def custom_fail():
        raise CustomUnknownException("unexpected failure")

    task = Task("unknown_task", callable=custom_fail, is_idempotent=True)

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_UNKNOWN
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0


# ---- 5. Retry limit exhaustion ----

def test_retry_exhaustion(test_env):
    """Verifies that failing continuously halts at max_retries with FAILED_RETRIES_EXHAUSTED."""
    orchestrator, _, sleeper, _ = test_env

    def persistent_failure():
        raise OSError("Resource temporarily unavailable")

    # max_retries=2: 1 initial attempt + 2 retries = 3 attempts total
    task = Task(
        name="persistent_task",
        callable=persistent_failure,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.05,
        backoff_factor=2.0,
    )

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_RETRIES_EXHAUSTED
    assert result.total_attempts == 3
    assert len(result.attempts) == 3
    assert [a.attempt_number for a in result.attempts] == [1, 2, 3]
    assert sleeper.delays == [0.05, 0.1]


# ---- 6. Backoff invoked through injected sleep function ----

def test_backoff_delays_recorded(test_env):
    """Verifies exponential delays passed to sleep_fn: base, base*factor, base*factor^2."""
    orchestrator, _, sleeper, _ = test_env

    def fail_always():
        raise ConnectionResetError("Reset by peer")

    task = Task(
        name="backoff_test",
        callable=fail_always,
        is_idempotent=True,
        max_retries=3,
        backoff_base_delay=0.1,
        backoff_factor=2.0,
        backoff_max_delay=1.0,
    )

    result = orchestrator.execute(task)

    assert result.total_attempts == 4
    # Expected backoff delays: 0.1, 0.2, 0.4
    assert pytest.approx(sleeper.delays) == [0.1, 0.2, 0.4]


# ---- 7. Budget exhaustion before retry delay ----

def test_budget_exhaustion_before_retry(test_env):
    """Verifies that exceeding recovery budget causes FAILED_BUDGET_EXCEEDED."""
    orchestrator, clock, sleeper, _ = test_env

    def slow_failing_work():
        # Simulate work that ran for 2.0s
        clock.advance(2.0)
        raise ConnectionError("timeout")

    task = Task(
        name="budgeted_work",
        callable=slow_failing_work,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=1.5,  # 1.5s total budget
        backoff_base_delay=0.1,
    )

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0  # Does not sleep because budget is already exhausted


# ---- 8. Budget exhaustion after backoff but before another attempt ----

def test_budget_exhaustion_after_backoff(test_env):
    """Verifies that if sleeping consumes remaining budget, next attempt is prevented."""
    orchestrator, clock, sleeper, _ = test_env

    def failing_work():
        clock.advance(0.8)
        raise ConnectionError("network blip")

    # Budget 2.0s. First attempt takes 0.8s. Delay is 0.5s.
    # Sleeper advances clock by 0.5s -> elapsed = 1.3s.
    # Second attempt takes 0.8s -> elapsed = 2.1s > 2.0s!
    task = Task(
        name="budget_tight",
        callable=failing_work,
        is_idempotent=True,
        max_retries=5,
        recovery_budget_seconds=2.0,
        backoff_base_delay=0.5,
    )

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.total_attempts == 2


# ---- 9. Non-idempotent task is not retried ----

def test_non_idempotent_task_not_retried(test_env):
    """Verifies that non-idempotent tasks halt immediately with FAILED_NON_IDEMPOTENT."""
    orchestrator, _, sleeper, _ = test_env

    def transfer_funds():
        raise ConnectionError("connection dropped mid-transfer")

    task = Task("bank_transfer", callable=transfer_funds, is_idempotent=False, max_retries=3)

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_NON_IDEMPOTENT
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0


# ---- 10. Unsafe timeout is not retried ----

def test_unsafe_timeout_not_retried(test_env):
    """Verifies that TIMEOUT failure without safe_to_retry_on_timeout stops with FAILED_TIMEOUT_UNSAFE."""
    orchestrator, _, sleeper, _ = test_env

    # We mock runner behavior by returning a TIMEOUT failure result
    class MockTimeoutRunner:
        def run(self, task: Task, correlation_id: str | None = None) -> TaskResult:
            return TaskResult(
                task_name=task.name,
                outcome=TaskOutcome.FAILURE,
                failure_type=FailureType.TIMEOUT,
                exception_type="TimeoutError",
                exception_message="Task timed out after 0.05s",
                correlation_id=correlation_id or "cid",
            )

    custom_orchestrator = RecoveryOrchestrator(
        runner=MockTimeoutRunner(),  # type: ignore[arg-type]
        sleep_fn=sleeper.sleep,
    )

    task = Task("hanging_op", callable=lambda: None, is_idempotent=True, safe_to_retry_on_timeout=False)

    result = custom_orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.FAILED_TIMEOUT_UNSAFE
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0


# ---- 11. Fallback success after primary failure ----

def test_fallback_success(test_env):
    """Verifies that when primary task fails, designated fallback is executed and succeeds."""
    orchestrator, _, _, _ = test_env

    def primary_fail():
        raise OSError("Primary storage read error")

    def fallback_ok():
        return "secondary_storage_data"

    fallback_task = Task("secondary_storage", callable=fallback_ok)
    primary_task = Task(
        name="primary_storage",
        callable=primary_fail,
        is_idempotent=True,
        max_retries=1,
        fallback=fallback_task,
    )

    result = orchestrator.execute(primary_task)

    assert result.is_success is True
    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    assert result.final_result.output == "secondary_storage_data"
    # Attempts: attempt 1 (primary fail), attempt 2 (primary fail retries exhausted), attempt 3 (fallback success)
    assert result.total_attempts == 3
    assert result.attempts[0].task_name == "primary_storage"
    assert result.attempts[1].task_name == "primary_storage"
    assert result.attempts[2].task_name == "secondary_storage"


# ---- 12. Fallback failure with accurate history ----

def test_fallback_failure_escalates(test_env):
    """Verifies that if fallback also fails, orchestrator terminates with ESCALATED and complete history."""
    orchestrator, _, _, _ = test_env

    def primary_fail():
        raise TypeError("Primary permanent bug")

    def fallback_fail():
        raise ConnectionError("Secondary down as well")

    fallback_task = Task("secondary_source", callable=fallback_fail)
    primary_task = Task(
        name="primary_source",
        callable=primary_fail,
        is_idempotent=True,
        fallback=fallback_task,
    )

    result = orchestrator.execute(primary_task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.ESCALATED
    assert result.total_attempts == 2
    assert result.attempts[0].task_name == "primary_source"
    assert result.attempts[0].task_result.exception_type == "TypeError"
    assert result.attempts[1].task_name == "secondary_source"
    assert result.attempts[1].task_result.exception_type == "ConnectionError"


# ---- 13. Fallback depth is bounded ----

def test_fallback_depth_is_bounded(test_env):
    """Verifies that nested fallbacks beyond max_fallback_depth=1 are halted with ESCALATED."""
    orchestrator, _, _, _ = test_env

    # 3-level chain: tertiary -> secondary -> primary
    tertiary = Task("tertiary", callable=lambda: "deep_ok")
    secondary = Task("secondary", callable=lambda: (_ for _ in ()).throw(ValueError("second fail")), fallback=tertiary)
    primary = Task("primary", callable=lambda: (_ for _ in ()).throw(ValueError("first fail")), fallback=secondary)

    result = orchestrator.execute(primary)

    # Secondary was executed (depth 1), tertiary was refused (depth 2 > 1) -> ESCALATED
    assert result.is_success is False
    assert result.status == RecoveryStatus.ESCALATED
    assert result.total_attempts == 2
    assert [a.task_name for a in result.attempts] == ["primary", "secondary"]


# ---- 14. Analysis component unexpectedly raising an exception ----

def test_analysis_component_crash_handled_conservatively(test_env):
    """Verifies that unexpected errors in detector/diagnosis do not trigger blind retries."""
    _, clock, sleeper, runner = test_env

    class CrashingDetector(FailureDetector):
        def classify(self, result: TaskResult) -> Any:
            raise RuntimeError("Internal detector corrupted")

    orchestrator = RecoveryOrchestrator(
        runner=runner,
        detector=CrashingDetector(),
        sleep_fn=sleeper.sleep,
        time_fn=clock.time,
    )

    task = Task("unlucky_task", callable=lambda: (_ for _ in ()).throw(ConnectionError("fail")), is_idempotent=True)

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.status == RecoveryStatus.ESCALATED
    assert result.total_attempts == 1
    assert "Internal detector corrupted" in (result.attempts[0].decision.reason or "")
    assert len(sleeper.delays) == 0


# ---- 15. Correlation ID preservation ----

def test_correlation_id_preservation(test_env):
    """Verifies that custom correlation_id is preserved on result and all attempt records."""
    orchestrator, _, _, _ = test_env
    custom_cid = "trace-uuid-abcdef-123456"

    count = 0
    def two_step_task():
        nonlocal count
        count += 1
        if count == 1:
            raise ConnectionError("first fail")
        return "success"

    task = Task("trace_task", callable=two_step_task, is_idempotent=True, max_retries=2)

    result = orchestrator.execute(task, correlation_id=custom_cid)

    assert result.correlation_id == custom_cid
    assert len(result.attempts) == 2
    for attempt in result.attempts:
        assert attempt.task_result.correlation_id == custom_cid


# ---- 16. Correct attempt numbering and total_attempts ----

def test_attempt_numbering_monotonically_increases(test_env):
    """Verifies that attempt_number in AttemptRecord strictly follows 1, 2, 3..."""
    orchestrator, _, _, _ = test_env

    def fail_until_3():
        nonlocal count
        count += 1
        if count < 3:
            raise ConnectionError("temporary")
        return "finally"

    count = 0
    task = Task("counting_task", callable=fail_until_3, is_idempotent=True, max_retries=3)

    result = orchestrator.execute(task)

    assert result.total_attempts == 3
    numbers = [a.attempt_number for a in result.attempts]
    assert numbers == [1, 2, 3]


# ---- 17. Correct final status and final TaskResult ----

def test_final_status_matches_task_result(test_env):
    """Verifies that final_result.is_success is consistent with is_success property."""
    orchestrator, _, _, _ = test_env
    task = Task("good_task", callable=lambda: "clean_return")

    result = orchestrator.execute(task)

    assert result.is_success == result.final_result.is_success
    assert result.final_result.outcome == TaskOutcome.SUCCESS
    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT


# ---- 18. No real sleeping in unit tests ----

def test_no_real_sleeping(test_env):
    """Verifies that mock sleeper intercepts all backoff pauses without wall-clock delays."""
    orchestrator, _, sleeper, _ = test_env

    # 3 retries with 10.0s delays each
    task = Task(
        name="long_delay_task",
        callable=lambda: (_ for _ in ()).throw(ConnectionError("fail")),
        is_idempotent=True,
        max_retries=3,
        backoff_base_delay=10.0,
        backoff_factor=1.0,
        backoff_max_delay=10.0,
    )

    result = orchestrator.execute(task)

    assert result.total_attempts == 4
    # All 3 sleep requests were recorded by mock sleeper without blocking test execution
    assert sleeper.delays == [10.0, 10.0, 10.0]


# ---- 19. Fallback budget inheritance regression tests ----

def test_fallback_budget_inheritance_exhausted_before_fallback(test_env):
    """Verifies that root task recovery budget is enforced before transitioning to fallback."""
    orchestrator, clock, sleeper, _ = test_env

    fallback_executed = False

    def fallback_work():
        nonlocal fallback_executed
        fallback_executed = True
        return "fallback_result"

    def primary_fail():
        # Consume the entire 2.0s recovery budget during the primary attempt
        clock.advance(2.0)
        raise OSError("Storage failure")

    fallback_task = Task(name="fallback_task", callable=fallback_work, recovery_budget_seconds=None)
    primary_task = Task(
        name="primary_task",
        callable=primary_fail,
        is_idempotent=True,
        max_retries=0,
        recovery_budget_seconds=2.0,
        fallback=fallback_task,
    )

    custom_cid = "regress-cid-fallback-budget-001"
    result = orchestrator.execute(primary_task, correlation_id=custom_cid)

    # 1. Fallback callable is never executed
    assert fallback_executed is False
    # 2. Budget exceeded failure status
    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.is_success is False
    # 3. Primary attempt remains in returned history
    assert len(result.attempts) == 1
    assert result.attempts[0].task_name == "primary_task"
    assert result.attempts[0].task_result.exception_type == "OSError"
    # 4. total_attempts equals number of recorded executions
    assert result.total_attempts == 1
    # 5. Correlation ID preserved
    assert result.correlation_id == custom_cid
    assert result.attempts[0].task_result.correlation_id == custom_cid
    # 6. Zero sleep calls
    assert len(sleeper.delays) == 0


def test_fallback_budget_inheritance_exhausted_at_attempt_boundary(test_env):
    """Verifies that root budget halts fallback execution at attempt boundary when fallback budget is None."""
    orchestrator, clock, sleeper, _ = test_env

    fallback_executed = False

    def fallback_work():
        nonlocal fallback_executed
        fallback_executed = True
        return "fallback_result"

    def primary_fail():
        raise OSError("Primary error")

    fallback_task = Task(name="unbudgeted_fallback", callable=fallback_work, recovery_budget_seconds=None)
    primary_task = Task(
        name="budgeted_primary",
        callable=primary_fail,
        is_idempotent=True,
        max_retries=0,
        recovery_budget_seconds=2.0,
        fallback=fallback_task,
    )

    # Simulate clock progression:
    # 1. Start: 100.0s (elapsed 0.0s)
    # 2. Attempt 1 ends: 101.5s (elapsed 1.5s)
    # 3. Pre-fallback check: 101.8s (elapsed 1.8s < 2.0s budget -> transition permitted)
    # 4. Attempt 2 start: 102.5s (elapsed 2.5s >= 2.0s budget -> attempt blocked)
    time_reads = 0
    def stepped_time_fn() -> float:
        nonlocal time_reads
        time_reads += 1
        if time_reads >= 7:
            return 102.5
        if time_reads >= 4:
            return 101.8
        if time_reads >= 2:
            return 101.5
        return 100.0

    orchestrator._time_fn = stepped_time_fn

    result = orchestrator.execute(primary_task)

    assert fallback_executed is False
    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.total_attempts == 1
    assert [a.task_name for a in result.attempts] == ["budgeted_primary"]


def test_budget_exhaustion_during_backoff_boundary_blocks_next_attempt(test_env):
    """Verifies that when backoff sleep lands exactly on or past budget, next attempt is blocked."""
    orchestrator, clock, sleeper, _ = test_env

    attempts_started = 0

    def failing_work():
        nonlocal attempts_started
        attempts_started += 1
        clock.advance(1.0)  # Attempt 1 takes 1.0s out of 2.0s budget
        raise ConnectionError("Transient network failure")

    task = Task(
        name="boundary_task",
        callable=failing_work,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=2.0,
        backoff_base_delay=0.5,
    )

    # Inject a sleeper that advances clock by 1.0s instead of 0.5s, bringing total elapsed to exactly 2.0s
    def sleep_advance_past_budget(delay: float) -> None:
        sleeper.delays.append(delay)
        clock.advance(1.0)  # Total elapsed becomes 1.0 + 1.0 = 2.0s >= 2.0s

    orchestrator._sleep_fn = sleep_advance_past_budget

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.is_success is False
    # Attempt 2 must never have started
    assert attempts_started == 1
    assert result.total_attempts == 1
    assert len(result.attempts) == 1


# ---- 20. Real TaskRunner timeout integration test ----

def test_real_taskrunner_timeout_integration():
    """Integration test using real TaskRunner, ThreadPoolExecutor, and real timeout path."""
    import threading

    runner = TaskRunner(max_workers=2)
    orchestrator = RecoveryOrchestrator(runner=runner)
    finish_event = threading.Event()

    def slow_blocking_worker():
        # Bounded wait with event fallback to guarantee clean thread exit without leaking
        finish_event.wait(timeout=0.4)
        return "late_output"

    fallback_called = False
    def fallback_worker():
        nonlocal fallback_called
        fallback_called = True
        return "fallback_output"

    fallback_task = Task("fallback_after_timeout", callable=fallback_worker)
    primary_task = Task(
        name="real_timeout_task",
        callable=slow_blocking_worker,
        timeout_seconds=0.05,
        safe_to_retry_on_timeout=False,  # Unsafe timeout: must never retry or fallback
        max_retries=2,
        fallback=fallback_task,
    )

    try:
        custom_cid = "real-timeout-trace-001"
        result = orchestrator.execute(primary_task, correlation_id=custom_cid)

        # 1. Real timeout reported through runner/orchestrator path
        assert result.is_success is False
        assert result.status == RecoveryStatus.FAILED_TIMEOUT_UNSAFE
        # 2. Unsafe timeout does not trigger retry or fallback
        assert result.total_attempts == 1
        assert len(result.attempts) == 1
        assert fallback_called is False
        # 3. Returned result records failed attempt accurately
        assert result.final_result.outcome == TaskOutcome.FAILURE
        assert result.final_result.failure_type == FailureType.TIMEOUT
        assert result.final_result.exception_type == "TimeoutError"
        assert "timed out after 0.05s" in (result.final_result.exception_message or "")
        # 4. Honest documentation that thread is not terminated
        assert "not terminated" in (result.final_result.exception_message or "")
        assert result.correlation_id == custom_cid
    finally:
        # Guarantee worker thread unblocks and terminates cleanly
        finish_event.set()
        orchestrator.close()
        runner.close()


# ===========================================================================
# Phase 4 Step 2: OutputVerifier Integration Tests
# ===========================================================================

# ---- 21. First-attempt success with verifier returning True ----

def test_first_attempt_success_verified(test_env):
    """Verifies that a first-attempt success with passing verifier yields VERIFIED and is_verified=True."""
    orchestrator, _, _, _ = test_env

    def schema_verifier(data: Any) -> bool:
        return isinstance(data, dict) and data.get("valid") is True

    task = Task(
        name="verified_success_task",
        callable=lambda: {"valid": True, "count": 42},
        verifier=schema_verifier,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is True
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.verification.is_verified is True
    assert result.verification.verifier_name == "schema_verifier"
    assert result.verification.error is None
    assert result.total_attempts == 1


# ---- 22. First-attempt success with verifier returning False ----

def test_first_attempt_success_verification_failed(test_env):
    """Verifies that execution success with failing verifier preserves SUCCEEDED_FIRST_ATTEMPT but is_verified=False."""
    orchestrator, _, sleeper, _ = test_env

    def strict_verifier(data: Any) -> bool:
        return isinstance(data, dict) and data.get("status") == "approved"

    task = Task(
        name="semantic_failure_task",
        callable=lambda: {"status": "rejected", "reason": "insufficient_balance"},
        verifier=strict_verifier,
        is_idempotent=True,
        max_retries=3,
    )

    result = orchestrator.execute(task)

    # Execution succeeded, but semantic verification failed
    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.FAILED
    assert result.verification.is_verified is False
    assert result.verification.verifier_name == "strict_verifier"
    assert result.total_attempts == 1
    assert len(sleeper.delays) == 0


# ---- 23. First-attempt success with no verifier configured ----

def test_first_attempt_success_unverified_when_no_verifier(test_env):
    """Verifies that execution without a verifier yields UNVERIFIED and is_verified=False."""
    orchestrator, _, _, _ = test_env

    task = Task(
        name="unverified_task",
        callable=lambda: "result_payload",
        verifier=None,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.UNVERIFIED
    assert result.verification.is_verified is False
    assert result.verification.verifier_name == "none"


# ---- 24. Success with verifier raising Exception ----

def test_success_with_verifier_raising_exception_is_inconclusive(test_env):
    """Verifies that a crashing verifier produces INCONCLUSIVE, captures error, and does not crash orchestrator."""
    orchestrator, _, _, _ = test_env

    def crashing_verifier(data: Any) -> bool:
        raise ValueError("Malformed output structure: field 'x' missing")

    task = Task(
        name="crashing_verifier_task",
        callable=lambda: {"y": 10},
        verifier=crashing_verifier,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.INCONCLUSIVE
    assert result.verification.is_verified is False
    assert result.verification.error == "Malformed output structure: field 'x' missing"
    assert result.verification.verifier_name == "crashing_verifier"


# ---- 25. Verifier returning non-Boolean value ----

@pytest.mark.parametrize("invalid_verdict", ["ok", 1, 0, [True], None, {"ok": True}])
def test_verifier_returning_non_boolean_is_inconclusive(test_env, invalid_verdict):
    """Verifies that non-bool return values (truthy strings, numbers, etc.) result in INCONCLUSIVE."""
    orchestrator, _, _, _ = test_env

    task = Task(
        name="non_bool_verifier_task",
        callable=lambda: "data",
        verifier=lambda _: invalid_verdict,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.INCONCLUSIVE
    assert result.verification.is_verified is False
    assert "UnsupportedReturnType" in (result.verification.error or "")


# ---- 26. Retry-recovered success verified using final result only ----

def test_retry_recovered_success_verified_using_final_result_only(test_env):
    """Verifies that retry-recovered tasks run verification exactly once on the final successful attempt."""
    orchestrator, _, _, _ = test_env

    attempt_counter = 0
    verifier_calls: list[Any] = []

    def flaky_work():
        nonlocal attempt_counter
        attempt_counter += 1
        if attempt_counter == 1:
            raise ConnectionError("Transient reset")
        return {"attempt": 2, "ready": True}

    def my_verifier(data: Any) -> bool:
        verifier_calls.append(data)
        return data.get("ready") is True

    task = Task(
        name="flaky_task_with_verifier",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        verifier=my_verifier,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.is_verified is True
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.total_attempts == 2
    # Verifier must be called exactly once, on the final attempt's output
    assert len(verifier_calls) == 1
    assert verifier_calls[0] == {"attempt": 2, "ready": True}


# ---- 27. Fallback-recovered success verified using fallback task's verifier ----

def test_fallback_recovered_success_verified_using_fallback_verifier(test_env):
    """Verifies that fallback recovery uses the fallback task's verifier, not the primary task's verifier."""
    orchestrator, _, _, _ = test_env

    primary_verifier_called = False
    fallback_verifier_called = False

    def primary_verifier(data: Any) -> bool:
        nonlocal primary_verifier_called
        primary_verifier_called = True
        return False

    def fallback_verifier(data: Any) -> bool:
        nonlocal fallback_verifier_called
        fallback_verifier_called = True
        return isinstance(data, str) and data.startswith("fallback_ok")

    fallback_task = Task(
        name="fallback_task",
        callable=lambda: "fallback_ok_data",
        verifier=fallback_verifier,
    )

    primary_task = Task(
        name="primary_failing_task",
        callable=lambda: (_ for _ in ()).throw(TypeError("Permanent primary crash")),
        verifier=primary_verifier,
        fallback=fallback_task,
    )

    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    assert result.is_success is True
    assert result.is_verified is True
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.VERIFIED
    assert result.verification.verifier_name == "fallback_verifier"
    assert primary_verifier_called is False
    assert fallback_verifier_called is True
    assert result.total_attempts == 2


# ---- 28. Final execution failure does not call semantic verifier ----

def test_final_execution_failure_does_not_call_semantic_verifier(test_env):
    """Verifies that terminal execution failures skip calling semantic verifier and record FAILED."""
    orchestrator, _, _, _ = test_env

    verifier_call_count = 0

    def record_verifier(data: Any) -> bool:
        nonlocal verifier_call_count
        verifier_call_count += 1
        return True

    task = Task(
        name="permanent_failure_task",
        callable=lambda: (_ for _ in ()).throw(KeyError("record_not_found")),
        is_idempotent=True,
        verifier=record_verifier,
    )

    result = orchestrator.execute(task)

    assert result.is_success is False
    assert result.is_verified is False
    assert result.status == RecoveryStatus.FAILED_PERMANENT
    # Semantic verifier must never be called on failed execution
    assert verifier_call_count == 0
    assert result.verification is not None
    assert result.verification.status == VerificationStatus.FAILED
    assert result.verification.is_verified is False
    assert result.verification.verifier_name == "none"
    assert "Failed executions cannot be verified" in result.verification.reason


# ---- 29. Verification failure does not trigger additional retries ----

def test_verification_failure_does_not_trigger_additional_retries(test_env):
    """Verifies that a semantic verification failure does not initiate retry loops."""
    orchestrator, _, sleeper, _ = test_env

    execution_count = 0

    def compute_work():
        nonlocal execution_count
        execution_count += 1
        return "unacceptable_data"

    task = Task(
        name="failing_verifier_retry_task",
        callable=compute_work,
        is_idempotent=True,
        max_retries=3,
        verifier=lambda out: False,
    )

    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification.status == VerificationStatus.FAILED
    # Exactly 1 attempt must have executed
    assert execution_count == 1
    assert result.total_attempts == 1
    assert len(result.attempts) == 1
    assert len(sleeper.delays) == 0


# ---- 30. Verification failure does not trigger fallback ----

def test_verification_failure_does_not_trigger_fallback(test_env):
    """Verifies that a semantic verification failure does not delegate to fallback tasks."""
    orchestrator, _, _, _ = test_env

    fallback_called = False

    def fallback_work():
        nonlocal fallback_called
        fallback_called = True
        return "fallback_result"

    fallback_task = Task("fallback_delegate", callable=fallback_work)

    primary_task = Task(
        name="primary_failing_verification",
        callable=lambda: "completed_output",
        verifier=lambda _: False,
        fallback=fallback_task,
    )

    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert result.is_success is True
    assert result.is_verified is False
    assert result.verification.status == VerificationStatus.FAILED
    # Fallback task must never have executed
    assert fallback_called is False
    assert result.total_attempts == 1
    assert len(result.attempts) == 1


# ---- 31. Execution status and attempt history remain unchanged by verification ----

def test_status_and_attempt_history_unchanged_by_verification(test_env):
    """Verifies that adding a failing or passing verifier leaves Phase 3 status and attempts completely identical."""
    orchestrator, _, _, _ = test_env

    task_plain = Task("task_plain", lambda: 123)
    task_verified = Task("task_verified", lambda: 123, verifier=lambda x: True)
    task_failed_ver = Task("task_failed_ver", lambda: 123, verifier=lambda x: False)

    res_plain = orchestrator.execute(task_plain)
    res_verified = orchestrator.execute(task_verified)
    res_failed_ver = orchestrator.execute(task_failed_ver)

    assert res_plain.status == res_verified.status == res_failed_ver.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert res_plain.is_success == res_verified.is_success == res_failed_ver.is_success is True
    assert res_plain.total_attempts == res_verified.total_attempts == res_failed_ver.total_attempts == 1

    assert res_plain.is_verified is False
    assert res_plain.verification.status == VerificationStatus.UNVERIFIED

    assert res_verified.is_verified is True
    assert res_verified.verification.status == VerificationStatus.VERIFIED

    assert res_failed_ver.is_verified is False
    assert res_failed_ver.verification.status == VerificationStatus.FAILED


# ---- 32. Correlation ID propagation to verification result ----

def test_correlation_id_propagated_to_verification_result(test_env):
    """Verifies that orchestrator correlation IDs cleanly propagate to VerificationResult."""
    orchestrator, _, _, _ = test_env
    custom_cid = "phase4-trace-integration-42"

    task = Task(
        name="trace_task",
        callable=lambda: {"ready": True},
        verifier=lambda out: True,
    )

    result = orchestrator.execute(task, correlation_id=custom_cid)

    assert result.correlation_id == custom_cid
    assert result.final_result.correlation_id == custom_cid
    assert result.verification is not None
    assert result.verification.correlation_id == custom_cid


# ---- 33. Existing callers without verifier remain 100% compatible ----

def test_backwards_compatibility_existing_callers_without_verifier():
    """Verifies that RecoveryOrchestrator can be constructed with legacy positional/keyword args."""
    runner = TaskRunner()
    detector = FailureDetector()
    engine = DiagnosisEngine()
    policy = RecoveryPolicy()

    # Legacy construction without verifier parameter
    orchestrator = RecoveryOrchestrator(
        runner,
        detector,
        engine,
        policy,
    )

    try:
        assert isinstance(orchestrator._verifier, OutputVerifier)
        task = Task("legacy_task", lambda: "ok")
        result = orchestrator.execute(task)

        assert result.is_success is True
        assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
        assert result.verification is not None
        assert result.verification.status == VerificationStatus.UNVERIFIED
        assert result.is_verified is False
    finally:
        orchestrator.close()
        runner.close()


# ===========================================================================
# Phase 4 Step 3B: Recovery Memory Observation Integration Tests
# ===========================================================================

# ---- 34. Orchestrator with memory=None behaves as before ----

def test_orchestrator_with_memory_none_behaves_as_before(test_env):
    """Verifies that an orchestrator without memory executes and recovers normally."""
    orchestrator, _, _, _ = test_env
    assert orchestrator._memory is None

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Temporary glitch")
        return "recovered"

    task = Task("flaky_no_mem", callable=flaky_work, is_idempotent=True, max_retries=2)
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.total_attempts == 2


# ---- 35. Retry-recovered workflow records one entry ----

def test_retry_recovered_workflow_records_one_entry(test_env):
    """Verifies that a retry-recovered workflow records exactly one memory entry."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Temporary glitch")
        return "data_ok"

    task = Task("flaky_mem_task", callable=flaky_work, is_idempotent=True, max_retries=2)
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert len(memory) == 1

    entries = memory.find_similar("flaky_mem_task", "network_connectivity")
    assert len(entries) == 1
    entry = entries[0]

    assert entry.task_name == "flaky_mem_task"
    assert entry.recovery_action == RecoveryAction.RETRY
    assert entry.terminal_status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert entry.attempt_count == 2
    assert entry.recovered_execution is True
    assert entry.fallback_task_name is None


# ---- 36. Retry-recovered verified success is recorded correctly ----

def test_retry_recovered_verified_success_recorded_correctly(test_env):
    """Verifies that a retry-recovered task with passing verifier is recorded with VERIFIED."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network reset")
        return {"count": 42}

    task = Task(
        name="verified_retry_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        verifier=lambda out: isinstance(out, dict) and out.get("count") == 42,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_verified is True
    assert len(memory) == 1

    entry = memory.find_similar("verified_retry_task", "network_connectivity")[0]
    assert entry.recovered_execution is True
    assert entry.verification_status == VerificationStatus.VERIFIED


# ---- 37. Retry-recovered unverified success is not counted as verified success ----

def test_retry_recovered_unverified_success_not_counted_as_verified(test_env):
    """Verifies that a retry-recovered task without verifier is recorded as UNVERIFIED."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network reset")
        return "raw_unverified_string"

    task = Task(
        name="unverified_retry_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        verifier=None,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_verified is False
    assert len(memory) == 1

    entry = memory.find_similar("unverified_retry_task", "network_connectivity")[0]
    assert entry.recovered_execution is True
    assert entry.verification_status == VerificationStatus.UNVERIFIED

    rec = memory.recommend("unverified_retry_task", "network_connectivity")
    assert rec.verified_success_count == 0
    assert rec.unverified_success_count == 1
    assert rec.confidence == ConfidenceLevel.LOW


# ---- 38. Verification FAILED remains distinct from execution failure ----

def test_verification_failed_remains_distinct_from_execution_failure(test_env):
    """Verifies that execution success with verification failure records recovered_execution=True and FAILED."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network reset")
        return {"status": "rejected"}

    task = Task(
        name="failed_verification_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        verifier=lambda out: out.get("status") == "approved",
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.is_verified is False
    assert len(memory) == 1

    entry = memory.find_similar("failed_verification_task", "network_connectivity")[0]
    assert entry.recovered_execution is True
    assert entry.verification_status == VerificationStatus.FAILED

    rec = memory.recommend("failed_verification_task", "network_connectivity")
    assert rec.verified_success_count == 0
    assert rec.failure_count == 1


# ---- 39. Verification INCONCLUSIVE is recorded accurately ----

def test_verification_inconclusive_recorded_accurately(test_env):
    """Verifies that a crashing verifier records recovered_execution=True and INCONCLUSIVE."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network reset")
        return {"data": 123}

    def crashing_verifier(out: Any) -> bool:
        raise ValueError("Malformed verification rule")

    task = Task(
        name="inconclusive_task",
        callable=flaky_work,
        is_idempotent=True,
        max_retries=2,
        verifier=crashing_verifier,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.is_verified is False
    assert len(memory) == 1

    entry = memory.find_similar("inconclusive_task", "network_connectivity")[0]
    assert entry.recovered_execution is True
    assert entry.verification_status == VerificationStatus.INCONCLUSIVE


# ---- 40. Terminal recovery failure is recorded accurately ----

def test_terminal_recovery_failure_recorded_accurately(test_env):
    """Verifies that an exhausted retry sequence records recovered_execution=False and FAILED_RETRIES_EXHAUSTED."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    def always_fails():
        raise ConnectionError("Persistent network outage")

    task = Task(
        name="exhausted_task",
        callable=always_fails,
        is_idempotent=True,
        max_retries=2,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_RETRIES_EXHAUSTED
    assert result.is_success is False
    assert result.total_attempts == 3
    assert len(memory) == 1

    entry = memory.find_similar("exhausted_task", "network_connectivity")[0]
    assert entry.recovered_execution is False
    assert entry.terminal_status == RecoveryStatus.FAILED_RETRIES_EXHAUSTED
    assert entry.recovery_action == RecoveryAction.RETRY
    assert entry.attempt_count == 3


# ---- 41. Fallback recovery attributes the entry to the primary task ----

def test_fallback_recovery_attributes_to_primary_task(test_env):
    """Verifies that fallback recovery records the primary task identity in task_name."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    fallback_task = Task("fallback_service", callable=lambda: "fallback_ok")
    primary_task = Task(
        name="primary_failing_service",
        callable=lambda: (_ for _ in ()).throw(TypeError("Permanent primary crash")),
        fallback=fallback_task,
    )
    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    assert len(memory) == 1

    entry = memory.find_similar("primary_failing_service", "type_mismatch")[0]
    assert entry.task_name == "primary_failing_service"
    assert entry.recovery_action == RecoveryAction.FALLBACK


# ---- 42. Fallback task name is recorded accurately ----

def test_fallback_task_name_recorded_accurately(test_env):
    """Verifies that fallback_task_name stores the delegated fallback task name."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    fallback_task = Task("my_fallback_worker", callable=lambda: "ok")
    primary_task = Task(
        name="primary_worker",
        callable=lambda: (_ for _ in ()).throw(KeyError("missing_key")),
        fallback=fallback_task,
    )
    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    entry = memory.find_similar("primary_worker", "missing_key")[0]
    assert entry.fallback_task_name == "my_fallback_worker"
    assert entry.terminal_status == RecoveryStatus.RECOVERED_VIA_FALLBACK


# ---- 43. Fallback verification status comes from the actual final task ----

def test_fallback_verification_status_from_final_task(test_env):
    """Verifies that fallback verification outcome is attributed to the entry."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    fallback_task = Task(
        name="verified_fallback",
        callable=lambda: {"res": "good"},
        verifier=lambda out: out.get("res") == "good",
    )
    primary_task = Task(
        name="primary_unverified_task",
        callable=lambda: (_ for _ in ()).throw(TypeError("Crash")),
        fallback=fallback_task,
    )
    result = orchestrator.execute(primary_task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK
    assert result.is_verified is True

    entry = memory.find_similar("primary_unverified_task", "type_mismatch")[0]
    assert entry.verification_status == VerificationStatus.VERIFIED
    assert entry.recovered_execution is True


# ---- 44. First-attempt success does not fabricate a recovery-memory entry ----

def test_first_attempt_success_does_not_fabricate_memory_entry(test_env):
    """Verifies that ordinary first-attempt success is never recorded in recovery memory."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    task = Task("easy_task", callable=lambda: 123)
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert len(memory) == 0


# ---- 45. No duplicate entry is recorded for one workflow ----

def test_no_duplicate_entry_recorded_for_one_workflow(test_env):
    """Verifies that an orchestrated workflow records at most one entry."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Retry me")
        return "finally_done"

    task = Task("multi_retry_task", callable=flaky_work, is_idempotent=True, max_retries=3)
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.total_attempts == 3
    # Despite 3 attempts, exactly 1 memory entry is recorded for the completed workflow
    assert len(memory) == 1


# ---- 46. Sensitive arguments and outputs are not stored ----

def test_sensitive_arguments_and_outputs_not_stored(test_env):
    """Verifies that task args, kwargs, raw outputs, or sensitive tokens are not stored in memory."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    secret_arg = "TOP_SECRET_PASSWORD_123"
    secret_kwarg = "SUPER_SECRET_API_KEY_XYZ"
    secret_output = "SECRET_PAYLOAD_OUTPUT_789"

    call_count = 0
    def sensitive_work(pwd, key=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Transient failure with secret data")
        return secret_output

    task = Task(
        name="sensitive_task",
        callable=sensitive_work,
        args=(secret_arg,),
        kwargs={"key": secret_kwarg},
        is_idempotent=True,
        max_retries=2,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert len(memory) == 1

    entry = memory.find_similar("sensitive_task", "network_connectivity")[0]

    # Convert entire entry to string and dictionary and check for sensitive tokens
    entry_dict = dataclasses.asdict(entry)
    entry_repr = repr(entry)

    for secret in (secret_arg, secret_kwarg, secret_output):
        assert secret not in entry_repr
        assert secret not in str(entry_dict)


# ---- 47. Memory recording exceptions do not change the returned RecoveryResult ----

def test_memory_recording_exceptions_do_not_change_recovery_result(test_env):
    """Verifies that an exception in memory.remember is handled conservatively without crashing."""
    orchestrator, _, _, _ = test_env

    class CrashingMemory(RecoveryMemory):
        def remember(self, entry):
            raise RuntimeError("Simulated memory storage fault")

    crashing_mem = CrashingMemory()
    orchestrator._memory = crashing_mem

    call_count = 0
    def flaky_work():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Flaky attempt")
        return "success_val"

    task = Task("flaky_task_crashing_mem", callable=flaky_work, is_idempotent=True, max_retries=2)

    # Must complete cleanly without raising
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.is_success is True
    assert result.final_result.output == "success_val"


# ---- 48. Memory does not alter retry count, fallback count, or status ----

def test_memory_does_not_alter_execution_or_status(test_env):
    """Verifies that adding memory produces completely identical RecoveryResult to running without memory."""
    orchestrator, _, _, runner = test_env

    def make_flaky_task(task_name: str) -> Task:
        count = 0
        def work():
            nonlocal count
            count += 1
            if count == 1:
                raise ConnectionError("Flaky")
            return "ok"
        return Task(task_name, work, is_idempotent=True, max_retries=2)

    # Run 1: without memory
    orchestrator._memory = None
    res_no_mem = orchestrator.execute(make_flaky_task("task_no_mem"))

    # Run 2: with memory
    memory = RecoveryMemory()
    orchestrator._memory = memory
    res_with_mem = orchestrator.execute(make_flaky_task("task_with_mem"))

    assert res_no_mem.status == res_with_mem.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert res_no_mem.is_success == res_with_mem.is_success is True
    assert res_no_mem.total_attempts == res_with_mem.total_attempts == 2


# ---- 49. Existing callers using positional constructor arguments remain compatible ----

def test_positional_constructor_compatibility_with_10_args():
    """Verifies that RecoveryOrchestrator can be constructed with all 10 positional arguments."""
    runner = TaskRunner()
    detector = FailureDetector()
    engine = DiagnosisEngine()
    policy = RecoveryPolicy()
    memory = RecoveryMemory()

    orchestrator = RecoveryOrchestrator(
        runner,             # 1. runner
        detector,           # 2. detector
        engine,             # 3. diagnosis_engine
        policy,             # 4. policy
        None,               # 5. logger
        lambda s: None,     # 6. sleep_fn
        time.perf_counter,  # 7. time_fn
        2,                  # 8. max_fallback_depth
        None,               # 9. verifier
        memory,             # 10. memory
    )

    try:
        assert orchestrator._memory is memory
        assert orchestrator._max_fallback_depth == 2
        assert isinstance(orchestrator._verifier, OutputVerifier)
    finally:
        orchestrator.close()
        runner.close()


# ---- 50. Custom memory object is not replaced by default ----

def test_custom_memory_object_not_replaced_by_default():
    """Verifies that an explicitly supplied custom memory instance is preserved."""
    custom_memory = RecoveryMemory(max_entries=42)
    orchestrator = RecoveryOrchestrator(memory=custom_memory)

    try:
        assert orchestrator._memory is custom_memory
        assert orchestrator._memory._max_entries == 42
    finally:
        orchestrator.close()


# ---- 51. Memory entry count changes only as expected ----

def test_memory_entry_count_changes_only_as_expected(test_env):
    """Verifies entry counts across mixed workflows (immediate success, retry recovery, attempt 1 failure)."""
    orchestrator, _, _, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    # 1. Immediate success -> 0 entries
    t1 = Task("task_immediate", lambda: "ok")
    orchestrator.execute(t1)
    assert len(memory) == 0

    # 2. Transient failure with retry recovery -> +1 entry
    cnt = 0
    def flaky():
        nonlocal cnt
        cnt += 1
        if cnt == 1:
            raise ConnectionError("err")
        return "ok"
    t2 = Task("task_flaky", flaky, is_idempotent=True, max_retries=2)
    orchestrator.execute(t2)
    assert len(memory) == 1

    # 3. Non-idempotent task fails on attempt 1 without retrying -> +0 entries (no recovery was attempted)
    t3 = Task("task_non_idempotent", lambda: (_ for _ in ()).throw(ConnectionError("fail")), is_idempotent=False)
    orchestrator.execute(t3)
    assert len(memory) == 1


# ---- 52. Budget exhaustion after recovery decision records memory entry ----

def test_budget_exhaustion_after_recovery_decision_records_memory(test_env):
    """Verifies that terminal budget exhaustion after an active retry decision is recorded in memory."""
    orchestrator, clock, sleeper, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    def slow_failing_work():
        clock.advance(0.8)
        raise ConnectionError("slow transient failure")

    task = Task(
        name="budget_exceeded_task",
        callable=slow_failing_work,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=1.0,
        backoff_base_delay=0.5,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.total_attempts == 1
    assert len(memory) == 1

    entry = memory.find_similar("budget_exceeded_task", "network_connectivity")[0]
    assert entry.task_name == "budget_exceeded_task"
    assert entry.terminal_status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert entry.recovered_execution is False
    assert entry.attempt_count == 1
    assert entry.recovery_action == RecoveryAction.RETRY
    assert entry.root_cause == "network_connectivity"


# ---- 53. Budget exhaustion before recovery decision records no memory entry ----

def test_budget_exhaustion_before_recovery_decision_records_no_memory(test_env):
    """Verifies that if budget is exhausted before any recovery/retry is proposed, no entry is recorded."""
    orchestrator, clock, sleeper, _ = test_env
    memory = RecoveryMemory()
    orchestrator._memory = memory

    def slow_failing_work():
        clock.advance(2.0)
        raise ConnectionError("timeout on initial attempt")

    task = Task(
        name="budget_exhausted_before_retry_task",
        callable=slow_failing_work,
        is_idempotent=True,
        max_retries=3,
        recovery_budget_seconds=1.5,
        backoff_base_delay=0.1,
    )
    result = orchestrator.execute(task)

    assert result.status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
    assert result.total_attempts == 1
    assert len(memory) == 0

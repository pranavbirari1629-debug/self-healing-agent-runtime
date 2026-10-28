"""Unit tests for RecoveryPolicy in shar.policy."""

import pytest

from shar.detector import ClassificationResult
from shar.models import (
    ConfidenceLevel,
    Diagnosis,
    DiagnosticContext,
    FailureCategory,
    FailureType,
    RecoveryAction,
    Task,
    TaskOutcome,
    TaskResult,
)
from shar.policy import RecoveryPolicy


@pytest.fixture
def policy():
    """Provides a default RecoveryPolicy instance."""
    return RecoveryPolicy()


def _make_context(
    task_name: str = "sample_task",
    outcome: TaskOutcome = TaskOutcome.FAILURE,
    failure_type: FailureType = FailureType.EXCEPTION,
    category: FailureCategory = FailureCategory.TRANSIENT,
    root_cause: str = "network_connectivity",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    correlation_id: str = "cid-test-123",
) -> tuple[TaskResult, ClassificationResult, Diagnosis]:
    """Helper creating a matching tuple of (TaskResult, ClassificationResult, Diagnosis)."""
    diag_ctx = DiagnosticContext(
        task_name=task_name,
        failure_type=failure_type,
        exception_type="ConnectionError",
        exception_message="socket closed",
    )
    task_res = TaskResult(
        task_name=task_name,
        outcome=outcome,
        failure_type=failure_type,
        exception_type="ConnectionError",
        exception_message="socket closed",
        correlation_id=correlation_id,
    )
    class_res = ClassificationResult(
        category=category,
        is_retryable=(category == FailureCategory.TRANSIENT),
        failure_type=failure_type,
        context=diag_ctx,
    )
    diag = Diagnosis(
        failure_type=failure_type,
        category=category,
        is_retryable=(category == FailureCategory.TRANSIENT),
        root_cause=root_cause,
        description="Observed transient network failure.",
        confidence=confidence,
        context=diag_ctx,
        correlation_id=correlation_id,
    )
    return task_res, class_res, diag


# ---- 1. Success never retried ----

def test_successful_task_never_retried(policy):
    """Verifies that a successful task result produces STOP with no retry."""
    task = Task(name="ok_task", callable=lambda: 42, is_idempotent=True)
    task_res = TaskResult(
        task_name="ok_task",
        outcome=TaskOutcome.SUCCESS,
        output=42,
        failure_type=FailureType.NONE,
    )
    class_res = ClassificationResult(
        category=FailureCategory.UNKNOWN,
        is_retryable=False,
        failure_type=FailureType.NONE,
        context=DiagnosticContext(task_name="ok_task"),
    )
    diag = Diagnosis(
        failure_type=FailureType.NONE,
        category=FailureCategory.UNKNOWN,
        is_retryable=False,
        root_cause="none",
        description="Success",
        confidence=ConfidenceLevel.HIGH,
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert decision.delay_seconds == 0.0
    assert "succeeded" in decision.reason


# ---- 2. Safe transient failure returns RETRY ----

def test_safe_transient_failure_returns_retry(policy):
    """Verifies that an idempotent transient failure with budget remaining authorizes RETRY."""
    task = Task(name="api_call", callable=lambda: None, is_idempotent=True, max_retries=3)
    task_res, class_res, diag = _make_context(
        task_name="api_call",
        category=FailureCategory.TRANSIENT,
        root_cause="network_connectivity",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag, elapsed_recovery_seconds=0.0)

    assert decision.action == RecoveryAction.RETRY
    assert decision.is_safe_to_retry is True
    assert decision.delay_seconds == task.backoff_base_delay
    assert "Retry 1/3" in decision.reason


# ---- 3. Permanent failure is not retried ----

def test_permanent_failure_not_retried(policy):
    """Verifies that permanent errors (e.g. TypeError) immediately halt without retry."""
    task = Task(name="calc_task", callable=lambda: None, is_idempotent=True)
    task_res, class_res, diag = _make_context(
        task_name="calc_task",
        category=FailureCategory.PERMANENT,
        root_cause="type_mismatch",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert "permanent" in decision.reason.lower()


# ---- 4. Unknown failure is not retried by default ----

def test_unknown_failure_not_retried_by_default(policy):
    """Verifies that unknown failure categories are rejected for retry by default."""
    task = Task(name="custom_task", callable=lambda: None, is_idempotent=True)
    task_res, class_res, diag = _make_context(
        task_name="custom_task",
        category=FailureCategory.UNKNOWN,
        root_cause="unrecognized_exception",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert "unknown" in decision.reason.lower()


# ---- 5. Non-idempotent task is not retried ----

def test_non_idempotent_task_not_retried(policy):
    """Verifies that a transient error on a non-idempotent task is refused retry."""
    task = Task(name="charge_card", callable=lambda: None, is_idempotent=False)
    task_res, class_res, diag = _make_context(
        task_name="charge_card",
        category=FailureCategory.TRANSIENT,
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert "is_idempotent=False" in decision.reason


# ---- 6. max_retries=0 blocks retry ----

def test_max_retries_zero_blocks_retry(policy):
    """Verifies that setting max_retries=0 permits zero retries after the initial attempt."""
    task = Task(name="no_retry_task", callable=lambda: None, is_idempotent=True, max_retries=0)
    task_res, class_res, diag = _make_context(task_name="no_retry_task")

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert "exhausted maximum retry limit of 0" in decision.reason


# ---- 7. Retry boundary semantics for max_retries=1 ----

def test_retry_boundary_semantics_max_retries_one(policy):
    """Verifies max_retries=1: attempt 1 allows 1 retry; attempt 2 exhausts retries."""
    task = Task(name="single_retry", callable=lambda: None, is_idempotent=True, max_retries=1)
    task_res, class_res, diag = _make_context(task_name="single_retry")

    # Attempt 1: First failure, 0 retries completed so far -> permits retry #1
    decision_1 = policy.evaluate(task, 1, task_res, class_res, diag)
    assert decision_1.action == RecoveryAction.RETRY
    assert decision_1.is_safe_to_retry is True
    assert "Retry 1/1" in decision_1.reason

    # Attempt 2: Second failure, 1 retry already completed -> limit reached, no retry
    decision_2 = policy.evaluate(task, 2, task_res, class_res, diag)
    assert decision_2.action == RecoveryAction.STOP
    assert decision_2.is_safe_to_retry is False
    assert "exhausted maximum retry limit of 1" in decision_2.reason


# ---- 8. Unsafe timeout blocks retry ----

def test_unsafe_timeout_blocks_retry(policy):
    """Verifies that TIMEOUT failure without safe_to_retry_on_timeout is refused."""
    task = Task(
        name="timed_worker",
        callable=lambda: None,
        is_idempotent=True,
        safe_to_retry_on_timeout=False,
    )
    task_res, class_res, diag = _make_context(
        task_name="timed_worker",
        failure_type=FailureType.TIMEOUT,
        category=FailureCategory.TRANSIENT,
        root_cause="operation_timeout",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert "safe_to_retry_on_timeout is False" in decision.reason
    assert "cannot be forcibly killed" in decision.reason


# ---- 9. Explicitly authorized timeout retry ----

def test_authorized_timeout_retry_permitted(policy):
    """Verifies that TIMEOUT failure with safe_to_retry_on_timeout=True and idempotent is retried."""
    task = Task(
        name="read_cache",
        callable=lambda: None,
        is_idempotent=True,
        safe_to_retry_on_timeout=True,
        max_retries=2,
    )
    task_res, class_res, diag = _make_context(
        task_name="read_cache",
        failure_type=FailureType.TIMEOUT,
        category=FailureCategory.TRANSIENT,
        root_cause="operation_timeout",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.RETRY
    assert decision.is_safe_to_retry is True
    assert decision.delay_seconds == task.backoff_base_delay


# ---- 10. Validation failure behavior ----

def test_validation_failure_behavior(policy):
    """Verifies that VALIDATION_FAILED is retried if idempotent, blocked if not idempotent."""
    task_res, class_res, diag = _make_context(
        task_name="validate_api",
        failure_type=FailureType.VALIDATION_FAILED,
        category=FailureCategory.TRANSIENT,
        root_cause="output_validation_failed",
    )

    # Idempotent task -> should retry
    idempotent_task = Task("validate_api", lambda: None, is_idempotent=True)
    decision = policy.evaluate(idempotent_task, 1, task_res, class_res, diag)
    assert decision.action == RecoveryAction.RETRY
    assert decision.is_safe_to_retry is True

    # Non-idempotent task -> should STOP
    unsafe_task = Task("validate_api", lambda: None, is_idempotent=False)
    decision_unsafe = policy.evaluate(unsafe_task, 1, task_res, class_res, diag)
    assert decision_unsafe.action == RecoveryAction.STOP
    assert decision_unsafe.is_safe_to_retry is False


# ---- 11. Exponential backoff calculation and cap ----

def test_exponential_backoff_calculation_and_cap(policy):
    """Verifies exponential progression: base * (factor^index), capped at max_delay."""
    task = Task(
        name="backoff_task",
        callable=lambda: None,
        is_idempotent=True,
        max_retries=5,
        backoff_base_delay=0.1,
        backoff_factor=2.0,
        backoff_max_delay=0.5,
    )
    task_res, class_res, diag = _make_context(task_name="backoff_task")

    # attempt 1 (retry_index 0): 0.1 * 2^0 = 0.1
    d1 = policy.evaluate(task, 1, task_res, class_res, diag)
    assert pytest.approx(d1.delay_seconds) == 0.1

    # attempt 2 (retry_index 1): 0.1 * 2^1 = 0.2
    d2 = policy.evaluate(task, 2, task_res, class_res, diag)
    assert pytest.approx(d2.delay_seconds) == 0.2

    # attempt 3 (retry_index 2): 0.1 * 2^2 = 0.4
    d3 = policy.evaluate(task, 3, task_res, class_res, diag)
    assert pytest.approx(d3.delay_seconds) == 0.4

    # attempt 4 (retry_index 3): 0.1 * 2^3 = 0.8 -> capped at 0.5
    d4 = policy.evaluate(task, 4, task_res, class_res, diag)
    assert pytest.approx(d4.delay_seconds) == 0.5


# ---- 12. Zero-delay configuration ----

def test_zero_delay_configuration(policy):
    """Verifies that configuring zero delays produces 0.0s delay decisions."""
    task = Task(
        name="immediate_retry",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=0.0,
        backoff_max_delay=0.0,
    )
    task_res, class_res, diag = _make_context(task_name="immediate_retry")

    decision = policy.evaluate(task, 1, task_res, class_res, diag)
    assert decision.action == RecoveryAction.RETRY
    assert decision.delay_seconds == 0.0


# ---- 13. Recovery budget exhausted ----

def test_recovery_budget_exhausted(policy):
    """Verifies that elapsed time meeting or exceeding recovery_budget_seconds triggers ESCALATE."""
    task = Task(
        name="budgeted_task",
        callable=lambda: None,
        is_idempotent=True,
        recovery_budget_seconds=5.0,
    )
    task_res, class_res, diag = _make_context(task_name="budgeted_task")

    decision = policy.evaluate(task, 1, task_res, class_res, diag, elapsed_recovery_seconds=5.0)

    assert decision.action == RecoveryAction.ESCALATE
    assert decision.is_safe_to_retry is False
    assert "exhausted overall recovery budget" in decision.reason


# ---- 14. Insufficient budget for proposed backoff ----

def test_insufficient_budget_for_proposed_backoff(policy):
    """Verifies that ESCALATE is returned if proposed delay exceeds remaining budget."""
    task = Task(
        name="tight_budget_task",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=2.0,
        recovery_budget_seconds=3.0,
    )
    task_res, class_res, diag = _make_context(task_name="tight_budget_task")

    # elapsed 1.5s -> remaining budget 1.5s. But proposed delay is 2.0s -> delay >= remaining
    decision = policy.evaluate(task, 1, task_res, class_res, diag, elapsed_recovery_seconds=1.5)

    assert decision.action == RecoveryAction.ESCALATE
    assert decision.is_safe_to_retry is False
    assert "insufficient for proposed backoff delay" in decision.reason


# ---- 15. Fallback selection safety ----

def test_fallback_selected_safely_when_retries_exhausted(policy):
    """Verifies that an authorized fallback is selected when retries are exhausted."""
    fallback_task = Task(name="fallback_db", callable=lambda: "cached_data")
    primary_task = Task(
        name="primary_db",
        callable=lambda: None,
        is_idempotent=True,
        max_retries=1,
        fallback=fallback_task,
    )
    task_res, class_res, diag = _make_context(task_name="primary_db")

    # Attempt 2: retries exhausted -> routes to fallback
    decision = policy.evaluate(primary_task, 2, task_res, class_res, diag)

    assert decision.action == RecoveryAction.FALLBACK
    assert decision.is_safe_to_retry is False
    assert decision.suggested_task == fallback_task
    assert "fallback_db" in decision.reason


def test_fallback_refused_when_timeout_unsafe(policy):
    """Verifies that fallback is refused if primary task timed out and cannot be confirmed stopped."""
    fallback_task = Task(name="fallback_db", callable=lambda: None)
    primary_task = Task(
        name="hanging_db",
        callable=lambda: None,
        is_idempotent=True,
        safe_to_retry_on_timeout=False,
        fallback=fallback_task,
    )
    task_res, class_res, diag = _make_context(
        task_name="hanging_db",
        failure_type=FailureType.TIMEOUT,
        root_cause="operation_timeout",
    )

    decision = policy.evaluate(primary_task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False
    assert decision.suggested_task is None
    assert "cannot be verified stopped" in decision.reason


# ---- 16. Diagnosis confidence does not override safety checks ----

def test_diagnosis_confidence_does_not_override_safety(policy):
    """Verifies that ConfidenceLevel.HIGH cannot authorize a retry for non-idempotent task."""
    task = Task(name="unsafe_write", callable=lambda: None, is_idempotent=False)
    task_res, class_res, diag = _make_context(
        task_name="unsafe_write",
        category=FailureCategory.TRANSIENT,
        confidence=ConfidenceLevel.HIGH,
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert decision.action == RecoveryAction.STOP
    assert decision.is_safe_to_retry is False


# ---- 17. RecoveryDecision consistency ----

def test_recovery_decision_consistency(policy):
    """Verifies that RETRY decisions always set is_safe_to_retry=True and delay >= 0."""
    task = Task(name="consistent_task", callable=lambda: None, is_idempotent=True)
    task_res, class_res, diag = _make_context(task_name="consistent_task")

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    if decision.action == RecoveryAction.RETRY:
        assert decision.is_safe_to_retry is True
        assert decision.delay_seconds >= 0.0
    else:
        assert decision.is_safe_to_retry is False


# ---- 18. Context preservation and input validation ----

def test_task_context_preserved_in_reason(policy):
    """Verifies that the task name and diagnosis root cause are explicitly recorded in decision reason."""
    task = Task(name="payment_sync", callable=lambda: None, is_idempotent=True)
    task_res, class_res, diag = _make_context(
        task_name="payment_sync",
        root_cause="network_connectivity",
    )

    decision = policy.evaluate(task, 1, task_res, class_res, diag)

    assert "payment_sync" in decision.reason
    assert "network_connectivity" in decision.reason


def test_invalid_attempt_or_elapsed_raises_value_error(policy):
    """Verifies input validation on evaluate arguments."""
    task = Task(name="t", callable=lambda: None)
    task_res, class_res, diag = _make_context(task_name="t")

    with pytest.raises(ValueError, match="attempt_number must be >= 1"):
        policy.evaluate(task, 0, task_res, class_res, diag)

    with pytest.raises(ValueError, match="elapsed_recovery_seconds must be non-negative"):
        policy.evaluate(task, 1, task_res, class_res, diag, elapsed_recovery_seconds=-1.0)


# ---- 19. Jitter behavior and deterministic bounds ----

def test_jitter_deterministic_bounds_and_effects():
    """Verifies that jitter uses injected random_fn and stays bounded within [0.5 * delay, delay]."""
    recorded_calls: list[tuple[float, float]] = []

    def mock_rng(low: float, high: float) -> float:
        recorded_calls.append((low, high))
        return (low + high) / 2.0  # Midpoint: 0.75 * capped_delay

    jitter_policy = RecoveryPolicy(random_fn=mock_rng)
    task = Task(
        name="jitter_task",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=1.0,
        backoff_factor=2.0,
        backoff_max_delay=10.0,
        jitter=True,
    )
    task_res, class_res, diag = _make_context(task_name="jitter_task")

    # attempt 1 (retry_index 0): nominal 1.0 -> bounds (0.5, 1.0) -> midpoint 0.75
    d1 = jitter_policy.evaluate(task, 1, task_res, class_res, diag)
    assert d1.action == RecoveryAction.RETRY
    assert pytest.approx(d1.delay_seconds) == 0.75
    assert recorded_calls[-1] == (0.5, 1.0)

    # attempt 2 (retry_index 1): nominal 2.0 -> bounds (1.0, 2.0) -> midpoint 1.5
    d2 = jitter_policy.evaluate(task, 2, task_res, class_res, diag)
    assert pytest.approx(d2.delay_seconds) == 1.5
    assert recorded_calls[-1] == (1.0, 2.0)


def test_jitter_respects_max_backoff_cap():
    """Verifies that jitter bounds never exceed task.backoff_max_delay."""
    def max_rng(low: float, high: float) -> float:
        # Return the upper bound
        return high

    jitter_policy = RecoveryPolicy(random_fn=max_rng)
    task = Task(
        name="capped_jitter_task",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=2.0,
        backoff_factor=3.0,
        backoff_max_delay=5.0,
        jitter=True,
    )
    task_res, class_res, diag = _make_context(task_name="capped_jitter_task")

    # attempt 3 (retry_index 2): nominal 2.0 * (3^2) = 18.0 -> capped at 5.0
    # bounds are (0.5 * 5.0, 5.0) = (2.5, 5.0) -> upper bound is 5.0
    d = jitter_policy.evaluate(task, 3, task_res, class_res, diag)
    assert pytest.approx(d.delay_seconds) == 5.0


def test_jitter_disabled_ignores_random_fn():
    """Verifies that when task.jitter is False, random_fn is never invoked."""
    def fail_rng(low: float, high: float) -> float:
        raise AssertionError("random_fn should not be called when jitter=False")

    policy_no_jitter = RecoveryPolicy(random_fn=fail_rng)
    task = Task(
        name="no_jitter_task",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=1.0,
        backoff_factor=2.0,
        backoff_max_delay=5.0,
        jitter=False,
    )
    task_res, class_res, diag = _make_context(task_name="no_jitter_task")

    decision = policy_no_jitter.evaluate(task, 1, task_res, class_res, diag)
    assert decision.action == RecoveryAction.RETRY
    assert pytest.approx(decision.delay_seconds) == 1.0


def test_jitter_zero_delay_skips_random_fn():
    """Verifies that zero capped delay returns 0.0 directly without invoking random_fn."""
    def fail_rng(low: float, high: float) -> float:
        raise AssertionError("random_fn should not be called when delay is 0.0")

    policy_zero = RecoveryPolicy(random_fn=fail_rng)
    task = Task(
        name="zero_delay_jitter",
        callable=lambda: None,
        is_idempotent=True,
        backoff_base_delay=0.0,
        backoff_max_delay=0.0,
        jitter=True,
    )
    task_res, class_res, diag = _make_context(task_name="zero_delay_jitter")

    decision = policy_zero.evaluate(task, 1, task_res, class_res, diag)
    assert decision.action == RecoveryAction.RETRY
    assert decision.delay_seconds == 0.0

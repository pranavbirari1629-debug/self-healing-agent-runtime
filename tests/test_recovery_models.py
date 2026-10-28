"""Unit tests for Phase 3 recovery models and Task configuration validation."""

import pytest

import shar
from shar.models import (
    AttemptRecord,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskOutcome,
    TaskResult,
)


def test_task_defaults():
    """Verifies that Phase 3 task configuration fields have safe, expected defaults."""
    task = Task(name="test_defaults", callable=lambda: "ok")

    assert task.is_idempotent is False
    assert task.safe_to_retry_on_timeout is False
    assert task.recovery_budget_seconds is None
    assert task.backoff_base_delay == 0.05
    assert task.backoff_factor == 2.0
    assert task.backoff_max_delay == 2.0
    assert task.jitter is False
    assert task.max_retries == 3
    assert task.verifier is None


def test_task_backward_compatible_constructor():
    """Verifies that the old constructor call signatures without Phase 3 arguments still work."""
    task = Task("legacy_task", lambda x: x * 2, args=(5,), timeout_seconds=1.0)

    assert task.name == "legacy_task"
    assert task.args == (5,)
    assert task.timeout_seconds == 1.0
    assert task.is_idempotent is False


def test_task_validation_negative_max_retries():
    """Verifies that negative max_retries is rejected."""
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        Task(name="bad_retries", callable=lambda: 1, max_retries=-1)


def test_task_validation_negative_timeout():
    """Verifies that negative timeout_seconds is rejected."""
    with pytest.raises(ValueError, match="timeout_seconds must be non-negative"):
        Task(name="bad_timeout", callable=lambda: 1, timeout_seconds=-0.5)


def test_task_validation_negative_recovery_budget():
    """Verifies that negative recovery_budget_seconds is rejected."""
    with pytest.raises(ValueError, match="recovery_budget_seconds must be non-negative"):
        Task(name="bad_budget", callable=lambda: 1, recovery_budget_seconds=-1.0)


def test_task_validation_negative_backoff_delays():
    """Verifies that negative backoff base delay is rejected."""
    with pytest.raises(ValueError, match="backoff_base_delay must be non-negative"):
        Task(name="bad_base", callable=lambda: 1, backoff_base_delay=-0.1)

    with pytest.raises(ValueError, match="backoff_max_delay must be non-negative"):
        Task(name="bad_max", callable=lambda: 1, backoff_base_delay=0.1, backoff_max_delay=-0.5)


def test_task_validation_max_delay_less_than_base():
    """Verifies that backoff_max_delay cannot be less than backoff_base_delay."""
    with pytest.raises(ValueError, match="cannot be less than backoff_base_delay"):
        Task(name="inverted_delays", callable=lambda: 1, backoff_base_delay=1.0, backoff_max_delay=0.5)


def test_task_validation_backoff_factor_minimum():
    """Verifies that backoff_factor must be >= 1.0."""
    with pytest.raises(ValueError, match="backoff_factor must be >= 1.0"):
        Task(name="bad_factor", callable=lambda: 1, backoff_factor=0.9)


def test_task_valid_custom_configuration():
    """Verifies that valid custom Phase 3 configuration values are accepted."""
    task = Task(
        name="custom_task",
        callable=lambda: "ok",
        max_retries=5,
        timeout_seconds=2.5,
        is_idempotent=True,
        safe_to_retry_on_timeout=True,
        recovery_budget_seconds=10.0,
        backoff_base_delay=0.1,
        backoff_factor=1.5,
        backoff_max_delay=5.0,
        jitter=True,
    )

    assert task.max_retries == 5
    assert task.timeout_seconds == 2.5
    assert task.is_idempotent is True
    assert task.safe_to_retry_on_timeout is True
    assert task.recovery_budget_seconds == 10.0
    assert task.backoff_base_delay == 0.1
    assert task.backoff_factor == 1.5
    assert task.backoff_max_delay == 5.0
    assert task.jitter is True


def test_recovery_action_enum_values():
    """Verifies the members of RecoveryAction."""
    assert RecoveryAction.RETRY.value == "retry"
    assert RecoveryAction.FALLBACK.value == "fallback"
    assert RecoveryAction.STOP.value == "stop"
    assert RecoveryAction.ESCALATE.value == "escalate"


def test_recovery_status_enum_values():
    """Verifies the members of RecoveryStatus."""
    expected = {
        "succeeded_first_attempt",
        "recovered_via_retry",
        "recovered_via_fallback",
        "failed_permanent",
        "failed_non_idempotent",
        "failed_timeout_unsafe",
        "failed_retries_exhausted",
        "failed_budget_exceeded",
        "failed_unknown",
        "escalated",
    }
    actual = {status.value for status in RecoveryStatus}
    assert actual == expected


def test_recovery_decision_model():
    """Verifies RecoveryDecision instantiation and delay validation."""
    decision = RecoveryDecision(
        action=RecoveryAction.RETRY,
        reason="Transient network error on idempotent task",
        delay_seconds=0.5,
        is_safe_to_retry=True,
    )

    assert decision.action == RecoveryAction.RETRY
    assert decision.delay_seconds == 0.5
    assert decision.is_safe_to_retry is True

    with pytest.raises(ValueError, match="delay_seconds must be non-negative"):
        RecoveryDecision(action=RecoveryAction.RETRY, reason="test", delay_seconds=-1.0)


def test_attempt_record_model():
    """Verifies AttemptRecord instantiation and validation."""
    task_res = TaskResult(
        task_name="t1",
        outcome=TaskOutcome.FAILURE,
        failure_type=FailureType.EXCEPTION,
        exception_type="ConnectionError",
    )
    decision = RecoveryDecision(action=RecoveryAction.RETRY, reason="test retry", delay_seconds=0.1)

    record = AttemptRecord(
        attempt_number=1,
        task_name="t1",
        task_result=task_res,
        decision=decision,
        elapsed_ms=12.5,
    )

    assert record.attempt_number == 1
    assert record.task_result.exception_type == "ConnectionError"
    assert record.decision.action == RecoveryAction.RETRY
    assert record.elapsed_ms == 12.5

    with pytest.raises(ValueError, match="attempt_number must be >= 1"):
        AttemptRecord(attempt_number=0, task_name="t1", task_result=task_res)

    with pytest.raises(ValueError, match="elapsed_ms must be non-negative"):
        AttemptRecord(attempt_number=1, task_name="t1", task_result=task_res, elapsed_ms=-5.0)


def test_recovery_result_model():
    """Verifies RecoveryResult instantiation, is_success property, and validation."""
    res_success = TaskResult(
        task_name="t1",
        outcome=TaskOutcome.SUCCESS,
        output="final_data",
        failure_type=FailureType.NONE,
    )

    result = RecoveryResult(
        task_name="t1",
        status=RecoveryStatus.RECOVERED_VIA_RETRY,
        final_result=res_success,
        total_attempts=2,
        total_elapsed_ms=150.0,
        correlation_id="corr-12345",
    )

    assert result.is_success is True
    assert result.status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert result.total_attempts == 2
    assert result.correlation_id == "corr-12345"

    with pytest.raises(ValueError, match="total_attempts must be non-negative"):
        RecoveryResult(
            task_name="t1",
            status=RecoveryStatus.FAILED_PERMANENT,
            final_result=res_success,
            total_attempts=-1,
        )

    with pytest.raises(ValueError, match="total_elapsed_ms must be non-negative"):
        RecoveryResult(
            task_name="t1",
            status=RecoveryStatus.FAILED_PERMANENT,
            final_result=res_success,
            total_elapsed_ms=-1.0,
        )


def test_shar_exports_phase3_models():
    """Verifies that all Phase 3 models and enums are exported from the top-level package."""
    assert hasattr(shar, "RecoveryAction")
    assert hasattr(shar, "RecoveryStatus")
    assert hasattr(shar, "RecoveryDecision")
    assert hasattr(shar, "AttemptRecord")
    assert hasattr(shar, "RecoveryResult")
    assert "RecoveryAction" in shar.__all__
    assert "RecoveryStatus" in shar.__all__
    assert "RecoveryDecision" in shar.__all__
    assert "AttemptRecord" in shar.__all__
    assert "RecoveryResult" in shar.__all__


def test_shar_exports_phase4_models():
    """Verifies that Phase 4 verification models and OutputVerifier are exported."""
    assert hasattr(shar, "VerificationStatus")
    assert hasattr(shar, "VerificationResult")
    assert hasattr(shar, "OutputVerifier")
    assert "VerificationStatus" in shar.__all__
    assert "VerificationResult" in shar.__all__
    assert "OutputVerifier" in shar.__all__

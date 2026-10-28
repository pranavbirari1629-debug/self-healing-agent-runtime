"""Unit tests for OutputVerifier in shar.verifier and Phase 4 verification models."""

import pytest

from shar.models import (
    FailureType,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskOutcome,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)
from shar.verifier import OutputVerifier


@pytest.fixture
def verifier():
    """Provides a default OutputVerifier instance."""
    return OutputVerifier()


def _make_task_result(
    output: object = "data",
    outcome: TaskOutcome = TaskOutcome.SUCCESS,
    failure_type: FailureType = FailureType.NONE,
    task_name: str = "test_task",
    correlation_id: str = "test-cid-123",
) -> TaskResult:
    """Helper to construct a TaskResult for testing."""
    return TaskResult(
        task_name=task_name,
        outcome=outcome,
        output=output,
        failure_type=failure_type,
        correlation_id=correlation_id,
    )


# ---- 1. True result produces VERIFIED ----

def test_true_result_produces_verified(verifier):
    """Verifies that when task.verifier returns literal True, VERIFIED status is produced."""
    def check_output(out):
        return out == "expected_value"

    task = Task("check_task", callable=lambda: "expected_value", verifier=check_output)
    result = _make_task_result(output="expected_value", task_name="check_task")

    v_res = verifier.verify(task, result)

    assert v_res.status == VerificationStatus.VERIFIED
    assert v_res.is_verified is True
    assert "satisfied verification check" in v_res.reason
    assert v_res.verifier_name == "check_output"
    assert v_res.error is None


# ---- 2. False result produces FAILED ----

def test_false_result_produces_failed(verifier):
    """Verifies that when task.verifier returns literal False, FAILED status is produced."""
    task = Task("reject_task", callable=lambda: 10, verifier=lambda x: x > 100)
    result = _make_task_result(output=10, task_name="reject_task")

    v_res = verifier.verify(task, result)

    assert v_res.status == VerificationStatus.FAILED
    assert v_res.is_verified is False
    assert "failed verification check" in v_res.reason
    assert v_res.error is None


# ---- 3. Missing task.verifier produces UNVERIFIED ----

def test_missing_verifier_produces_unverified(verifier):
    """Verifies that task without a verifier produces UNVERIFIED without asserting correctness."""
    task = Task("unverified_task", callable=lambda: "ok", verifier=None)
    result = _make_task_result(output="ok", task_name="unverified_task")

    v_res = verifier.verify(task, result)

    assert v_res.status == VerificationStatus.UNVERIFIED
    assert v_res.is_verified is False
    assert "without a configured verifier" in v_res.reason
    assert v_res.verifier_name == "none"


# ---- 4. Verifier exception produces INCONCLUSIVE ----

def test_verifier_exception_produces_inconclusive(verifier):
    """Verifies that exceptions raised by verifier produce INCONCLUSIVE without crashing."""
    def buggy_verifier(out):
        raise KeyError("missing_field")

    task = Task("buggy_task", callable=lambda: {}, verifier=buggy_verifier)
    result = _make_task_result(output={}, task_name="buggy_task")

    v_res = verifier.verify(task, result)

    assert v_res.status == VerificationStatus.INCONCLUSIVE
    assert v_res.is_verified is False
    assert "raised an exception" in v_res.reason
    assert v_res.error == "'missing_field'"
    assert v_res.verifier_name == "buggy_verifier"


# ---- 5. Unsupported verifier return types produce INCONCLUSIVE ----

@pytest.mark.parametrize(
    "unsupported_return",
    [
        "truthy_string",
        "",
        1,  # int 1 (even though 1 == True in Python, type(1) is int, not bool)
        0,
        [1, 2, 3],
        {"status": "ok"},
        None,
    ],
)
def test_unsupported_return_type_produces_inconclusive(verifier, unsupported_return):
    """Verifies that non-boolean return values are rejected as INCONCLUSIVE rather than truthy/falsy."""
    task = Task("type_test", callable=lambda: "data", verifier=lambda x: unsupported_return)
    result = _make_task_result(output="data", task_name="type_test")

    v_res = verifier.verify(task, result)

    assert v_res.status == VerificationStatus.INCONCLUSIVE
    assert v_res.is_verified is False
    assert "unsupported type" in v_res.reason
    assert f"UnsupportedReturnType: {type(unsupported_return).__name__}" in (v_res.error or "")


# ---- 6. Failed execution produces FAILED without calling verifier ----

def test_failed_execution_produces_failed_without_calling_verifier(verifier):
    """Verifies that an unsuccessful task result immediately fails verification without calling verifier."""
    verifier_called = False

    def never_call(out):
        nonlocal verifier_called
        verifier_called = True
        return True

    task = Task("failed_task", callable=lambda: None, verifier=never_call)
    failed_result = TaskResult(
        task_name="failed_task",
        outcome=TaskOutcome.FAILURE,
        output=None,
        failure_type=FailureType.EXCEPTION,
        exception_type="ValueError",
        exception_message="Failed to compute",
        correlation_id="fail-cid",
    )

    v_res = verifier.verify(task, failed_result)

    assert verifier_called is False
    assert v_res.status == VerificationStatus.FAILED
    assert v_res.is_verified is False
    assert "failed execution" in v_res.reason


# ---- 7. Correlation ID is propagated correctly ----

def test_correlation_id_propagation(verifier):
    """Verifies correlation ID is preserved from explicit argument or falls back to result."""
    task = Task("cid_task", callable=lambda: 1, verifier=lambda x: True)
    result = _make_task_result(output=1, correlation_id="result-cid-999")

    # Explicit correlation_id
    v1 = verifier.verify(task, result, correlation_id="explicit-cid-111")
    assert v1.correlation_id == "explicit-cid-111"

    # Fallback to result.correlation_id
    v2 = verifier.verify(task, result)
    assert v2.correlation_id == "result-cid-999"


# ---- 8. VerificationResult invariant enforcement ----

def test_verification_result_invariant_enforcement():
    """Verifies that is_verified is True if and only if status is VERIFIED."""
    # Valid combinations
    res_verified = VerificationResult(status=VerificationStatus.VERIFIED, is_verified=True, reason="ok")
    assert res_verified.is_verified is True

    res_failed = VerificationResult(status=VerificationStatus.FAILED, is_verified=False, reason="bad")
    assert res_failed.is_verified is False

    res_unverified = VerificationResult(status=VerificationStatus.UNVERIFIED, is_verified=False, reason="none")
    assert res_unverified.is_verified is False

    res_inconclusive = VerificationResult(status=VerificationStatus.INCONCLUSIVE, is_verified=False, reason="crash")
    assert res_inconclusive.is_verified is False

    # Inconsistent combinations must raise ValueError
    with pytest.raises(ValueError, match="Inconsistent VerificationResult invariant"):
        VerificationResult(status=VerificationStatus.VERIFIED, is_verified=False, reason="inconsistent")

    with pytest.raises(ValueError, match="Inconsistent VerificationResult invariant"):
        VerificationResult(status=VerificationStatus.FAILED, is_verified=True, reason="inconsistent")

    with pytest.raises(ValueError, match="Inconsistent VerificationResult invariant"):
        VerificationResult(status=VerificationStatus.UNVERIFIED, is_verified=True, reason="inconsistent")

    with pytest.raises(ValueError, match="Inconsistent VerificationResult invariant"):
        VerificationResult(status=VerificationStatus.INCONCLUSIVE, is_verified=True, reason="inconsistent")


# ---- 9. Verifier does not invoke task callable ----

def test_verifier_does_not_invoke_task_callable(verifier):
    """Verifies that OutputVerifier inspects result.output and never calls task.callable."""
    callable_calls = 0

    def work():
        nonlocal callable_calls
        callable_calls += 1
        return "first_run"

    task = Task("side_effect_task", callable=work, verifier=lambda out: out == "first_run")
    result = _make_task_result(output="first_run")

    assert callable_calls == 0
    verifier.verify(task, result)
    assert callable_calls == 0  # Callable was not called


# ---- 10. Existing validator is not called a second time ----

def test_existing_validator_not_called_by_output_verifier(verifier):
    """Verifies that OutputVerifier does not call task.validator."""
    validator_calls = 0

    def task_validator(out):
        nonlocal validator_calls
        validator_calls += 1
        return True

    task = Task("validator_only_task", callable=lambda: 42, validator=task_validator, verifier=None)
    result = _make_task_result(output=42)

    v_res = verifier.verify(task, result)

    assert validator_calls == 0  # Validator was NOT called by OutputVerifier
    assert v_res.status == VerificationStatus.UNVERIFIED


# ---- 11. Process-control exceptions are not caught ----

def test_system_exit_and_keyboard_interrupt_not_swallowed(verifier):
    """Verifies that SystemExit and KeyboardInterrupt propagate and are not swallowed."""
    def exiting_verifier(out):
        raise SystemExit(1)

    task = Task("exit_task", callable=lambda: 1, verifier=exiting_verifier)
    result = _make_task_result(output=1)

    with pytest.raises(SystemExit):
        verifier.verify(task, result)


# ---- 12. RecoveryResult.is_verified property consistency ----

def test_recovery_result_is_verified_property():
    """Verifies that RecoveryResult.is_verified returns True only when verification is VERIFIED."""
    task_res = _make_task_result(output="hello")

    # None verification
    res_none = RecoveryResult(
        task_name="t1",
        status=RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT,
        final_result=task_res,
        verification=None,
    )
    assert res_none.is_success is True
    assert res_none.is_verified is False

    # VERIFIED verification
    v_verified = VerificationResult(status=VerificationStatus.VERIFIED, is_verified=True, reason="ok")
    res_verified = RecoveryResult(
        task_name="t1",
        status=RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT,
        final_result=task_res,
        verification=v_verified,
    )
    assert res_verified.is_success is True
    assert res_verified.is_verified is True

    # FAILED verification on successful execution
    v_failed = VerificationResult(status=VerificationStatus.FAILED, is_verified=False, reason="bad output")
    res_failed = RecoveryResult(
        task_name="t1",
        status=RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT,
        final_result=task_res,
        verification=v_failed,
    )
    # Execution succeeded, but semantic verification failed!
    assert res_failed.is_success is True
    assert res_failed.is_verified is False
    assert res_failed.status == RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
    assert res_failed.verification.status == VerificationStatus.FAILED


# ---- 13. Task backward-compatible constructor with verifier ----

def test_task_constructor_with_verifier():
    """Verifies Task constructor accepts verifier and defaults to None."""
    t_default = Task("t_def", callable=lambda: 1)
    assert t_default.verifier is None

    t_custom = Task("t_custom", callable=lambda: 1, verifier=lambda x: x == 1)
    assert t_custom.verifier is not None
    assert t_custom.verifier(1) is True

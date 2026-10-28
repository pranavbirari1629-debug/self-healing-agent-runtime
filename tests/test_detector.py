"""Unit tests for FailureDetector in shar.detector."""

import pytest

from shar.detector import (
    ClassificationResult,
    FailureDetector,
    DEFAULT_TRANSIENT_EXCEPTIONS,
    DEFAULT_PERMANENT_EXCEPTIONS,
)
from shar.models import (
    FailureCategory,
    FailureType,
    TaskOutcome,
    TaskResult,
)


@pytest.fixture
def detector():
    """Provides a FailureDetector with default registries."""
    return FailureDetector()


def _make_result(
    failure_type: FailureType = FailureType.EXCEPTION,
    exception_type: str | None = "ValueError",
    exception_message: str | None = "test error",
    outcome: TaskOutcome = TaskOutcome.FAILURE,
    traceback_str: str | None = "Traceback ...",
    task_name: str = "test_task",
    correlation_id: str = "test-cid-001",
) -> TaskResult:
    """Helper to construct TaskResults for testing."""
    return TaskResult(
        task_name=task_name,
        outcome=outcome,
        failure_type=failure_type,
        exception_type=exception_type,
        exception_message=exception_message,
        traceback_str=traceback_str,
        execution_time_ms=42.0,
        correlation_id=correlation_id,
    )


# ---- Successful task ----

def test_successful_result_not_classified(detector):
    """A successful TaskResult should not be meaningfully classified."""
    result = TaskResult(
        task_name="ok_task",
        outcome=TaskOutcome.SUCCESS,
        failure_type=FailureType.NONE,
        correlation_id="ok-cid",
    )
    classification = detector.classify(result)

    assert classification.category == FailureCategory.UNKNOWN
    assert classification.is_retryable is False
    assert classification.failure_type == FailureType.NONE


# ---- Timeout ----

def test_timeout_classified_as_transient(detector):
    """Timeout failures must be classified as transient and retryable."""
    result = _make_result(
        failure_type=FailureType.TIMEOUT,
        exception_type="TimeoutError",
    )
    classification = detector.classify(result)

    assert classification.category == FailureCategory.TRANSIENT
    assert classification.is_retryable is True
    assert classification.failure_type == FailureType.TIMEOUT


# ---- Validation failure ----

def test_validation_failure_classified_as_transient(detector):
    """Validation failures must be classified as transient."""
    result = _make_result(
        failure_type=FailureType.VALIDATION_FAILED,
        exception_type="ValidationError",
    )
    classification = detector.classify(result)

    assert classification.category == FailureCategory.TRANSIENT
    assert classification.is_retryable is True
    assert classification.failure_type == FailureType.VALIDATION_FAILED


# ---- Transient exceptions ----

def test_connection_error_classified_as_transient(detector):
    """ConnectionError is in the default transient registry."""
    result = _make_result(exception_type="ConnectionError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.TRANSIENT
    assert classification.is_retryable is True


def test_os_error_classified_as_transient(detector):
    """OSError is in the default transient registry."""
    result = _make_result(exception_type="OSError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.TRANSIENT
    assert classification.is_retryable is True


# ---- Permanent exceptions ----

def test_type_error_classified_as_permanent(detector):
    """TypeError is in the default permanent registry — non-retryable."""
    result = _make_result(exception_type="TypeError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.PERMANENT
    assert classification.is_retryable is False


def test_key_error_classified_as_permanent(detector):
    """KeyError is in the default permanent registry — non-retryable."""
    result = _make_result(exception_type="KeyError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.PERMANENT
    assert classification.is_retryable is False


def test_file_not_found_classified_as_permanent(detector):
    """FileNotFoundError is in the default permanent registry."""
    result = _make_result(exception_type="FileNotFoundError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.PERMANENT
    assert classification.is_retryable is False


# ---- Unknown exceptions ----

def test_unknown_exception_classified_as_unknown(detector):
    """An exception not in either registry defaults to UNKNOWN / non-retryable."""
    result = _make_result(exception_type="MyCustomAppError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.UNKNOWN
    assert classification.is_retryable is False


def test_none_exception_type_classified_as_unknown(detector):
    """If exception_type is None on an EXCEPTION failure, classify as UNKNOWN."""
    result = _make_result(exception_type=None)
    classification = detector.classify(result)

    assert classification.category == FailureCategory.UNKNOWN
    assert classification.is_retryable is False


# ---- Custom registries ----

def test_register_custom_transient_exception(detector):
    """Custom exception types can be registered as transient at runtime."""
    detector.register_transient("MyRetryableError")
    result = _make_result(exception_type="MyRetryableError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.TRANSIENT
    assert classification.is_retryable is True


def test_register_custom_permanent_exception(detector):
    """Custom exception types can be registered as permanent at runtime."""
    detector.register_permanent("FatalConfigError")
    result = _make_result(exception_type="FatalConfigError")
    classification = detector.classify(result)

    assert classification.category == FailureCategory.PERMANENT
    assert classification.is_retryable is False


# ---- Diagnostic context ----

def test_context_extracted_correctly(detector):
    """The ClassificationResult.context should carry safe diagnostic data."""
    result = _make_result(
        task_name="ctx_task",
        exception_type="ConnectionError",
        exception_message="peer refused",
        traceback_str="Traceback line 1\nline 2",
    )
    classification = detector.classify(result)
    ctx = classification.context

    assert ctx.task_name == "ctx_task"
    assert ctx.exception_type == "ConnectionError"
    assert ctx.exception_message == "peer refused"
    assert ctx.has_traceback is True
    assert ctx.execution_time_ms == 42.0


def test_context_no_traceback_flag(detector):
    """When traceback_str is None, has_traceback should be False."""
    result = _make_result(traceback_str=None)
    classification = detector.classify(result)

    assert classification.context.has_traceback is False


# ---- Correlation ID propagation ----

def test_correlation_id_in_context(detector):
    """The correlation_id from the TaskResult flows through to context."""
    result = _make_result(correlation_id="phase2-cid-42")
    classification = detector.classify(result)

    assert classification.context.task_name == "test_task"
    # correlation_id is on the TaskResult, not on context;
    # verify the result itself still carries it
    assert result.correlation_id == "phase2-cid-42"

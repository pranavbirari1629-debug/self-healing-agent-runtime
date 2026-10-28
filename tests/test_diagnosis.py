"""Unit tests for DiagnosisEngine in shar.diagnosis."""

import pytest

from shar.detector import ClassificationResult, FailureDetector
from shar.diagnosis import DiagnosisEngine
from shar.models import (
    ConfidenceLevel,
    Diagnosis,
    DiagnosticContext,
    FailureCategory,
    FailureType,
    TaskOutcome,
    TaskResult,
)


@pytest.fixture
def detector():
    return FailureDetector()


@pytest.fixture
def engine():
    return DiagnosisEngine()


def _make_result(
    failure_type: FailureType = FailureType.EXCEPTION,
    exception_type: str | None = "ValueError",
    exception_message: str | None = "bad value",
    outcome: TaskOutcome = TaskOutcome.FAILURE,
    traceback_str: str | None = "Traceback ...",
    task_name: str = "diag_task",
    correlation_id: str = "diag-cid-001",
) -> TaskResult:
    return TaskResult(
        task_name=task_name,
        outcome=outcome,
        failure_type=failure_type,
        exception_type=exception_type,
        exception_message=exception_message,
        traceback_str=traceback_str,
        execution_time_ms=55.0,
        correlation_id=correlation_id,
    )


# ---- Successful tasks ----

def test_successful_task_diagnosed_as_no_failure(detector, engine):
    """Successful tasks produce a trivial diagnosis with root_cause='none'."""
    result = TaskResult(
        task_name="ok_task",
        outcome=TaskOutcome.SUCCESS,
        failure_type=FailureType.NONE,
        correlation_id="ok-cid",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "none"
    assert diagnosis.category == FailureCategory.UNKNOWN
    assert diagnosis.confidence == ConfidenceLevel.HIGH
    assert diagnosis.is_retryable is False
    assert diagnosis.correlation_id == "ok-cid"


# ---- Timeout ----

def test_timeout_diagnosis_includes_thread_warning(detector, engine):
    """Timeout diagnosis must warn that the thread was not terminated."""
    result = _make_result(
        failure_type=FailureType.TIMEOUT,
        exception_type="TimeoutError",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "operation_timeout"
    assert diagnosis.category == FailureCategory.TRANSIENT
    assert diagnosis.is_retryable is True
    assert diagnosis.confidence == ConfidenceLevel.HIGH
    assert "NOT terminated" in diagnosis.description
    assert diagnosis.failure_type == FailureType.TIMEOUT


# ---- Validation failures ----

def test_validation_failure_diagnosis(detector, engine):
    """Validation failure produces a diagnosis noting the output was invalid."""
    result = _make_result(
        failure_type=FailureType.VALIDATION_FAILED,
        exception_type="ValidationError",
        exception_message="Output validation failed for task 'diag_task'.",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "output_validation_failed"
    assert diagnosis.category == FailureCategory.TRANSIENT
    assert diagnosis.is_retryable is True
    assert diagnosis.confidence == ConfidenceLevel.MEDIUM


def test_validator_crash_diagnosis(detector, engine):
    """When the validator itself crashes, the diagnosis should note this."""
    result = _make_result(
        failure_type=FailureType.VALIDATION_FAILED,
        exception_type="KeyError",  # Not "ValidationError" — the validator raised
        exception_message="Task validator raised an exception: 'missing_key'",
        traceback_str="Traceback ...",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "validator_error"
    assert "validator" in diagnosis.description.lower()
    assert diagnosis.confidence == ConfidenceLevel.MEDIUM
    assert "validator_source_code" in diagnosis.missing_info


# ---- Known transient exceptions ----

def test_connection_error_diagnosis(detector, engine):
    """ConnectionError gets a high-confidence, known root cause."""
    result = _make_result(exception_type="ConnectionError")
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "network_connectivity"
    assert diagnosis.category == FailureCategory.TRANSIENT
    assert diagnosis.is_retryable is True
    assert diagnosis.confidence == ConfidenceLevel.HIGH


# ---- Known permanent exceptions ----

def test_type_error_diagnosis(detector, engine):
    """TypeError gets a high-confidence permanent diagnosis."""
    result = _make_result(exception_type="TypeError")
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "type_mismatch"
    assert diagnosis.category == FailureCategory.PERMANENT
    assert diagnosis.is_retryable is False
    assert diagnosis.confidence == ConfidenceLevel.HIGH


def test_file_not_found_diagnosis(detector, engine):
    """FileNotFoundError maps to a known permanent root cause."""
    result = _make_result(exception_type="FileNotFoundError")
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "file_not_found"
    assert diagnosis.category == FailureCategory.PERMANENT
    assert diagnosis.is_retryable is False
    assert diagnosis.confidence == ConfidenceLevel.HIGH


def test_memory_error_diagnosis(detector, engine):
    """MemoryError is diagnosed as resource_exhausted, permanent."""
    result = _make_result(exception_type="MemoryError")
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "resource_exhausted"
    assert diagnosis.category == FailureCategory.PERMANENT
    assert diagnosis.is_retryable is False


# ---- Unknown exceptions ----

def test_unknown_exception_low_confidence(detector, engine):
    """Unknown exception types produce a LOW confidence diagnosis."""
    result = _make_result(
        exception_type="MyCustomAppError",
        exception_message="something went wrong",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "unrecognized_exception"
    assert diagnosis.category == FailureCategory.UNKNOWN
    assert diagnosis.is_retryable is False
    assert diagnosis.confidence == ConfidenceLevel.LOW
    assert "exception_classification" in diagnosis.missing_info


def test_none_exception_type_notes_missing_info(detector, engine):
    """When exception_type is None, the diagnosis records it as missing."""
    result = _make_result(exception_type=None, traceback_str=None)
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.root_cause == "unknown_exception"
    assert diagnosis.confidence == ConfidenceLevel.LOW
    assert "exception_type" in diagnosis.missing_info


# ---- Missing traceback ----

def test_missing_traceback_noted(detector, engine):
    """When traceback is absent, missing_info should include 'traceback'."""
    result = _make_result(
        exception_type="ConnectionError",
        traceback_str=None,
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert "traceback" in diagnosis.missing_info
    # Even without traceback, root cause should still be identified
    assert diagnosis.root_cause == "network_connectivity"


# ---- Correlation ID propagation ----

def test_correlation_id_propagated(detector, engine):
    """The correlation_id flows through to the Diagnosis."""
    result = _make_result(correlation_id="phase2-trace-99")
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.correlation_id == "phase2-trace-99"


# ---- Context integrity ----

def test_diagnosis_context_matches_classification(detector, engine):
    """The Diagnosis.context should match what the detector extracted."""
    result = _make_result(
        task_name="ctx_check",
        exception_type="ValueError",
        exception_message="bad input",
    )
    classification = detector.classify(result)
    diagnosis = engine.diagnose(result, classification)

    assert diagnosis.context.task_name == "ctx_check"
    assert diagnosis.context.exception_type == "ValueError"
    assert diagnosis.context.exception_message == "bad input"
    assert diagnosis.context.execution_time_ms == 55.0

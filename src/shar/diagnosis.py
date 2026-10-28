"""Structured failure diagnosis for the shar runtime.

The DiagnosisEngine takes a TaskResult and its ClassificationResult and
produces a Diagnosis — a structured record that separates observed facts
from inferred root-cause hypotheses.

Design decisions:
- Diagnosis uses structured exception type inspection, not free-form
  string matching on error messages.
- Root cause descriptions are phrased as hypotheses ("likely", "probably")
  when confidence is not HIGH.
- When diagnostic information is incomplete, the missing signals are
  explicitly listed in ``Diagnosis.missing_info``.
- The engine does not trigger retries or recovery actions — it produces
  a diagnosis that downstream recovery policies can act on.
- No LLM or external API is used; classification is fully deterministic
  and testable.
"""

from __future__ import annotations

import logging
from typing import Optional

from shar.detector import ClassificationResult
from shar.logger import setup_logger
from shar.models import (
    ConfidenceLevel,
    Diagnosis,
    DiagnosticContext,
    FailureCategory,
    FailureType,
    TaskResult,
)


# ---------------------------------------------------------------------------
# Root-cause mapping tables
# ---------------------------------------------------------------------------

#: Maps exception type names to (root_cause, description) tuples for
#: well-known exception patterns. These are observed facts, not guesses.
_KNOWN_ROOT_CAUSES: dict[str, tuple[str, str]] = {
    # Transient / network
    "ConnectionError": (
        "network_connectivity",
        "A network connection could not be established. This is typically "
        "transient and may resolve on retry.",
    ),
    "ConnectionRefusedError": (
        "service_unavailable",
        "The target service refused the connection. The service may be "
        "temporarily down or restarting.",
    ),
    "ConnectionResetError": (
        "connection_interrupted",
        "The connection was reset by the remote host. This is typically "
        "a transient network or server issue.",
    ),
    "TimeoutError": (
        "operation_timeout",
        "The operation exceeded its time limit. This may be caused by "
        "network latency, server load, or a slow dependency.",
    ),
    "BrokenPipeError": (
        "connection_interrupted",
        "The write end of a pipe was closed unexpectedly, typically "
        "indicating a broken connection.",
    ),
    "OSError": (
        "os_level_error",
        "An operating system level error occurred. This may be transient "
        "(e.g., resource temporarily unavailable) or permanent.",
    ),
    # Permanent / logic
    "TypeError": (
        "type_mismatch",
        "A type mismatch occurred in the task logic. This is a code-level "
        "bug that will not resolve on retry.",
    ),
    "ValueError": (
        "invalid_value",
        "An invalid value was provided to an operation. This typically "
        "indicates incorrect input data or configuration.",
    ),
    "KeyError": (
        "missing_key",
        "A dictionary lookup failed because the expected key is absent. "
        "This usually indicates a data schema mismatch.",
    ),
    "IndexError": (
        "index_out_of_range",
        "A sequence index was out of range. This indicates a logic error "
        "or unexpected data length.",
    ),
    "AttributeError": (
        "missing_attribute",
        "An attribute was accessed on an object that does not have it. "
        "This typically indicates an API mismatch or incorrect type.",
    ),
    "ImportError": (
        "missing_dependency",
        "A required module or package could not be imported. This usually "
        "indicates a missing installation or incorrect environment.",
    ),
    "ModuleNotFoundError": (
        "missing_module",
        "A required module was not found. This indicates a missing package "
        "installation or incorrect import path.",
    ),
    "FileNotFoundError": (
        "file_not_found",
        "A required file does not exist at the expected path. This may "
        "indicate incorrect configuration or a missing resource.",
    ),
    "PermissionError": (
        "insufficient_permissions",
        "The process lacks the required permissions for the operation. "
        "This requires configuration or permission changes to resolve.",
    ),
    "ZeroDivisionError": (
        "division_by_zero",
        "A division or modulo by zero occurred. This is a logic error "
        "that will not resolve on retry.",
    ),
    "MemoryError": (
        "resource_exhausted",
        "The process ran out of available memory. This may require "
        "reducing data size or increasing available resources.",
    ),
    "RecursionError": (
        "infinite_recursion",
        "Maximum recursion depth was exceeded. This indicates a logic "
        "error involving unbounded recursion.",
    ),
    "NotImplementedError": (
        "not_implemented",
        "The operation is not implemented. This is a permanent condition "
        "that requires code changes to resolve.",
    ),
}


class DiagnosisEngine:
    """Produces structured Diagnosis records from classified failures.

    The engine inspects a TaskResult alongside its ClassificationResult
    to produce a Diagnosis that separates observed facts from inferred
    root-cause hypotheses. When diagnostic information is incomplete,
    the missing signals are explicitly noted.

    Usage::

        engine = DiagnosisEngine()
        diagnosis = engine.diagnose(task_result, classification)
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or setup_logger("shar.diagnosis")

    def diagnose(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Produce a structured diagnosis for a failed task execution.

        For successful results, returns a minimal diagnosis with
        root_cause "none" and ConfidenceLevel.HIGH.

        Args:
            result: The TaskResult from the runner.
            classification: The ClassificationResult from the detector.

        Returns:
            A Diagnosis with root cause, confidence, retryability hint,
            and any missing diagnostic information noted.
        """
        if result.is_success:
            return self._diagnose_success(result, classification)

        diagnosis = self._diagnose_failure(result, classification)

        self._logger.info(
            f"Diagnosed task '{result.task_name}': "
            f"root_cause={diagnosis.root_cause}, "
            f"category={diagnosis.category.value}, "
            f"confidence={diagnosis.confidence.value}, "
            f"is_retryable={diagnosis.is_retryable}",
            extra={
                "correlation_id": result.correlation_id,
                "task_name": result.task_name,
            },
        )

        return diagnosis

    def _diagnose_success(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Handle the trivial case of a successful task."""
        return Diagnosis(
            failure_type=FailureType.NONE,
            category=FailureCategory.UNKNOWN,
            is_retryable=False,
            root_cause="none",
            description="Task completed successfully. No failure to diagnose.",
            confidence=ConfidenceLevel.HIGH,
            context=classification.context,
            missing_info=[],
            correlation_id=result.correlation_id,
        )

    def _diagnose_failure(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Route to the appropriate diagnosis method by failure type."""
        if result.failure_type == FailureType.TIMEOUT:
            return self._diagnose_timeout(result, classification)

        if result.failure_type == FailureType.VALIDATION_FAILED:
            return self._diagnose_validation(result, classification)

        if result.failure_type == FailureType.EXCEPTION:
            return self._diagnose_exception(result, classification)

        # Unrecognized failure type
        return self._diagnose_unknown(result, classification)

    def _diagnose_timeout(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Diagnose a timeout failure.

        A timeout means the runtime stopped waiting — it does NOT mean
        the underlying operation was cancelled or terminated.
        """
        missing: list[str] = []
        if classification.context.timeout_seconds is None:
            missing.append("configured_timeout_seconds")

        return Diagnosis(
            failure_type=FailureType.TIMEOUT,
            category=FailureCategory.TRANSIENT,
            is_retryable=True,
            root_cause="operation_timeout",
            description=(
                f"Task '{result.task_name}' exceeded its configured timeout. "
                "The runtime stopped waiting for the result, but the "
                "underlying thread was NOT terminated and may still be "
                "running. Retry may succeed if the slowness was transient."
            ),
            confidence=ConfidenceLevel.HIGH,
            context=classification.context,
            missing_info=missing,
            correlation_id=result.correlation_id,
        )

    def _diagnose_validation(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Diagnose a validation failure."""
        # Check if the validator itself crashed vs. returned False
        validator_crashed = (
            result.exception_type is not None
            and result.exception_type != "ValidationError"
        )

        if validator_crashed:
            return Diagnosis(
                failure_type=FailureType.VALIDATION_FAILED,
                category=FailureCategory.TRANSIENT,
                is_retryable=True,
                root_cause="validator_error",
                description=(
                    f"The validator for task '{result.task_name}' raised "
                    f"an exception ({result.exception_type}). This may "
                    "indicate a bug in the validator itself rather than "
                    "invalid task output."
                ),
                confidence=ConfidenceLevel.MEDIUM,
                context=classification.context,
                missing_info=["validator_source_code"],
                correlation_id=result.correlation_id,
            )

        return Diagnosis(
            failure_type=FailureType.VALIDATION_FAILED,
            category=FailureCategory.TRANSIENT,
            is_retryable=True,
            root_cause="output_validation_failed",
            description=(
                f"Task '{result.task_name}' executed without error but "
                "its output did not satisfy the configured validator. "
                "Retry may produce different output if the task depends "
                "on external state."
            ),
            confidence=ConfidenceLevel.MEDIUM,
            context=classification.context,
            missing_info=[],
            correlation_id=result.correlation_id,
        )

    def _diagnose_exception(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Diagnose an exception-based failure using the known root-cause table."""
        missing: list[str] = []

        if result.exception_type is None:
            missing.append("exception_type")
            return Diagnosis(
                failure_type=FailureType.EXCEPTION,
                category=classification.category,
                is_retryable=classification.is_retryable,
                root_cause="unknown_exception",
                description=(
                    f"Task '{result.task_name}' failed with an exception "
                    "but the exception type was not captured."
                ),
                confidence=ConfidenceLevel.LOW,
                context=classification.context,
                missing_info=missing,
                correlation_id=result.correlation_id,
            )

        if not classification.context.has_traceback:
            missing.append("traceback")

        # Check known root-cause table
        known = _KNOWN_ROOT_CAUSES.get(result.exception_type)
        if known is not None:
            root_cause, description = known
            return Diagnosis(
                failure_type=FailureType.EXCEPTION,
                category=classification.category,
                is_retryable=classification.is_retryable,
                root_cause=root_cause,
                description=description,
                confidence=ConfidenceLevel.HIGH,
                context=classification.context,
                missing_info=missing,
                correlation_id=result.correlation_id,
            )

        # Unknown exception type — provide best-effort diagnosis
        return Diagnosis(
            failure_type=FailureType.EXCEPTION,
            category=classification.category,
            is_retryable=classification.is_retryable,
            root_cause="unrecognized_exception",
            description=(
                f"Task '{result.task_name}' raised {result.exception_type}: "
                f"{result.exception_message or '(no message)'}. "
                "This exception type is not in the known classification "
                "registry. Defaulting to non-retryable."
            ),
            confidence=ConfidenceLevel.LOW,
            context=classification.context,
            missing_info=missing + ["exception_classification"],
            correlation_id=result.correlation_id,
        )

    def _diagnose_unknown(
        self,
        result: TaskResult,
        classification: ClassificationResult,
    ) -> Diagnosis:
        """Diagnose an unrecognized failure type."""
        return Diagnosis(
            failure_type=result.failure_type,
            category=FailureCategory.UNKNOWN,
            is_retryable=False,
            root_cause="unrecognized_failure_type",
            description=(
                f"Task '{result.task_name}' failed with an unrecognized "
                f"failure type: {result.failure_type.value}. Unable to "
                "determine root cause."
            ),
            confidence=ConfidenceLevel.LOW,
            context=classification.context,
            missing_info=["failure_type_classification"],
            correlation_id=result.correlation_id,
        )

"""Deterministic failure classification for the shar runtime.

Classifies task execution failures into broad categories (transient,
permanent, unknown) based on the exception type, failure type, and
structured execution context recorded in a TaskResult.

Design decisions:
- Classification uses exception type identity, not string matching.
- Unknown exceptions default to FailureCategory.UNKNOWN (non-retryable)
  rather than optimistically assuming they are transient.
- Timeouts are classified as transient because the operation may succeed
  if given more time or if a transient resource constraint clears.
  A timeout does NOT mean the underlying operation was cancelled.
- Validation failures are classified as transient because a retry may
  produce different (valid) output, e.g. when the callable depends on
  external state that changes between attempts.
- The exception registry is configurable: callers can register custom
  exception types as transient or permanent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from shar.logger import setup_logger
from shar.models import (
    DiagnosticContext,
    FailureCategory,
    FailureType,
    TaskOutcome,
    TaskResult,
)


# ---------------------------------------------------------------------------
# Default exception type registries
# ---------------------------------------------------------------------------

#: Exception types whose failures are typically transient — the same
#: operation may succeed if retried after a delay.
DEFAULT_TRANSIENT_EXCEPTIONS: frozenset[str] = frozenset({
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "TimeoutError",
    "OSError",
    "IOError",
    "BrokenPipeError",
    "InterruptedError",
    "BlockingIOError",
})

#: Exception types whose failures are almost certainly permanent — retrying
#: the identical call will produce the same error.
DEFAULT_PERMANENT_EXCEPTIONS: frozenset[str] = frozenset({
    "TypeError",
    "ValueError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "SyntaxError",
    "ZeroDivisionError",
    "NotImplementedError",
    "RecursionError",
    "StopIteration",
    "MemoryError",
    "OverflowError",
    "UnicodeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "FileNotFoundError",
    "PermissionError",
    "IsADirectoryError",
})


@dataclass
class ClassificationResult:
    """Output of the FailureDetector: a category and retryability hint.

    Attributes:
        category: Broad failure classification.
        is_retryable: Policy hint. True means retry *may* help; False
            means retry is unlikely to resolve the failure.
        failure_type: The original FailureType from the TaskResult.
        context: Structured diagnostic context extracted from the result.
    """

    category: FailureCategory
    is_retryable: bool
    failure_type: FailureType
    context: DiagnosticContext


class FailureDetector:
    """Classifies TaskResult failures into transient, permanent, or unknown.

    The detector inspects the ``failure_type`` and ``exception_type`` fields
    of a TaskResult and maps them to a ``FailureCategory`` using configurable
    exception registries.

    Usage::

        detector = FailureDetector()
        result = runner.run(task)
        if not result.is_success:
            classification = detector.classify(result)
            print(classification.category, classification.is_retryable)
    """

    def __init__(
        self,
        transient_exceptions: Optional[frozenset[str]] = None,
        permanent_exceptions: Optional[frozenset[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the detector with exception registries.

        Args:
            transient_exceptions: Exception class names considered transient.
                Defaults to DEFAULT_TRANSIENT_EXCEPTIONS.
            permanent_exceptions: Exception class names considered permanent.
                Defaults to DEFAULT_PERMANENT_EXCEPTIONS.
            logger: Optional logger instance.
        """
        self._transient = (
            transient_exceptions
            if transient_exceptions is not None
            else DEFAULT_TRANSIENT_EXCEPTIONS
        )
        self._permanent = (
            permanent_exceptions
            if permanent_exceptions is not None
            else DEFAULT_PERMANENT_EXCEPTIONS
        )
        self._logger = logger or setup_logger("shar.detector")

    def register_transient(self, exception_name: str) -> None:
        """Register an additional exception type name as transient."""
        self._transient = self._transient | frozenset({exception_name})

    def register_permanent(self, exception_name: str) -> None:
        """Register an additional exception type name as permanent."""
        self._permanent = self._permanent | frozenset({exception_name})

    def classify(self, result: TaskResult) -> ClassificationResult:
        """Classify a failed TaskResult into a FailureCategory.

        For successful results (outcome == SUCCESS), returns a classification
        with FailureCategory.TRANSIENT set to category UNKNOWN and
        is_retryable False — but callers should check ``result.is_success``
        before calling classify.

        Args:
            result: The TaskResult to classify.

        Returns:
            A ClassificationResult with category, retryability hint, and
            diagnostic context.
        """
        context = self._extract_context(result)

        # Successful tasks need no classification
        if result.is_success:
            return ClassificationResult(
                category=FailureCategory.UNKNOWN,
                is_retryable=False,
                failure_type=FailureType.NONE,
                context=context,
            )

        category, is_retryable = self._classify_failure(result)

        self._logger.info(
            f"Classified failure for task '{result.task_name}': "
            f"category={category.value}, is_retryable={is_retryable}",
            extra={
                "correlation_id": result.correlation_id,
                "task_name": result.task_name,
                "failure_type": result.failure_type.value,
            },
        )

        return ClassificationResult(
            category=category,
            is_retryable=is_retryable,
            failure_type=result.failure_type,
            context=context,
        )

    def _classify_failure(
        self, result: TaskResult
    ) -> tuple[FailureCategory, bool]:
        """Core classification logic. Returns (category, is_retryable)."""

        # Timeout: transient — the operation may succeed with more time
        if result.failure_type == FailureType.TIMEOUT:
            return FailureCategory.TRANSIENT, True

        # Validation failure: transient — retry may produce valid output
        if result.failure_type == FailureType.VALIDATION_FAILED:
            return FailureCategory.TRANSIENT, True

        # Exception-based classification
        if result.failure_type == FailureType.EXCEPTION:
            return self._classify_exception(result.exception_type)

        # Fallback for any unrecognized failure type
        return FailureCategory.UNKNOWN, False

    def _classify_exception(
        self, exception_type: Optional[str]
    ) -> tuple[FailureCategory, bool]:
        """Classify based on exception type name against registries."""
        if exception_type is None:
            return FailureCategory.UNKNOWN, False

        if exception_type in self._transient:
            return FailureCategory.TRANSIENT, True

        if exception_type in self._permanent:
            return FailureCategory.PERMANENT, False

        # Unknown exception — default to non-retryable
        return FailureCategory.UNKNOWN, False

    @staticmethod
    def _extract_context(result: TaskResult) -> DiagnosticContext:
        """Build a safe DiagnosticContext from a TaskResult.

        Intentionally excludes the raw task output, arguments, and full
        traceback text to avoid leaking secrets or large payloads into
        diagnostic records.
        """
        return DiagnosticContext(
            exception_type=result.exception_type,
            exception_message=result.exception_message,
            failure_type=result.failure_type,
            task_name=result.task_name,
            timeout_seconds=None,  # Not available on TaskResult; set by caller if needed
            execution_time_ms=result.execution_time_ms,
            has_traceback=result.traceback_str is not None,
            has_validator=False,  # Not available on TaskResult; set by caller if needed
        )

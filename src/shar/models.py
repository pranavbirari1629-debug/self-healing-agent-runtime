"""Core data models and enums for the Self-Healing AI Agent Runtime (shar).

This module defines the basic abstractions used for task representation,
result capture, failure classification, structured diagnosis, and recovery orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from shar.detector import ClassificationResult


class TaskOutcome(str, Enum):
    """Represents the terminal execution status of a task attempt."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class FailureType(str, Enum):
    """Categorizes the nature of a task execution failure."""

    NONE = "none"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class Task:
    """Defines an executable unit of work with safety boundaries and validation.

    Attributes:
        name: A unique, human-readable identifier for the task.
        callable: The Python function or callable to execute.
        args: Positional arguments passed to the callable.
        kwargs: Keyword arguments passed to the callable.
        timeout_seconds: Maximum wall-clock seconds to wait for execution.
            If None, the task runs without an enforced timeout.
        max_retries: The maximum number of retry attempts permitted AFTER the initial attempt.
            Defaults to 3. Must be >= 0.
        validator: Optional callable that inspects the return value. Must return True
            if valid, or False if the output fails semantic correctness checks.
        fallback: Optional alternate Task to run if recovery fails.
        metadata: User-defined contextual key-value attributes.
        is_idempotent: Explicit declaration by caller whether this task is safe to
            repeat without unintended side effects. Defaults to False (conservative safety).
        safe_to_retry_on_timeout: Explicit declaration by caller whether this task may be
            retried if a previous attempt timed out.
            SAFETY NOTICE: This declaration does NOT terminate or isolate any background
            worker thread that is still running; it is strictly a caller-provided policy authorization.
            Defaults to False.
        recovery_budget_seconds: Optional overall wall-clock recovery budget in seconds across
            all attempts. If exceeded, further retries are halted.
        backoff_base_delay: Initial delay in seconds before the first retry. Must be >= 0.
        backoff_factor: Multiplier for exponential backoff between successive retries. Must be >= 1.0.
        backoff_max_delay: Maximum delay cap in seconds for backoff. Must be >= backoff_base_delay.
        jitter: Whether to add random jitter to retry delays. Disabled by default for deterministic testing.
        verifier: Optional post-execution verification callable that inspects the return value.
            Must return literal True or False. If None, output is considered unverified.
    """

    name: str
    callable: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    validator: Optional[Callable[[Any], bool]] = None
    fallback: Optional[Task] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_idempotent: bool = False
    safe_to_retry_on_timeout: bool = False
    recovery_budget_seconds: Optional[float] = None
    backoff_base_delay: float = 0.05
    backoff_factor: float = 2.0
    backoff_max_delay: float = 2.0
    jitter: bool = False
    verifier: Optional[Callable[[Any], bool]] = None

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")
        if self.timeout_seconds is not None and self.timeout_seconds < 0:
            raise ValueError(f"timeout_seconds must be non-negative, got {self.timeout_seconds}")
        if self.recovery_budget_seconds is not None and self.recovery_budget_seconds < 0:
            raise ValueError(
                f"recovery_budget_seconds must be non-negative, got {self.recovery_budget_seconds}"
            )
        if self.backoff_base_delay < 0:
            raise ValueError(f"backoff_base_delay must be non-negative, got {self.backoff_base_delay}")
        if self.backoff_max_delay < 0:
            raise ValueError(f"backoff_max_delay must be non-negative, got {self.backoff_max_delay}")
        if self.backoff_max_delay < self.backoff_base_delay:
            raise ValueError(
                f"backoff_max_delay ({self.backoff_max_delay}) cannot be less than "
                f"backoff_base_delay ({self.backoff_base_delay})"
            )
        if self.backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {self.backoff_factor}")


@dataclass
class TaskResult:
    """Captures the comprehensive execution outcome and diagnostics of a Task.

    Attributes:
        task_name: The name of the task that was run.
        outcome: Terminal state (SUCCESS or FAILURE).
        output: Value returned by the callable (None if failed or returned None).
        failure_type: Classification of failure if outcome is FAILURE, else NONE.
        exception_type: Class name of the exception caught, if any.
        exception_message: String representation of the exception caught, if any.
        traceback_str: Formatted stack trace string if an exception was raised.
        execution_time_ms: Total elapsed wall-clock execution time in milliseconds.
        correlation_id: Unique tracking ID correlating this execution event.
    """

    task_name: str
    outcome: TaskOutcome
    output: Any = None
    failure_type: FailureType = FailureType.NONE
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    traceback_str: Optional[str] = None
    execution_time_ms: float = 0.0
    correlation_id: str = ""

    @property
    def is_success(self) -> bool:
        """Helper returning True if the outcome is SUCCESS."""
        return self.outcome == TaskOutcome.SUCCESS


# ---------------------------------------------------------------------------
# Phase 2: Failure Detection & Diagnosis models
# ---------------------------------------------------------------------------


class FailureCategory(str, Enum):
    """Broad classification of a failure for recovery policy decisions.

    This classification determines whether a failure is likely to resolve
    on retry (transient), unlikely to resolve without intervention (permanent),
    or insufficiently understood to make that determination (unknown).

    Recovery policies should treat ``UNKNOWN`` as non-retryable by default
    to avoid wasting resources on failures that may never recover.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Indicates how much evidence supports a diagnosis.

    HIGH — The failure matches a well-known, deterministic pattern.
    MEDIUM — The failure matches a known pattern but context is incomplete.
    LOW — The diagnosis is a best-effort inference with limited evidence.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DiagnosticContext:
    """Safe, structured context about a failure for diagnostic purposes.

    This captures only information that is useful for diagnosis without
    storing secrets, credentials, or arbitrary task inputs.

    Attributes:
        exception_type: The class name of the exception, if any.
        exception_message: The string representation of the exception.
        failure_type: The FailureType from the TaskResult.
        task_name: The name of the task that failed.
        timeout_seconds: The configured timeout, if applicable.
        execution_time_ms: How long the task ran before failure.
        has_traceback: Whether a traceback was captured (not the traceback itself).
        has_validator: Whether the task had a validator configured.
    """

    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    failure_type: FailureType = FailureType.NONE
    task_name: str = ""
    timeout_seconds: Optional[float] = None
    execution_time_ms: float = 0.0
    has_traceback: bool = False
    has_validator: bool = False


@dataclass
class Diagnosis:
    """Structured result of analyzing a task failure.

    Separates observed facts from inferred hypotheses and provides
    machine-readable fields for downstream recovery policy selection.

    Attributes:
        failure_type: The original FailureType from the TaskResult.
        category: Broad classification (transient, permanent, unknown).
        is_retryable: Policy hint — True suggests retry *may* help.
            This is a hint, not a guarantee that retrying is safe.
        root_cause: A short, deterministic description of the probable
            root cause. Phrased as a hypothesis when certainty is low.
        description: Human-readable explanation of the diagnosis.
        confidence: How much evidence supports this diagnosis.
        context: Structured diagnostic context (safe subset of failure info).
        missing_info: List of diagnostic signals that were unavailable.
        correlation_id: Propagated from the TaskResult for traceability.
    """

    failure_type: FailureType
    category: FailureCategory
    is_retryable: bool
    root_cause: str
    description: str
    confidence: ConfidenceLevel
    context: DiagnosticContext = field(default_factory=DiagnosticContext)
    missing_info: list[str] = field(default_factory=list)
    correlation_id: str = ""


# ---------------------------------------------------------------------------
# Phase 3: Recovery Engine models
# ---------------------------------------------------------------------------


class RecoveryAction(str, Enum):
    """Actions dictated by a recovery policy in response to an attempt outcome."""

    RETRY = "retry"
    FALLBACK = "fallback"
    STOP = "stop"
    ESCALATE = "escalate"


class RecoveryStatus(str, Enum):
    """Terminal outcome classifications for an orchestrated recovery sequence."""

    SUCCEEDED_FIRST_ATTEMPT = "succeeded_first_attempt"
    RECOVERED_VIA_RETRY = "recovered_via_retry"
    RECOVERED_VIA_FALLBACK = "recovered_via_fallback"
    FAILED_PERMANENT = "failed_permanent"
    FAILED_NON_IDEMPOTENT = "failed_non_idempotent"
    FAILED_TIMEOUT_UNSAFE = "failed_timeout_unsafe"
    FAILED_RETRIES_EXHAUSTED = "failed_retries_exhausted"
    FAILED_BUDGET_EXCEEDED = "failed_budget_exceeded"
    FAILED_UNKNOWN = "failed_unknown"
    ESCALATED = "escalated"


@dataclass
class RecoveryDecision:
    """Explicit decision produced by a RecoveryPolicy for a specific attempt.

    Attributes:
        action: The recommended course of action (RETRY, FALLBACK, STOP, ESCALATE).
        reason: Human-readable rationale explaining why the action was selected.
        delay_seconds: Wall-clock seconds to wait before initiating the next action. Must be >= 0.
        is_safe_to_retry: Explicit confirmation of whether repeating the task is deemed safe.
        suggested_task: Alternate Task to run when action is FALLBACK.
    """

    action: RecoveryAction
    reason: str
    delay_seconds: float = 0.0
    is_safe_to_retry: bool = False
    suggested_task: Optional[Task] = None

    def __post_init__(self) -> None:
        if self.delay_seconds < 0:
            raise ValueError(f"delay_seconds must be non-negative, got {self.delay_seconds}")


@dataclass
class AttemptRecord:
    """Historical snapshot of an individual execution attempt and its diagnostics.

    Attributes:
        attempt_number: 1-based index of this attempt (1 for initial run, 2 for first retry, etc.).
        task_name: The name of the task executed in this attempt.
        task_result: The raw execution result produced by TaskRunner.
        classification: Optional classification result from FailureDetector.
        diagnosis: Optional diagnosis produced by DiagnosisEngine.
        decision: Optional recovery decision produced by RecoveryPolicy.
        elapsed_ms: Elapsed wall-clock time for this specific attempt in milliseconds.
    """

    attempt_number: int
    task_name: str
    task_result: TaskResult
    classification: Optional[ClassificationResult] = None
    diagnosis: Optional[Diagnosis] = None
    decision: Optional[RecoveryDecision] = None
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.attempt_number < 1:
            raise ValueError(f"attempt_number must be >= 1, got {self.attempt_number}")
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")


@dataclass
class RecoveryResult:
    """Final comprehensive result returned by the Recovery Orchestrator.

    Attributes:
        task_name: Name of the top-level task that was orchestrated.
        status: The terminal recovery status.
        final_result: The TaskResult of the last executed attempt.
        attempts: Ordered history of all attempts executed.
        total_attempts: Total number of attempts executed.
        total_elapsed_ms: Cumulative wall-clock execution time across all attempts in milliseconds.
        correlation_id: Propagated correlation ID for traceability.
        final_diagnosis: The diagnosis associated with the final attempt, if applicable.
    """

    task_name: str
    status: RecoveryStatus
    final_result: TaskResult
    attempts: list[AttemptRecord] = field(default_factory=list)
    total_attempts: int = 0
    total_elapsed_ms: float = 0.0
    correlation_id: str = ""
    final_diagnosis: Optional[Diagnosis] = None
    verification: Optional[VerificationResult] = None

    def __post_init__(self) -> None:
        if self.total_attempts < 0:
            raise ValueError(f"total_attempts must be non-negative, got {self.total_attempts}")
        if self.total_elapsed_ms < 0:
            raise ValueError(f"total_elapsed_ms must be non-negative, got {self.total_elapsed_ms}")

    @property
    def is_success(self) -> bool:
        """Helper returning True if the final executed attempt succeeded."""
        return self.final_result.is_success

    @property
    def is_verified(self) -> bool:
        """Helper returning True if a verification result exists and its status is VERIFIED."""
        return (
            self.verification is not None
            and self.verification.status == VerificationStatus.VERIFIED
        )


# ---------------------------------------------------------------------------
# Phase 4: Verification models
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Explicit status indicating the outcome of semantic output verification."""

    VERIFIED = "verified"
    FAILED = "failed"
    UNVERIFIED = "unverified"
    INCONCLUSIVE = "inconclusive"


@dataclass
class VerificationResult:
    """Detailed record of post-execution output verification.

    Attributes:
        status: The explicit VerificationStatus.
        is_verified: True if and only if status is VERIFIED.
        reason: Human-readable explanation of the verification outcome.
        verifier_name: Identifier of the verifier callable or rule applied. Defaults to "none".
        error: String representation of exception if the verifier crashed. Defaults to None.
        correlation_id: Propagated correlation ID for traceability. Defaults to "".
    """

    status: VerificationStatus
    is_verified: bool
    reason: str
    verifier_name: str = "none"
    error: Optional[str] = None
    correlation_id: str = ""

    def __post_init__(self) -> None:
        expected_verified = (self.status == VerificationStatus.VERIFIED)
        if self.is_verified != expected_verified:
            raise ValueError(
                f"Inconsistent VerificationResult invariant: is_verified={self.is_verified} "
                f"must be True if and only if status is VerificationStatus.VERIFIED, got status={self.status.value}"
            )

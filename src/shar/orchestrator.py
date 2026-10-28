"""Recovery Orchestrator for the Self-Healing AI Agent Runtime (shar).

Coordinates task execution, failure detection, diagnosis, recovery policy evaluation,
bounded retries, fallback delegation, structured attempt history, and observability.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Optional

from shar.detector import FailureDetector
from shar.diagnosis import DiagnosisEngine
from shar.logger import setup_logger
from shar.models import (
    AttemptRecord,
    Diagnosis,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)
from shar.memory import RecoveryMemory, RecoveryMemoryEntry
from shar.policy import RecoveryPolicy
from shar.runner import TaskRunner
from shar.verifier import OutputVerifier


class RecoveryOrchestrator:
    """Coordinates task execution and self-healing recovery workflows.

    Coordinates:
    - TaskRunner: Executes individual task attempts with timeout boundaries.
    - FailureDetector: Classifies failures (transient, permanent, unknown).
    - DiagnosisEngine: Determines root causes, confidence, and missing context.
    - RecoveryPolicy: Decides recovery action (RETRY, FALLBACK, STOP, ESCALATE).

    Safety Guarantees:
    - Never retries without explicit idempotency confirmation.
    - Never retries unsafe timeouts where background threads may still run.
    - Strictly bounds retries to task.max_retries (counted after initial attempt).
    - Enforces cumulative recovery budgets (wall-clock time across all attempts).
    - Enforces a maximum fallback depth (default 1) to prevent infinite chains.
    - Maintains comprehensive attempt history and preserves correlation IDs.
    """

    def __init__(
        self,
        runner: Optional[TaskRunner] = None,
        detector: Optional[FailureDetector] = None,
        diagnosis_engine: Optional[DiagnosisEngine] = None,
        policy: Optional[RecoveryPolicy] = None,
        logger: Optional[logging.Logger] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.perf_counter,
        max_fallback_depth: int = 1,
        verifier: Optional[OutputVerifier] = None,
        memory: Optional[RecoveryMemory] = None,
    ) -> None:
        """Initializes the RecoveryOrchestrator.

        Args:
            runner: TaskRunner instance. Created if omitted.
            detector: FailureDetector instance. Created if omitted.
            diagnosis_engine: DiagnosisEngine instance. Created if omitted.
            policy: RecoveryPolicy instance. Created if omitted.
            logger: Custom logger instance. Defaults to 'shar.orchestrator'.
            sleep_fn: Injected sleep function for backoff delays (allows fast unit tests).
            time_fn: Injected monotonic clock for timing and budgets.
            max_fallback_depth: Maximum recursion depth for fallback tasks (defaults to 1).
            verifier: OutputVerifier instance. Created if omitted.
            memory: Optional RecoveryMemory instance for observing completed recovery workflows.
        """
        self._owns_runner = runner is None
        self._runner = runner or TaskRunner()
        self._detector = detector or FailureDetector()
        self._diagnosis_engine = diagnosis_engine or DiagnosisEngine()
        self._policy = policy or RecoveryPolicy()
        self._logger = logger or setup_logger("shar.orchestrator")
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._max_fallback_depth = max_fallback_depth
        self._verifier = verifier or OutputVerifier()
        self._memory = memory

    def close(self) -> None:
        """Closes the underlying TaskRunner if owned by this orchestrator."""
        if self._owns_runner:
            self._runner.close()

    def __enter__(self) -> RecoveryOrchestrator:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def execute(
        self,
        task: Task,
        correlation_id: Optional[str] = None,
    ) -> RecoveryResult:
        """Executes a task and orchestrates recovery according to safety policies.

        Args:
            task: The primary Task definition to execute.
            correlation_id: Optional tracking identifier. Generated if omitted.

        Returns:
            A RecoveryResult detailing final outcome, status, total attempts,
            execution timing, attempt history, and diagnosis.
        """
        cid = correlation_id or uuid.uuid4().hex
        start_time = self._time_fn()
        root_recovery_budget = task.recovery_budget_seconds

        self._logger.info(
            f"Starting orchestration for task '{task.name}'",
            extra={"task_name": task.name, "correlation_id": cid},
        )

        attempts: list[AttemptRecord] = []
        current_task = task
        task_attempt_number = 1  # 1-based attempt count for current_task
        fallback_depth = 0
        final_diagnosis: Optional[Diagnosis] = None

        while True:
            # 1. Overall budget check before task invocation
            elapsed_total = self._time_fn() - start_time
            effective_budget = self._get_effective_budget(root_recovery_budget, current_task)
            if (
                effective_budget is not None
                and elapsed_total >= effective_budget
            ):
                if attempts:
                    last_result = attempts[-1].task_result
                    budget_msg = (
                        f"Task '{task.name}' exceeded overall recovery budget of "
                        f"{effective_budget}s before attempt could start."
                    )
                    self._logger.warning(
                        budget_msg,
                        extra={"task_name": task.name, "correlation_id": cid},
                    )
                    return self._build_recovery_result(
                        root_task=task,
                        current_task=current_task,
                        status=RecoveryStatus.FAILED_BUDGET_EXCEEDED,
                        final_result=last_result,
                        attempts=attempts,
                        start_time=start_time,
                        correlation_id=cid,
                        final_diagnosis=final_diagnosis,
                    )

            # 2. Execute attempt
            attempt_index = len(attempts) + 1  # monotonic overall attempt index
            attempt_start = self._time_fn()
            task_result = self._runner.run(current_task, correlation_id=cid)
            attempt_ms = (self._time_fn() - attempt_start) * 1000.0

            # 3. Check for immediate attempt success
            if task_result.is_success:
                record = AttemptRecord(
                    attempt_number=attempt_index,
                    task_name=current_task.name,
                    task_result=task_result,
                    classification=None,
                    diagnosis=None,
                    decision=None,
                    elapsed_ms=attempt_ms,
                )
                attempts.append(record)

                if attempt_index == 1 and fallback_depth == 0:
                    status = RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT
                elif fallback_depth > 0:
                    status = RecoveryStatus.RECOVERED_VIA_FALLBACK
                else:
                    status = RecoveryStatus.RECOVERED_VIA_RETRY

                total_elapsed_ms = (self._time_fn() - start_time) * 1000.0
                self._logger.info(
                    f"Task '{task.name}' reached terminal success status: {status.value}",
                    extra={
                        "task_name": task.name,
                        "correlation_id": cid,
                        "status": status.value,
                        "total_attempts": len(attempts),
                        "total_elapsed_ms": total_elapsed_ms,
                    },
                )
                return self._build_recovery_result(
                    root_task=task,
                    current_task=current_task,
                    status=status,
                    final_result=task_result,
                    attempts=attempts,
                    start_time=start_time,
                    correlation_id=cid,
                    final_diagnosis=final_diagnosis,
                )

            # 4. Attempt failed: Run failure analysis and policy evaluation
            elapsed_so_far = self._time_fn() - start_time
            try:
                classification = self._detector.classify(task_result)
                diagnosis = self._diagnosis_engine.diagnose(task_result, classification)
                final_diagnosis = diagnosis
                decision = self._policy.evaluate(
                    task=current_task,
                    attempt_number=task_attempt_number,
                    result=task_result,
                    classification=classification,
                    diagnosis=diagnosis,
                    elapsed_recovery_seconds=elapsed_so_far,
                )
            except Exception as analysis_exc:
                # Conservative error handling: diagnostic component failure must not cause blind retries
                err_msg = (
                    f"Unexpected exception during failure analysis for task '{current_task.name}': "
                    f"{analysis_exc}"
                )
                self._logger.error(
                    err_msg,
                    extra={"task_name": current_task.name, "correlation_id": cid},
                    exc_info=True,
                )
                emergency_decision = RecoveryDecision(
                    action=RecoveryAction.ESCALATE,
                    reason=err_msg,
                    delay_seconds=0.0,
                    is_safe_to_retry=False,
                )
                record = AttemptRecord(
                    attempt_number=attempt_index,
                    task_name=current_task.name,
                    task_result=task_result,
                    classification=None,
                    diagnosis=None,
                    decision=emergency_decision,
                    elapsed_ms=attempt_ms,
                )
                attempts.append(record)
                return self._build_recovery_result(
                    root_task=task,
                    current_task=current_task,
                    status=RecoveryStatus.ESCALATED,
                    final_result=task_result,
                    attempts=attempts,
                    start_time=start_time,
                    correlation_id=cid,
                    final_diagnosis=final_diagnosis,
                )

            # Record failed attempt details
            record = AttemptRecord(
                attempt_number=attempt_index,
                task_name=current_task.name,
                task_result=task_result,
                classification=classification,
                diagnosis=diagnosis,
                decision=decision,
                elapsed_ms=attempt_ms,
            )
            attempts.append(record)

            # If this failure occurred during a fallback attempt, do not retry fallback indefinitely
            if fallback_depth > 0:
                self._logger.warning(
                    f"Fallback task '{current_task.name}' failed. Bounded fallback depth reached; escalating.",
                    extra={"task_name": current_task.name, "correlation_id": cid},
                )
                return self._build_recovery_result(
                    root_task=task,
                    current_task=current_task,
                    status=RecoveryStatus.ESCALATED,
                    final_result=task_result,
                    attempts=attempts,
                    start_time=start_time,
                    correlation_id=cid,
                    final_diagnosis=final_diagnosis,
                )

            # 5. Process RecoveryDecision for primary task
            if decision.action == RecoveryAction.RETRY:
                # Safety assertion: Decision must declare retry safe
                if not decision.is_safe_to_retry:
                    self._logger.warning(
                        f"Inconsistent decision for '{current_task.name}': RETRY with is_safe_to_retry=False.",
                        extra={"task_name": current_task.name, "correlation_id": cid},
                    )
                    return self._build_recovery_result(
                        root_task=task,
                        current_task=current_task,
                        status=RecoveryStatus.ESCALATED,
                        final_result=task_result,
                        attempts=attempts,
                        start_time=start_time,
                        correlation_id=cid,
                        final_diagnosis=final_diagnosis,
                    )

                # Check budget before sleeping
                elapsed_before_sleep = self._time_fn() - start_time
                if effective_budget is not None:
                    remaining_budget = effective_budget - elapsed_before_sleep
                    if remaining_budget <= 0 or decision.delay_seconds >= remaining_budget:
                        return self._build_recovery_result(
                            root_task=task,
                            current_task=current_task,
                            status=RecoveryStatus.FAILED_BUDGET_EXCEEDED,
                            final_result=task_result,
                            attempts=attempts,
                            start_time=start_time,
                            correlation_id=cid,
                            final_diagnosis=final_diagnosis,
                        )

                # Wait backoff delay
                if decision.delay_seconds > 0.0:
                    self._sleep_fn(decision.delay_seconds)

                # Check budget after sleeping
                elapsed_after_sleep = self._time_fn() - start_time
                if (
                    effective_budget is not None
                    and elapsed_after_sleep >= effective_budget
                ):
                    return self._build_recovery_result(
                        root_task=task,
                        current_task=current_task,
                        status=RecoveryStatus.FAILED_BUDGET_EXCEEDED,
                        final_result=task_result,
                        attempts=attempts,
                        start_time=start_time,
                        correlation_id=cid,
                        final_diagnosis=final_diagnosis,
                    )

                # Proceed to next attempt
                task_attempt_number += 1
                continue

            elif decision.action == RecoveryAction.FALLBACK:
                # Fallback depth boundary check
                if fallback_depth >= self._max_fallback_depth or decision.suggested_task is None:
                    self._logger.warning(
                        f"Fallback execution refused: depth limit ({self._max_fallback_depth}) reached.",
                        extra={"task_name": current_task.name, "correlation_id": cid},
                    )
                    return self._build_recovery_result(
                        root_task=task,
                        current_task=current_task,
                        status=RecoveryStatus.ESCALATED,
                        final_result=task_result,
                        attempts=attempts,
                        start_time=start_time,
                        correlation_id=cid,
                        final_diagnosis=final_diagnosis,
                    )

                # Check budget before transitioning to fallback
                elapsed_before_fallback = self._time_fn() - start_time
                fallback_budget = self._get_effective_budget(
                    root_recovery_budget, decision.suggested_task
                )
                if (
                    fallback_budget is not None
                    and elapsed_before_fallback >= fallback_budget
                ):
                    return self._build_recovery_result(
                        root_task=task,
                        current_task=current_task,
                        status=RecoveryStatus.FAILED_BUDGET_EXCEEDED,
                        final_result=task_result,
                        attempts=attempts,
                        start_time=start_time,
                        correlation_id=cid,
                        final_diagnosis=final_diagnosis,
                    )

                self._logger.info(
                    f"Transitioning to fallback task '{decision.suggested_task.name}'",
                    extra={
                        "primary_task": current_task.name,
                        "fallback_task": decision.suggested_task.name,
                        "correlation_id": cid,
                    },
                )
                current_task = decision.suggested_task
                task_attempt_number = 1
                fallback_depth += 1
                continue

            elif decision.action == RecoveryAction.STOP:
                # Determine specific terminal failure status
                status = self._map_stop_status(current_task, task_result, classification, task_attempt_number)
                return self._build_recovery_result(
                    root_task=task,
                    current_task=current_task,
                    status=status,
                    final_result=task_result,
                    attempts=attempts,
                    start_time=start_time,
                    correlation_id=cid,
                    final_diagnosis=final_diagnosis,
                )

            else:  # RecoveryAction.ESCALATE
                if "budget" in decision.reason.lower():
                    status = RecoveryStatus.FAILED_BUDGET_EXCEEDED
                else:
                    status = RecoveryStatus.ESCALATED

                return self._build_recovery_result(
                    root_task=task,
                    current_task=current_task,
                    status=status,
                    final_result=task_result,
                    attempts=attempts,
                    start_time=start_time,
                    correlation_id=cid,
                    final_diagnosis=final_diagnosis,
                )

    def _build_recovery_result(
        self,
        root_task: Task,
        current_task: Task,
        status: RecoveryStatus,
        final_result: TaskResult,
        attempts: list[AttemptRecord],
        start_time: float,
        correlation_id: str,
        final_diagnosis: Optional[Diagnosis],
    ) -> RecoveryResult:
        """Constructs the terminal RecoveryResult, performing output verification exactly once."""
        total_elapsed_ms = (self._time_fn() - start_time) * 1000.0
        verification = self._verifier.verify(
            task=current_task,
            result=final_result,
            correlation_id=correlation_id,
        )

        if verification.status == VerificationStatus.VERIFIED:
            self._logger.info(
                f"Task '{root_task.name}' output verified successfully (verifier={verification.verifier_name})",
                extra={
                    "task_name": root_task.name,
                    "correlation_id": correlation_id,
                    "verification_status": verification.status.value,
                },
            )
        elif verification.status in (VerificationStatus.FAILED, VerificationStatus.INCONCLUSIVE):
            self._logger.warning(
                f"Task '{root_task.name}' output verification {verification.status.value}: {verification.reason}",
                extra={
                    "task_name": root_task.name,
                    "correlation_id": correlation_id,
                    "verification_status": verification.status.value,
                },
            )
        else:  # VerificationStatus.UNVERIFIED
            self._logger.debug(
                f"Task '{root_task.name}' output unverified: {verification.reason}",
                extra={
                    "task_name": root_task.name,
                    "correlation_id": correlation_id,
                    "verification_status": verification.status.value,
                },
            )

        # Record workflow outcome to recovery memory if configured
        self._record_memory_safe(
            root_task=root_task,
            current_task=current_task,
            status=status,
            final_result=final_result,
            attempts=attempts,
            verification=verification,
            final_diagnosis=final_diagnosis,
            correlation_id=correlation_id,
        )

        return RecoveryResult(
            task_name=root_task.name,
            status=status,
            final_result=final_result,
            attempts=attempts,
            total_attempts=len(attempts),
            total_elapsed_ms=total_elapsed_ms,
            correlation_id=correlation_id,
            final_diagnosis=final_diagnosis,
            verification=verification,
        )

    def _record_memory_safe(
        self,
        root_task: Task,
        current_task: Task,
        status: RecoveryStatus,
        final_result: TaskResult,
        attempts: list[AttemptRecord],
        verification: VerificationResult,
        final_diagnosis: Optional[Diagnosis],
        correlation_id: str,
    ) -> None:
        """Safely records a completed recovery workflow into memory without affecting execution."""
        if self._memory is None:
            return

        # Determine whether a recovery action was actively engaged
        last_decision = attempts[-1].decision if attempts else None

        has_proposed_retry = (
            last_decision is not None
            and (
                last_decision.action == RecoveryAction.RETRY
                or (
                    last_decision.action == RecoveryAction.ESCALATE
                    and "proposed backoff" in last_decision.reason.lower()
                )
            )
        )

        has_proposed_fallback = (
            last_decision is not None
            and last_decision.action == RecoveryAction.FALLBACK
        )

        # Record workflows where a retry or fallback was actively proposed before budget exhaustion
        is_budget_exceeded_recovery = (
            status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
            and len(attempts) >= 1
            and not attempts[0].task_result.is_success
            and (has_proposed_retry or has_proposed_fallback)
        )

        is_recovery_workflow = (
            status in (RecoveryStatus.RECOVERED_VIA_RETRY, RecoveryStatus.RECOVERED_VIA_FALLBACK)
            or len(attempts) >= 2
            or is_budget_exceeded_recovery
        )
        if not is_recovery_workflow:
            return

        try:
            # Find the initial failure that triggered recovery
            first_failure = next((a for a in attempts if not a.task_result.is_success), attempts[0])

            # Determine failure category
            if first_failure.classification is not None:
                failure_category = first_failure.classification.category
            elif final_diagnosis is not None:
                failure_category = final_diagnosis.category
            else:
                failure_category = FailureCategory.UNKNOWN

            # Determine root cause label
            if first_failure.diagnosis is not None and first_failure.diagnosis.root_cause:
                root_cause = first_failure.diagnosis.root_cause
            elif final_diagnosis is not None and final_diagnosis.root_cause:
                root_cause = final_diagnosis.root_cause
            else:
                root_cause = "unclassified_failure"

            # Determine recovery action and fallback attribution
            is_fallback = (
                status == RecoveryStatus.RECOVERED_VIA_FALLBACK
                or current_task.name != root_task.name
                or any(a.task_name != root_task.name for a in attempts)
                or (
                    status == RecoveryStatus.FAILED_BUDGET_EXCEEDED
                    and attempts
                    and attempts[-1].decision is not None
                    and attempts[-1].decision.action == RecoveryAction.FALLBACK
                )
            )

            if is_fallback:
                recovery_action = RecoveryAction.FALLBACK
                if current_task.name != root_task.name:
                    fallback_name = current_task.name
                elif any(a.task_name != root_task.name for a in attempts):
                    fallback_name = attempts[-1].task_name
                elif (
                    attempts
                    and attempts[-1].decision is not None
                    and attempts[-1].decision.suggested_task is not None
                ):
                    fallback_name = attempts[-1].decision.suggested_task.name
                else:
                    fallback_name = None
            else:
                recovery_action = RecoveryAction.RETRY
                fallback_name = None

            entry = RecoveryMemoryEntry(
                entry_id=uuid.uuid4().hex,
                task_name=root_task.name,
                failure_type=first_failure.task_result.failure_type,
                failure_category=failure_category,
                root_cause=root_cause,
                exception_type=first_failure.task_result.exception_type,
                recovery_action=recovery_action,
                recovered_execution=final_result.is_success,
                verification_status=verification.status,
                attempt_count=len(attempts),
                terminal_status=status,
                fallback_task_name=fallback_name,
                timestamp=self._time_fn(),
            )
            self._memory.remember(entry)

        except Exception as exc:
            self._logger.warning(
                f"Failed to record recovery memory entry for task '{root_task.name}': {exc}",
                extra={
                    "task_name": root_task.name,
                    "correlation_id": correlation_id,
                },
            )

    @staticmethod
    def _get_effective_budget(
        root_budget: Optional[float],
        task: Task,
    ) -> Optional[float]:
        """Computes the effective recovery budget for a task attempt.

        The root task's recovery budget belongs to the entire execution workflow.
        A fallback task's budget cannot expand the root budget; if both specify a
        budget, the stricter (smaller) constraint applies.
        """
        if root_budget is not None:
            if task.recovery_budget_seconds is not None:
                return min(root_budget, task.recovery_budget_seconds)
            return root_budget
        return task.recovery_budget_seconds

    def _map_stop_status(
        self,
        task: Task,
        result: TaskResult,
        classification: object,
        task_attempt_number: int,
    ) -> RecoveryStatus:
        """Determines the exact RecoveryStatus corresponding to a STOP decision."""
        if result.failure_type == FailureType.TIMEOUT and not task.safe_to_retry_on_timeout:
            return RecoveryStatus.FAILED_TIMEOUT_UNSAFE

        if not task.is_idempotent:
            return RecoveryStatus.FAILED_NON_IDEMPOTENT

        category = getattr(classification, "category", None)
        if category == FailureCategory.PERMANENT:
            return RecoveryStatus.FAILED_PERMANENT

        if category == FailureCategory.UNKNOWN:
            return RecoveryStatus.FAILED_UNKNOWN

        retries_completed = task_attempt_number - 1
        if retries_completed >= task.max_retries:
            return RecoveryStatus.FAILED_RETRIES_EXHAUSTED

        return RecoveryStatus.ESCALATED

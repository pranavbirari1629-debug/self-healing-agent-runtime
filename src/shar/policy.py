"""Deterministic recovery policy decision engine for the shar runtime.

This module evaluates attempt diagnostics against safety boundaries (idempotency,
timeout semantics, retry limits, and execution budgets) to produce a RecoveryDecision.

Design Principles:
- Decision logic ONLY: this module calculates what action should occur and what delay
  to use, but never sleeps, executes callables, or triggers retries directly.
- Safety-first gating: Transient failure category is a necessary condition for retry,
  but NEVER a sufficient one. Operations must be explicitly declared idempotent.
- Honest timeout policy: Python threads that time out are NOT killed. Retrying a timed-out
  task is forbidden unless the caller explicitly confirmed safe_to_retry_on_timeout.
- Informational confidence: Diagnosis confidence does not independently authorize retries.
- Bounded budgets: Recovery budgets apply to the cumulative orchestration, not per-attempt.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Optional

from shar.detector import ClassificationResult
from shar.logger import setup_logger
from shar.models import (
    Diagnosis,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    Task,
    TaskResult,
)


class RecoveryPolicy:
    """Evaluates task execution results and diagnostics to determine the recovery action.

    Usage::

        policy = RecoveryPolicy()
        decision = policy.evaluate(
            task=task,
            attempt_number=1,
            result=task_result,
            classification=classification_result,
            diagnosis=diagnosis,
            elapsed_recovery_seconds=0.15,
        )
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        random_fn: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        """Initializes the RecoveryPolicy.

        Args:
            logger: Optional custom logger. Defaults to 'shar.policy'.
            random_fn: Optional random number generator for jitter calculations.
                Defaults to random.uniform.
        """
        self._logger = logger or setup_logger("shar.policy")
        self._random_fn = random_fn or random.uniform

    def evaluate(
        self,
        task: Task,
        attempt_number: int,
        result: TaskResult,
        classification: ClassificationResult,
        diagnosis: Diagnosis,
        elapsed_recovery_seconds: float = 0.0,
    ) -> RecoveryDecision:
        """Determines the recovery action for a given attempt.

        Attempt Number Semantics:
            `attempt_number` is 1-based:
            - attempt_number = 1: The initial execution of the task.
            - attempt_number = 2: The 1st retry attempt (1 retry performed so far).
            - attempt_number = k: The (k - 1)-th retry attempt.
            `task.max_retries` defines the maximum allowed retries AFTER the initial attempt.
            Therefore, retry attempts are exhausted when `(attempt_number - 1) >= task.max_retries`.

        Args:
            task: The Task definition being executed.
            attempt_number: 1-based index of the current attempt. Must be >= 1.
            result: The TaskResult produced by TaskRunner.
            classification: Failure classification from FailureDetector.
            diagnosis: Diagnosis produced by DiagnosisEngine.
            elapsed_recovery_seconds: Cumulative wall-clock seconds elapsed in this
                orchestration across prior attempts and delays. Must be >= 0.

        Returns:
            A RecoveryDecision specifying action (RETRY, FALLBACK, STOP, ESCALATE),
            delay in seconds, safety status, and human-readable reason.
        """
        if attempt_number < 1:
            raise ValueError(f"attempt_number must be >= 1, got {attempt_number}")
        if elapsed_recovery_seconds < 0:
            raise ValueError(
                f"elapsed_recovery_seconds must be non-negative, got {elapsed_recovery_seconds}"
            )

        # 1. Success check: Successful execution never requests recovery
        if result.is_success:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=f"Task '{task.name}' succeeded; no recovery action required.",
                delay_seconds=0.0,
                is_safe_to_retry=False,
            )

        # 2. Gate 1: Failure category check (only TRANSIENT is eligible for retry)
        if classification.category == FailureCategory.PERMANENT:
            stop_reason = (
                f"Task '{task.name}' failed with permanent error '{diagnosis.root_cause}'. "
                "Retrying is not expected to resolve this failure."
            )
            return self._decide_fallback_or_stop(task, result, stop_reason)

        if classification.category == FailureCategory.UNKNOWN:
            stop_reason = (
                f"Task '{task.name}' failed with unknown error '{diagnosis.root_cause}'. "
                "Unknown failures are not retryable by default to prevent unsafe loops."
            )
            return self._decide_fallback_or_stop(task, result, stop_reason)

        # 3. Gate 2: Idempotency safety check
        if not task.is_idempotent:
            stop_reason = (
                f"Task '{task.name}' is not declared idempotent (is_idempotent=False). "
                "Retrying non-idempotent operations may cause duplicate side effects."
            )
            return self._decide_fallback_or_stop(task, result, stop_reason)

        # 4. Gate 3: Timeout safety check (underlying thread cannot be safely killed)
        if result.failure_type == FailureType.TIMEOUT and not task.safe_to_retry_on_timeout:
            stop_reason = (
                f"Task '{task.name}' timed out and safe_to_retry_on_timeout is False. "
                "In Python, running worker threads cannot be forcibly killed; rerunning "
                "risks duplicate concurrent execution and resource leaks."
            )
            return self._decide_fallback_or_stop(task, result, stop_reason)

        # 5. Gate 4: Retry limit boundary check
        # attempt_number=1 means initial run (0 retries so far).
        retries_completed = attempt_number - 1
        if retries_completed >= task.max_retries:
            stop_reason = (
                f"Task '{task.name}' has exhausted maximum retry limit of {task.max_retries} "
                f"(completed {retries_completed} retries after initial attempt)."
            )
            return self._decide_fallback_or_stop(task, result, stop_reason)

        # 6. Calculate proposed backoff delay
        # Retry index 0 is first retry (after attempt 1), retry index 1 is second retry, etc.
        delay = self.calculate_backoff_delay(task, retries_completed)

        # 7. Gate 5: Recovery budget verification
        if task.recovery_budget_seconds is not None:
            remaining_budget = task.recovery_budget_seconds - elapsed_recovery_seconds

            if remaining_budget <= 0:
                escalate_reason = (
                    f"Task '{task.name}' exhausted overall recovery budget of "
                    f"{task.recovery_budget_seconds}s (elapsed: {elapsed_recovery_seconds:.3f}s)."
                )
                self._logger.warning(
                    escalate_reason,
                    extra={"task_name": task.name, "correlation_id": result.correlation_id},
                )
                return RecoveryDecision(
                    action=RecoveryAction.ESCALATE,
                    reason=escalate_reason,
                    delay_seconds=0.0,
                    is_safe_to_retry=False,
                )

            if delay >= remaining_budget:
                escalate_reason = (
                    f"Task '{task.name}' remaining recovery budget ({remaining_budget:.3f}s) "
                    f"is insufficient for proposed backoff delay ({delay:.3f}s)."
                )
                self._logger.warning(
                    escalate_reason,
                    extra={"task_name": task.name, "correlation_id": result.correlation_id},
                )
                return RecoveryDecision(
                    action=RecoveryAction.ESCALATE,
                    reason=escalate_reason,
                    delay_seconds=0.0,
                    is_safe_to_retry=False,
                )

        # 8. All retry gates passed: Authorize RETRY
        next_retry_number = retries_completed + 1
        retry_reason = (
            f"Task '{task.name}' failed with transient error '{diagnosis.root_cause}'. "
            f"Retry {next_retry_number}/{task.max_retries} authorized with delay {delay:.3f}s."
        )
        self._logger.info(
            retry_reason,
            extra={
                "task_name": task.name,
                "correlation_id": result.correlation_id,
                "attempt_number": attempt_number,
                "delay_seconds": delay,
            },
        )
        return RecoveryDecision(
            action=RecoveryAction.RETRY,
            reason=retry_reason,
            delay_seconds=delay,
            is_safe_to_retry=True,
        )

    def calculate_backoff_delay(self, task: Task, retry_index: int) -> float:
        """Calculates deterministic exponential backoff delay capped at backoff_max_delay.

        Formula:
            nominal_delay = min(base_delay * (factor ** retry_index), max_delay)

        If task.jitter is True and nominal_delay > 0:
            delay = uniform(0.5 * nominal_delay, nominal_delay)

        Args:
            task: Task containing backoff parameters.
            retry_index: 0-based index of the retry (0 for 1st retry, 1 for 2nd retry, etc.).

        Returns:
            Calculated delay in seconds.
        """
        nominal_delay = task.backoff_base_delay * (task.backoff_factor ** retry_index)
        capped_delay = min(nominal_delay, task.backoff_max_delay)

        if task.jitter and capped_delay > 0.0:
            # Bounded equal jitter: minimum 50% of capped delay to prevent zero-sleep stampedes
            return self._random_fn(0.5 * capped_delay, capped_delay)

        return capped_delay

    def _decide_fallback_or_stop(
        self,
        task: Task,
        result: TaskResult,
        stop_reason: str,
    ) -> RecoveryDecision:
        """Determines whether a task that cannot be retried can safely route to a fallback.

        Conservative Fallback Safety Rules:
        1. `task.fallback` must be explicitly configured.
        2. If the failure was a TIMEOUT and `safe_to_retry_on_timeout` is False:
           The primary task's worker thread is still actively running in the background.
           Initiating a fallback while the original execution may still be producing side
           effects is UNSAFE. In this scenario, fallback is refused.
        3. If primary task is not declared idempotent, running a fallback after partial execution
           could introduce compounding side effects. However, if the caller explicitly provided
           `task.fallback`, it represents an intentional alternate strategy.

        Args:
            task: The primary Task.
            result: The TaskResult of the failed attempt.
            stop_reason: Explanation of why retry was rejected.

        Returns:
            RecoveryDecision with action FALLBACK (if safe) or STOP.
        """
        if task.fallback is None:
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=f"{stop_reason} No fallback task configured.",
                delay_seconds=0.0,
                is_safe_to_retry=False,
            )

        # Safety check: If a thread timed out and is not safe for concurrent side effects,
        # executing a fallback while the thread is still active is prohibited.
        if result.failure_type == FailureType.TIMEOUT and not task.safe_to_retry_on_timeout:
            unsafe_msg = (
                f"{stop_reason} Fallback task '{task.fallback.name}' refused because primary "
                "worker thread timed out and cannot be verified stopped."
            )
            self._logger.warning(
                unsafe_msg,
                extra={"task_name": task.name, "correlation_id": result.correlation_id},
            )
            return RecoveryDecision(
                action=RecoveryAction.STOP,
                reason=unsafe_msg,
                delay_seconds=0.0,
                is_safe_to_retry=False,
            )

        fallback_reason = (
            f"{stop_reason} Transitioning to designated fallback task '{task.fallback.name}'."
        )
        self._logger.info(
            fallback_reason,
            extra={
                "task_name": task.name,
                "fallback_task": task.fallback.name,
                "correlation_id": result.correlation_id,
            },
        )
        return RecoveryDecision(
            action=RecoveryAction.FALLBACK,
            reason=fallback_reason,
            delay_seconds=0.0,
            is_safe_to_retry=False,
            suggested_task=task.fallback,
        )

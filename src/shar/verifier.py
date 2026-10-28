"""Post-execution output verification for the Self-Healing AI Agent Runtime (shar).

This module provides a standalone, decoupled verification component (OutputVerifier)
that evaluates task output correctness independently of execution status, failure detection,
or recovery policies.
"""

from __future__ import annotations

import logging
from typing import Optional

from shar.logger import setup_logger
from shar.models import (
    Task,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)


class OutputVerifier:
    """Evaluates task output correctness against configured verifier callables.

    Responsibilities:
    - Distinguishes execution success from verified semantic correctness.
    - Evaluates `task.verifier` strictly on successful execution outputs.
    - Catches verifier exceptions conservatively without raising.
    - Rejects non-boolean return values (e.g. non-empty strings, integers).
    - Preserves correlation IDs and emits safe structured logs.
    - Never invokes the original task callable or initiates recovery loops.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the OutputVerifier.

        Args:
            logger: Custom logger instance. Defaults to 'shar.verifier'.
        """
        self._logger = logger or setup_logger("shar.verifier")

    def verify(
        self,
        task: Task,
        result: TaskResult,
        correlation_id: Optional[str] = None,
    ) -> VerificationResult:
        """Verifies the output of a task execution.

        Args:
            task: The Task definition that was executed.
            result: The TaskResult produced by TaskRunner.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            A VerificationResult detailing the explicit VerificationStatus,
            reasons, verifier name, and error messages if applicable.
        """
        cid = correlation_id or result.correlation_id or ""

        # 1. Execution failure check
        if not result.is_success:
            fail_reason = (
                f"Task '{task.name}' failed execution ({result.failure_type.value}: "
                f"{result.exception_message or 'no error message'}). "
                "Failed executions cannot be verified."
            )
            self._logger.info(
                f"Verification skipped for failed task '{task.name}'",
                extra={"task_name": task.name, "correlation_id": cid},
            )
            return VerificationResult(
                status=VerificationStatus.FAILED,
                is_verified=False,
                reason=fail_reason,
                verifier_name="none",
                correlation_id=cid,
            )

        # 2. Missing verifier check
        if task.verifier is None:
            unverified_reason = (
                f"Task '{task.name}' completed execution without a configured verifier. "
                "Output correctness cannot be determined."
            )
            self._logger.info(
                f"Task '{task.name}' has no verifier configured; marked UNVERIFIED",
                extra={"task_name": task.name, "correlation_id": cid},
            )
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                is_verified=False,
                reason=unverified_reason,
                verifier_name="none",
                correlation_id=cid,
            )

        verifier_name = getattr(task.verifier, "__name__", "custom_verifier")

        # 3. Execute verifier under strict exception containment
        try:
            verdict = task.verifier(result.output)
        except Exception as exc:
            err_msg = f"Verifier '{verifier_name}' raised an exception: {exc}"
            self._logger.error(
                err_msg,
                extra={"task_name": task.name, "correlation_id": cid},
                exc_info=True,
            )
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                is_verified=False,
                reason=err_msg,
                verifier_name=verifier_name,
                error=str(exc),
                correlation_id=cid,
            )

        # 4. Strict literal Boolean verification
        # In Python, bool is a subclass of int. type(verdict) is bool ensures only True and False are accepted.
        if type(verdict) is not bool:
            err_msg = (
                f"Verifier '{verifier_name}' returned unsupported type '{type(verdict).__name__}'; "
                "expected literal bool (True or False)."
            )
            self._logger.warning(
                err_msg,
                extra={"task_name": task.name, "correlation_id": cid},
            )
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                is_verified=False,
                reason=err_msg,
                verifier_name=verifier_name,
                error=f"UnsupportedReturnType: {type(verdict).__name__}",
                correlation_id=cid,
            )

        if verdict is True:
            self._logger.info(
                f"Task '{task.name}' output successfully verified by '{verifier_name}'",
                extra={"task_name": task.name, "correlation_id": cid},
            )
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                is_verified=True,
                reason=f"Output for task '{task.name}' satisfied verification check '{verifier_name}'.",
                verifier_name=verifier_name,
                correlation_id=cid,
            )
        else:
            self._logger.warning(
                f"Task '{task.name}' output failed verification check '{verifier_name}'",
                extra={"task_name": task.name, "correlation_id": cid},
            )
            return VerificationResult(
                status=VerificationStatus.FAILED,
                is_verified=False,
                reason=f"Output for task '{task.name}' failed verification check '{verifier_name}'.",
                verifier_name=verifier_name,
                correlation_id=cid,
            )

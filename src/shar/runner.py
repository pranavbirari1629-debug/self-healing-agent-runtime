"""Task execution engine with timeout enforcement, validation, and error capture.

IMPORTANT SAFETY NOTICE:
In CPython, worker threads cannot be forcibly killed or safely terminated externally.
When a Task with `timeout_seconds` exceeds its allocated time limit:
1. `TaskRunner.run` ceases waiting for the worker and returns a `TaskResult` with
   `FailureType.TIMEOUT`.
2. The underlying Python worker thread continues executing in the background until its
   target function completes or the process exits.
3. The timed-out task is NOT cancelled and must NOT be assumed to have terminated.
4. No automatic retries occur in Phase 1.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
import traceback
import uuid
from typing import Any, Optional

from shar.logger import setup_logger
from shar.models import FailureType, Task, TaskOutcome, TaskResult


class TaskRunner:
    """Executes Tasks, enforces timeouts, captures exceptions, and validates outputs.

    Thread Safety & Timeout Semantics:
        Tasks configured with `timeout_seconds` are dispatched to an internal
        `concurrent.futures.ThreadPoolExecutor`. If a task runs longer than
        `timeout_seconds`, the runner unblocks and flags the result as a TIMEOUT
        failure. However, the thread remains active in Python's thread pool until
        the callable naturally finishes.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 4,
    ) -> None:
        """Initializes the TaskRunner.

        Args:
            logger: Optional custom logger. Defaults to JSON-configured 'shar.runner'.
            max_workers: Maximum worker threads for running timed tasks.
        """
        self._logger = logger or setup_logger("shar.runner")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="shar-runner-worker",
        )

    def close(self) -> None:
        """Shuts down the internal executor pool."""
        self._executor.shutdown(wait=False)

    def __enter__(self) -> TaskRunner:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def run(self, task: Task, correlation_id: Optional[str] = None) -> TaskResult:
        """Executes a single task attempt, applies validation, and logs the outcome.

        Args:
            task: The Task definition to execute.
            correlation_id: Optional tracking identifier. A UUID is generated if omitted.

        Returns:
            A TaskResult containing execution outcome, captured values, diagnostics,
            and timing data.
        """
        cid = correlation_id or uuid.uuid4().hex
        start_time = time.perf_counter()

        self._logger.info(
            f"Starting execution of task '{task.name}'",
            extra={"correlation_id": cid, "task_name": task.name},
        )

        try:
            if task.timeout_seconds is not None and task.timeout_seconds > 0:
                # Dispatch to thread pool to enforce deadline
                future = self._executor.submit(
                    task.callable, *task.args, **task.kwargs
                )
                try:
                    raw_output = future.result(timeout=task.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    timeout_msg = (
                        f"Task '{task.name}' timed out after {task.timeout_seconds}s. "
                        "Warning: Underlying thread was not terminated and may continue "
                        "running in the background."
                    )
                    self._logger.warning(
                        timeout_msg,
                        extra={
                            "correlation_id": cid,
                            "task_name": task.name,
                            "failure_type": FailureType.TIMEOUT.value,
                            "execution_time_ms": elapsed_ms,
                        },
                    )
                    return TaskResult(
                        task_name=task.name,
                        outcome=TaskOutcome.FAILURE,
                        output=None,
                        failure_type=FailureType.TIMEOUT,
                        exception_type="TimeoutError",
                        exception_message=timeout_msg,
                        traceback_str=None,
                        execution_time_ms=elapsed_ms,
                        correlation_id=cid,
                    )
            else:
                # Execute synchronously in caller thread when no timeout is requested
                raw_output = task.callable(*task.args, **task.kwargs)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            tb_str = traceback.format_exc()
            self._logger.error(
                f"Task '{task.name}' raised an exception: {exc}",
                extra={
                    "correlation_id": cid,
                    "task_name": task.name,
                    "failure_type": FailureType.EXCEPTION.value,
                    "execution_time_ms": elapsed_ms,
                },
                exc_info=True,
            )
            return TaskResult(
                task_name=task.name,
                outcome=TaskOutcome.FAILURE,
                output=None,
                failure_type=FailureType.EXCEPTION,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                traceback_str=tb_str,
                execution_time_ms=elapsed_ms,
                correlation_id=cid,
            )

        # Output validation step
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if task.validator is not None:
            try:
                is_valid = bool(task.validator(raw_output))
            except Exception as val_exc:
                val_tb = traceback.format_exc()
                val_msg = f"Task validator raised an exception: {val_exc}"
                self._logger.error(
                    val_msg,
                    extra={
                        "correlation_id": cid,
                        "task_name": task.name,
                        "failure_type": FailureType.VALIDATION_FAILED.value,
                        "execution_time_ms": elapsed_ms,
                    },
                    exc_info=True,
                )
                return TaskResult(
                    task_name=task.name,
                    outcome=TaskOutcome.FAILURE,
                    output=raw_output,
                    failure_type=FailureType.VALIDATION_FAILED,
                    exception_type=type(val_exc).__name__,
                    exception_message=val_msg,
                    traceback_str=val_tb,
                    execution_time_ms=elapsed_ms,
                    correlation_id=cid,
                )

            if not is_valid:
                val_fail_msg = (
                    f"Output validation failed for task '{task.name}'. "
                    f"Returned value was: {raw_output!r}"
                )
                self._logger.warning(
                    val_fail_msg,
                    extra={
                        "correlation_id": cid,
                        "task_name": task.name,
                        "failure_type": FailureType.VALIDATION_FAILED.value,
                        "execution_time_ms": elapsed_ms,
                    },
                )
                return TaskResult(
                    task_name=task.name,
                    outcome=TaskOutcome.FAILURE,
                    output=raw_output,
                    failure_type=FailureType.VALIDATION_FAILED,
                    exception_type="ValidationError",
                    exception_message=val_fail_msg,
                    traceback_str=None,
                    execution_time_ms=elapsed_ms,
                    correlation_id=cid,
                )

        # Execution and validation succeeded
        self._logger.info(
            f"Task '{task.name}' completed successfully",
            extra={
                "correlation_id": cid,
                "task_name": task.name,
                "outcome": TaskOutcome.SUCCESS.value,
                "execution_time_ms": elapsed_ms,
            },
        )
        return TaskResult(
            task_name=task.name,
            outcome=TaskOutcome.SUCCESS,
            output=raw_output,
            failure_type=FailureType.NONE,
            exception_type=None,
            exception_message=None,
            traceback_str=None,
            execution_time_ms=elapsed_ms,
            correlation_id=cid,
        )

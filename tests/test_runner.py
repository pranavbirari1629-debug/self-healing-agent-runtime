"""Unit tests for TaskRunner in shar.runner."""

import time
import pytest

from shar.models import FailureType, Task, TaskOutcome
from shar.runner import TaskRunner


@pytest.fixture
def runner():
    """Provides a TaskRunner instance and ensures clean shutdown."""
    r = TaskRunner()
    yield r
    r.close()


def test_successful_task_returns_result(runner):
    """Verifies that a basic task executes and returns expected output and SUCCESS outcome."""
    task = Task(name="simple_add", callable=lambda a, b: a + b, args=(3, 5))
    result = runner.run(task)

    assert result.is_success
    assert result.outcome == TaskOutcome.SUCCESS
    assert result.output == 8
    assert result.failure_type == FailureType.NONE
    assert result.exception_type is None
    assert result.execution_time_ms >= 0.0
    assert result.correlation_id != ""


def test_successful_task_with_validator_pass(runner):
    """Verifies that a task whose output satisfies the validator succeeds."""
    task = Task(
        name="valid_payload",
        callable=lambda: {"status": "ok", "items": [1, 2, 3]},
        validator=lambda data: isinstance(data, dict) and data.get("status") == "ok",
    )
    result = runner.run(task)

    assert result.is_success
    assert result.outcome == TaskOutcome.SUCCESS
    assert result.output == {"status": "ok", "items": [1, 2, 3]}
    assert result.failure_type == FailureType.NONE


def test_successful_task_with_validator_fail(runner):
    """Verifies that execution success without passing validation produces a VALIDATION_FAILED failure."""
    task = Task(
        name="invalid_payload",
        callable=lambda: {"status": "error", "items": []},
        validator=lambda data: data.get("status") == "ok",
    )
    result = runner.run(task)

    assert not result.is_success
    assert result.outcome == TaskOutcome.FAILURE
    assert result.failure_type == FailureType.VALIDATION_FAILED
    assert result.output == {"status": "error", "items": []}
    assert "Output validation failed" in (result.exception_message or "")


def test_failing_task_captures_exception(runner):
    """Verifies that an exception raised by the callable is captured with diagnostics."""
    def blow_up():
        raise ValueError("Invalid configuration parameter")

    task = Task(name="faulty_task", callable=blow_up)
    result = runner.run(task)

    assert not result.is_success
    assert result.outcome == TaskOutcome.FAILURE
    assert result.failure_type == FailureType.EXCEPTION
    assert result.exception_type == "ValueError"
    assert "Invalid configuration parameter" in (result.exception_message or "")
    assert result.traceback_str is not None
    assert "ValueError: Invalid configuration parameter" in result.traceback_str


def test_timeout_task_returns_timeout_failure(runner):
    """Verifies that tasks exceeding timeout_seconds produce a TIMEOUT failure result."""
    def slow_worker():
        time.sleep(0.5)
        return "finished"

    # Enforce a short timeout of 0.05 seconds
    task = Task(name="slow_task", callable=slow_worker, timeout_seconds=0.05)
    result = runner.run(task)

    assert not result.is_success
    assert result.outcome == TaskOutcome.FAILURE
    assert result.failure_type == FailureType.TIMEOUT
    assert result.exception_type == "TimeoutError"
    assert "timed out after 0.05s" in (result.exception_message or "")
    # Verifies safety requirement: documentation in result that thread is not killed
    assert "not terminated" in (result.exception_message or "")


def test_task_without_timeout_runs_normally(runner):
    """Verifies that a task with timeout_seconds=None runs directly to completion."""
    task = Task(
        name="no_timeout_task",
        callable=lambda msg: f"hello {msg}",
        args=("world",),
        timeout_seconds=None,
    )
    result = runner.run(task)

    assert result.is_success
    assert result.output == "hello world"
    assert result.failure_type == FailureType.NONE


def test_validator_exception_handled_gracefully(runner):
    """Verifies that an exception raised inside the validator is safely captured."""
    def buggy_validator(output):
        raise KeyError("missing_key")

    task = Task(
        name="validator_crash",
        callable=lambda: "some_data",
        validator=buggy_validator,
    )
    result = runner.run(task)

    assert not result.is_success
    assert result.outcome == TaskOutcome.FAILURE
    assert result.failure_type == FailureType.VALIDATION_FAILED
    assert result.exception_type == "KeyError"
    assert "missing_key" in (result.exception_message or "")
    assert result.traceback_str is not None


def test_custom_correlation_id_propagated(runner):
    """Verifies that an externally provided correlation_id is retained in the result."""
    task = Task(name="correlated_task", callable=lambda: 42)
    custom_cid = "test-corr-id-12345"
    result = runner.run(task, correlation_id=custom_cid)

    assert result.is_success
    assert result.correlation_id == custom_cid

"""Tests for the Deterministic Benchmark Harness and Metric Invariants.

Verifies mathematical invariants, denominator safety, outcome separation,
and the equivalence property between Baselines B and C.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_benchmark import (
    BenchmarkMetrics,
    MockClock,
    MockSleeper,
    build_benchmark_scenarios,
    calculate_benchmark_metrics,
    run_baseline_a,
    run_baseline_b,
    run_baseline_c,
    run_full_benchmark,
)
from shar.models import (
    AttemptRecord,
    FailureType,
    RecoveryResult,
    RecoveryStatus,
    TaskOutcome,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)


def _make_dummy_result(
    task_name: str,
    status: RecoveryStatus,
    is_success: bool,
    attempts_outcomes: list[bool],
    verification_status: VerificationStatus,
    fallback_attempted: bool = False,
) -> RecoveryResult:
    """Helper creating a synthetic RecoveryResult for targeted metric testing."""
    attempt_records: list[AttemptRecord] = []
    for idx, succ in enumerate(attempts_outcomes, start=1):
        tname = f"{task_name}_fallback" if (fallback_attempted and idx > 1) else task_name
        tres = TaskResult(
            task_name=tname,
            outcome=TaskOutcome.SUCCESS if succ else TaskOutcome.FAILURE,
            output="ok" if succ else None,
            failure_type=FailureType.NONE if succ else FailureType.EXCEPTION,
        )
        attempt_records.append(
            AttemptRecord(
                attempt_number=idx,
                task_name=tname,
                task_result=tres,
                elapsed_ms=1.0,
            )
        )

    last_tres = attempt_records[-1].task_result if attempt_records else TaskResult(
        task_name=task_name,
        outcome=TaskOutcome.SUCCESS if is_success else TaskOutcome.FAILURE,
    )

    ver_res = VerificationResult(
        status=verification_status,
        is_verified=(verification_status == VerificationStatus.VERIFIED),
        reason=f"Status: {verification_status.value}",
    )

    return RecoveryResult(
        task_name=task_name,
        status=status,
        final_result=last_tres,
        attempts=attempt_records,
        total_attempts=len(attempt_records),
        total_elapsed_ms=10.0,
        correlation_id=f"test-{task_name}",
        verification=ver_res,
    )


# ===========================================================================
# Metric Invariant Tests
# ===========================================================================

def test_verified_success_rate_cannot_exceed_execution_success_rate():
    """Mathematical Invariant: verified_success_rate <= execution_success_rate."""
    results = [
        _make_dummy_result("t1", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.VERIFIED),
        _make_dummy_result("t2", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.UNVERIFIED),
        _make_dummy_result("t3", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.FAILED),
        _make_dummy_result("t4", RecoveryStatus.FAILED_PERMANENT, False, [False], VerificationStatus.FAILED),
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.execution_success_count == 3
    assert metrics.verified_success_count == 1
    assert metrics.verified_success_rate <= metrics.execution_success_rate
    assert metrics.verified_success_rate == pytest.approx(0.25)
    assert metrics.execution_success_rate == pytest.approx(0.75)


def test_successful_execution_with_failed_verification_counts_as_execution_success():
    """Execution success with FAILED verification increments execution count, not verified count."""
    results = [
        _make_dummy_result("t_fail_ver", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.FAILED)
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.execution_success_count == 1
    assert metrics.execution_success_rate == 1.0
    assert metrics.verified_success_count == 0
    assert metrics.verified_success_rate == 0.0
    assert metrics.verification_failed_count == 1


def test_unverified_not_counted_as_verified():
    """UNVERIFIED status counts towards execution success, but never verified success."""
    results = [
        _make_dummy_result("t_unver", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.UNVERIFIED)
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.execution_success_count == 1
    assert metrics.verified_success_count == 0
    assert metrics.verified_success_rate == 0.0
    assert metrics.verification_unverified_count == 1


def test_recovery_success_does_not_imply_verified_success():
    """A workflow that recovers execution but fails verification is counted as recovered, not verified."""
    results = [
        _make_dummy_result(
            "t_rec_unver",
            RecoveryStatus.RECOVERED_VIA_RETRY,
            True,
            [False, True],
            VerificationStatus.FAILED,
        )
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.eligible_recovery_count == 1
    assert metrics.successful_recovery_count == 1
    assert metrics.recovery_success_rate == 1.0
    assert metrics.execution_success_count == 1
    assert metrics.verified_success_count == 0
    assert metrics.verified_success_rate == 0.0
    assert metrics.verification_failed_count == 1


def test_retry_exhaustion_and_budget_exhaustion_statuses():
    """Asserts that retry exhaustion and budget exhaustion counts use exact terminal statuses."""
    results = [
        _make_dummy_result("t_exh", RecoveryStatus.FAILED_RETRIES_EXHAUSTED, False, [False, False], VerificationStatus.FAILED),
        _make_dummy_result("t_bud", RecoveryStatus.FAILED_BUDGET_EXCEEDED, False, [False], VerificationStatus.FAILED),
        _make_dummy_result("t_perm", RecoveryStatus.FAILED_PERMANENT, False, [False], VerificationStatus.FAILED),
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.retry_exhaustion_count == 1
    assert metrics.budget_exhaustion_count == 1
    assert metrics.total_workflows == 3


def test_fallback_success_rate_uses_correct_denominator():
    """Fallback success rate denominator must count only workflows that actually attempted fallback."""
    results = [
        # 1. Fallback attempted and SUCCEEDED
        _make_dummy_result("t_fb_ok", RecoveryStatus.RECOVERED_VIA_FALLBACK, True, [False, True], VerificationStatus.VERIFIED, fallback_attempted=True),
        # 2. Fallback attempted and FAILED (escalated)
        _make_dummy_result("t_fb_fail", RecoveryStatus.ESCALATED, False, [False, False], VerificationStatus.FAILED, fallback_attempted=True),
        # 3. Retry workflow (never attempted fallback)
        _make_dummy_result("t_retry", RecoveryStatus.FAILED_RETRIES_EXHAUSTED, False, [False, False], VerificationStatus.FAILED, fallback_attempted=False),
        # 4. First-attempt success (never attempted fallback)
        _make_dummy_result("t_direct", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.VERIFIED, fallback_attempted=False),
    ]
    metrics = calculate_benchmark_metrics(results)

    assert metrics.attempted_fallback_count == 2
    assert metrics.successful_fallback_count == 1
    assert metrics.fallback_success_rate == pytest.approx(0.5)


def test_zero_denominators_handled_safely():
    """Guarantees zero division is avoided when denominators are zero."""
    # Case 1: Empty results
    metrics_empty = calculate_benchmark_metrics([])
    assert metrics_empty.total_workflows == 0
    assert metrics_empty.execution_success_rate == 0.0
    assert metrics_empty.verified_success_rate == 0.0
    assert metrics_empty.recovery_success_rate == 0.0
    assert metrics_empty.fallback_success_rate == 0.0
    assert metrics_empty.average_execution_attempts == 0.0

    # Case 2: No failures (0 workflows eligible for recovery)
    results_no_failures = [
        _make_dummy_result("t_ok", RecoveryStatus.SUCCEEDED_FIRST_ATTEMPT, True, [True], VerificationStatus.VERIFIED)
    ]
    metrics_no_failures = calculate_benchmark_metrics(results_no_failures)
    assert metrics_no_failures.eligible_recovery_count == 0
    assert metrics_no_failures.recovery_success_rate == 0.0

    # Case 3: No fallbacks attempted
    results_no_fallback = [
        _make_dummy_result("t_rec", RecoveryStatus.RECOVERED_VIA_RETRY, True, [False, True], VerificationStatus.VERIFIED, fallback_attempted=False)
    ]
    metrics_no_fallback = calculate_benchmark_metrics(results_no_fallback)
    assert metrics_no_fallback.attempted_fallback_count == 0
    assert metrics_no_fallback.fallback_success_rate == 0.0


# ===========================================================================
# Baseline B vs C Invariance Integration Tests
# ===========================================================================

def test_baselines_b_and_c_produce_identical_outcomes():
    """Architectural Invariant: Baselines B and C must yield identical execution and verification outcomes.

    Proves that enabling observational RecoveryMemory does not alter:
    - Task execution outcomes
    - Verification outcomes
    - Attempt counts
    - Terminal recovery statuses
    """
    scenarios = build_benchmark_scenarios()

    results_b, metrics_b = run_baseline_b(scenarios)
    results_c, metrics_c, memory_c = run_baseline_c(scenarios)

    assert len(results_b) == len(results_c) == len(scenarios)

    # Per-scenario outcome equivalence
    for rb, rc in zip(results_b, results_c):
        assert rb.task_name == rc.task_name
        assert rb.status == rc.status
        assert rb.is_success == rc.is_success
        assert rb.is_verified == rc.is_verified
        assert rb.total_attempts == rc.total_attempts
        if rb.verification is not None:
            assert rc.verification is not None
            assert rb.verification.status == rc.verification.status

    # Aggregate metric equivalence
    assert metrics_b.total_workflows == metrics_c.total_workflows
    assert metrics_b.execution_success_count == metrics_c.execution_success_count
    assert metrics_b.execution_success_rate == metrics_c.execution_success_rate
    assert metrics_b.verified_success_count == metrics_c.verified_success_count
    assert metrics_b.verified_success_rate == metrics_c.verified_success_rate
    assert metrics_b.recovery_success_rate == metrics_c.recovery_success_rate
    assert metrics_b.fallback_success_rate == metrics_c.fallback_success_rate
    assert metrics_b.average_execution_attempts == metrics_c.average_execution_attempts

    # The ONLY expected divergence is that memory C recorded entries
    assert metrics_b.memory_entries_recorded == 0
    assert metrics_c.memory_entries_recorded > 0
    assert len(memory_c) == metrics_c.memory_entries_recorded


def test_full_benchmark_generation_and_json_structure(tmp_path):
    """Verifies that run_full_benchmark creates valid, complete JSON report."""
    report = run_full_benchmark(output_dir=tmp_path)

    json_file = tmp_path / "benchmark_results.json"
    assert json_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify structural keys
    assert "benchmark_metadata" in data
    assert "metric_definitions" in data
    assert "scenarios" in data
    assert "baselines" in data
    assert "comparison_summary" in data

    # Verify baselines present
    assert "baseline_a_no_recovery" in data["baselines"]
    assert "baseline_b_self_healing" in data["baselines"]
    assert "baseline_c_observational_memory" in data["baselines"]

    # Verify invariance property recorded
    assert data["comparison_summary"]["b_and_c_identical_execution"] is True
    assert data["comparison_summary"]["b_and_c_identical_verification"] is True

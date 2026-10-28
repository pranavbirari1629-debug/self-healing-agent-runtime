"""Deterministic unit tests for RecoveryMemory in shar.memory."""

import dataclasses
import threading
import time
import pytest

from shar.models import (
    ConfidenceLevel,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryStatus,
    VerificationStatus,
)
from shar.memory import (
    MemoryRecommendation,
    RecoveryMemory,
    RecoveryMemoryEntry,
)


def _make_entry(
    entry_id: str = "entry-001",
    task_name: str = "payment_task",
    failure_type: FailureType = FailureType.EXCEPTION,
    failure_category: FailureCategory = FailureCategory.TRANSIENT,
    root_cause: str = "network_timeout",
    exception_type: str | None = "ConnectionError",
    recovery_action: RecoveryAction = RecoveryAction.RETRY,
    recovered_execution: bool = True,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    attempt_count: int = 2,
    terminal_status: RecoveryStatus = RecoveryStatus.RECOVERED_VIA_RETRY,
    fallback_task_name: str | None = None,
    timestamp: float = 100.0,
) -> RecoveryMemoryEntry:
    """Helper to construct valid RecoveryMemoryEntry instances with sensible defaults."""
    return RecoveryMemoryEntry(
        entry_id=entry_id,
        task_name=task_name,
        failure_type=failure_type,
        failure_category=failure_category,
        root_cause=root_cause,
        exception_type=exception_type,
        recovery_action=recovery_action,
        recovered_execution=recovered_execution,
        verification_status=verification_status,
        attempt_count=attempt_count,
        terminal_status=terminal_status,
        fallback_task_name=fallback_task_name,
        timestamp=timestamp,
    )


# ---- 1. Empty memory ----

def test_empty_memory():
    """Verifies behavior when memory is empty."""
    memory = RecoveryMemory()

    assert len(memory) == 0
    assert memory.find_similar("payment_task", "network_timeout") == []

    rec = memory.recommend("payment_task", "network_timeout")
    assert rec.suggested_action is None
    assert rec.confidence == ConfidenceLevel.LOW
    assert rec.sample_size == 0
    assert rec.verified_success_count == 0
    assert rec.unverified_success_count == 0
    assert rec.failure_count == 0
    assert "No historical recovery entries found" in rec.rationale


# ---- 2. Remembering and retrieving entries ----

def test_remember_and_retrieve_entry():
    """Verifies that a valid entry is stored and retrieved with exact field fidelity."""
    memory = RecoveryMemory()
    entry = _make_entry(
        entry_id="rec-42",
        task_name="order_service",
        failure_type=FailureType.EXCEPTION,
        failure_category=FailureCategory.TRANSIENT,
        root_cause="connection_interrupted",
        exception_type="BrokenPipeError",
        recovery_action=RecoveryAction.RETRY,
        recovered_execution=True,
        verification_status=VerificationStatus.VERIFIED,
        attempt_count=2,
        terminal_status=RecoveryStatus.RECOVERED_VIA_RETRY,
        timestamp=105.0,
    )

    memory.remember(entry)
    assert len(memory) == 1

    results = memory.find_similar("order_service", "connection_interrupted", exception_type="BrokenPipeError")
    assert len(results) == 1
    retrieved = results[0]

    assert retrieved.entry_id == "rec-42"
    assert retrieved.task_name == "order_service"
    assert retrieved.failure_type == FailureType.EXCEPTION
    assert retrieved.failure_category == FailureCategory.TRANSIENT
    assert retrieved.root_cause == "connection_interrupted"
    assert retrieved.exception_type == "BrokenPipeError"
    assert retrieved.recovery_action == RecoveryAction.RETRY
    assert retrieved.recovered_execution is True
    assert retrieved.verification_status == VerificationStatus.VERIFIED
    assert retrieved.attempt_count == 2
    assert retrieved.terminal_status == RecoveryStatus.RECOVERED_VIA_RETRY
    assert retrieved.fallback_task_name is None
    assert retrieved.timestamp == 105.0


# ---- 3. Exact matching ----

def test_exact_matching():
    """Verifies that find_similar isolates entries by task_name and root_cause."""
    memory = RecoveryMemory()
    e1 = _make_entry(entry_id="e1", task_name="task_a", root_cause="cause_1")
    e2 = _make_entry(entry_id="e2", task_name="task_a", root_cause="cause_2")
    e3 = _make_entry(entry_id="e3", task_name="task_b", root_cause="cause_1")

    memory.remember(e1)
    memory.remember(e2)
    memory.remember(e3)

    results = memory.find_similar("task_a", "cause_1")
    assert len(results) == 1
    assert results[0].entry_id == "e1"


# ---- 4. Exception-type matching behavior ----

def test_exception_type_matching():
    """Verifies exception-type filtering and exact-match sorting."""
    memory = RecoveryMemory()
    e_conn = _make_entry(entry_id="e_conn", task_name="task_x", root_cause="net_err", exception_type="ConnectionError")
    e_os = _make_entry(entry_id="e_os", task_name="task_x", root_cause="net_err", exception_type="OSError")
    e_none = _make_entry(entry_id="e_none", task_name="task_x", root_cause="net_err", exception_type=None)

    memory.remember(e_os)
    memory.remember(e_none)
    memory.remember(e_conn)

    # Query with specific exception type: exact match prioritized over None; different exception excluded
    matches = memory.find_similar("task_x", "net_err", exception_type="ConnectionError")
    assert [e.entry_id for e in matches] == ["e_conn", "e_none"]

    # Query without exception type: matches all entries with same task and root_cause
    all_matches = memory.find_similar("task_x", "net_err", exception_type=None)
    assert len(all_matches) == 3


# ---- 5. Bounded result count ----

def test_bounded_result_count():
    """Verifies that query limit strictly truncates returned results."""
    memory = RecoveryMemory()
    for i in range(10):
        memory.remember(_make_entry(entry_id=f"e-{i}", task_name="t", root_cause="rc", timestamp=float(i)))

    results = memory.find_similar("t", "rc", limit=3)
    assert len(results) == 3


# ---- 6. Invalid query limits ----

def test_invalid_query_limits():
    """Verifies that non-positive query limits raise ValueError."""
    memory = RecoveryMemory()
    with pytest.raises(ValueError, match="limit must be positive"):
        memory.find_similar("t", "rc", limit=0)

    with pytest.raises(ValueError, match="limit must be positive"):
        memory.find_similar("t", "rc", limit=-5)


# ---- 7. clear() ----

def test_clear():
    """Verifies that clear() removes all stored entries."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(entry_id="e1"))
    memory.remember(_make_entry(entry_id="e2"))
    assert len(memory) == 2

    memory.clear()
    assert len(memory) == 0
    assert memory.find_similar("payment_task", "network_timeout") == []


# ---- 8. FIFO eviction at capacity ----

def test_fifo_eviction_at_capacity():
    """Verifies that exceeding max_entries evicts the oldest entry first."""
    memory = RecoveryMemory(max_entries=3)
    memory.remember(_make_entry(entry_id="e1", timestamp=10.0))
    memory.remember(_make_entry(entry_id="e2", timestamp=20.0))
    memory.remember(_make_entry(entry_id="e3", timestamp=30.0))
    assert len(memory) == 3

    # Adding 4th entry must evict e1
    memory.remember(_make_entry(entry_id="e4", timestamp=40.0))
    assert len(memory) == 3

    ids = [e.entry_id for e in memory.find_similar("payment_task", "network_timeout", limit=10)]
    assert "e1" not in ids
    assert set(ids) == {"e2", "e3", "e4"}


# ---- 9. Stable deterministic ordering ----

def test_stable_deterministic_ordering():
    """Verifies ordering: exact exception matches, verified successes, newer timestamps, entry_id."""
    memory = RecoveryMemory()
    # e_old_ver: verified success, older timestamp
    e_old_ver = _make_entry(entry_id="e1", task_name="t", root_cause="rc", exception_type="ConnErr",
                            verification_status=VerificationStatus.VERIFIED, timestamp=100.0)
    # e_new_unver: unverified success, newer timestamp
    e_new_unver = _make_entry(entry_id="e2", task_name="t", root_cause="rc", exception_type="ConnErr",
                              verification_status=VerificationStatus.UNVERIFIED, timestamp=200.0)
    # e_new_ver: verified success, newer timestamp
    e_new_ver = _make_entry(entry_id="e3", task_name="t", root_cause="rc", exception_type="ConnErr",
                            verification_status=VerificationStatus.VERIFIED, timestamp=300.0)

    memory.remember(e_old_ver)
    memory.remember(e_new_unver)
    memory.remember(e_new_ver)

    results = memory.find_similar("t", "rc", exception_type="ConnErr", limit=5)
    # e3 (verified, 300.0) -> e1 (verified, 100.0) -> e2 (unverified, 200.0)
    assert [e.entry_id for e in results] == ["e3", "e1", "e2"]


# ---- 10. One verified success not producing HIGH confidence ----

def test_one_verified_success_not_producing_high_confidence():
    """Rule: A single verified success cannot produce HIGH confidence (produces MEDIUM)."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(
        entry_id="single_succ",
        task_name="checkout",
        root_cause="gateway_timeout",
        recovery_action=RecoveryAction.RETRY,
        recovered_execution=True,
        verification_status=VerificationStatus.VERIFIED,
    ))

    rec = memory.recommend("checkout", "gateway_timeout")
    assert rec.suggested_action == RecoveryAction.RETRY
    assert rec.confidence == ConfidenceLevel.MEDIUM
    assert rec.sample_size == 1
    assert rec.verified_success_count == 1
    assert rec.failure_count == 0
    assert "minimum 2 verified successes required for HIGH" in rec.rationale


# ---- 11. Multiple verified successes produce HIGH confidence ----

def test_multiple_verified_successes_produce_high_confidence():
    """Rule: 2 or more verified successes with 0 failures produces HIGH confidence."""
    memory = RecoveryMemory()
    for i in range(3):
        memory.remember(_make_entry(
            entry_id=f"succ_{i}",
            task_name="checkout",
            root_cause="gateway_timeout",
            recovery_action=RecoveryAction.RETRY,
            recovered_execution=True,
            verification_status=VerificationStatus.VERIFIED,
        ))

    rec = memory.recommend("checkout", "gateway_timeout")
    assert rec.suggested_action == RecoveryAction.RETRY
    assert rec.confidence == ConfidenceLevel.HIGH
    assert rec.sample_size == 3
    assert rec.verified_success_count == 3
    assert rec.failure_count == 0
    assert "Strong consensus" in rec.rationale


# ---- 12. UNVERIFIED execution recovery counted separately ----

def test_unverified_execution_recovery_counted_separately():
    """Rule: Unverified recoveries are counted separately and can never confer HIGH confidence."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(
        entry_id="unver_1",
        task_name="sync_task",
        root_cause="socket_error",
        recovery_action=RecoveryAction.RETRY,
        recovered_execution=True,
        verification_status=VerificationStatus.UNVERIFIED,
    ))
    memory.remember(_make_entry(
        entry_id="unver_2",
        task_name="sync_task",
        root_cause="socket_error",
        recovery_action=RecoveryAction.RETRY,
        recovered_execution=True,
        verification_status=VerificationStatus.UNVERIFIED,
    ))

    rec = memory.recommend("sync_task", "socket_error")
    assert rec.suggested_action == RecoveryAction.RETRY
    assert rec.confidence == ConfidenceLevel.LOW
    assert rec.verified_success_count == 0
    assert rec.unverified_success_count == 2
    assert rec.failure_count == 0
    assert "unverified recovery attempt(s)" in rec.rationale


# ---- 13. FAILED verification never counted as proven success ----

def test_failed_verification_never_counted_as_proven_success():
    """Rule: Recovered execution whose verification failed is counted as a failure, never a verified success."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(
        entry_id="ver_fail",
        task_name="data_fetch",
        root_cause="schema_error",
        recovery_action=RecoveryAction.RETRY,
        recovered_execution=True,
        verification_status=VerificationStatus.FAILED,
    ))

    rec = memory.recommend("data_fetch", "schema_error")
    assert rec.suggested_action is None
    assert rec.confidence == ConfidenceLevel.LOW
    assert rec.verified_success_count == 0
    assert rec.unverified_success_count == 0
    assert rec.failure_count == 1
    assert "failed verification" in rec.rationale


# ---- 14. INCONCLUSIVE verification never counted as proven success ----

def test_inconclusive_verification_never_counted_as_proven_success():
    """Rule: Inconclusive verification is counted as a failure, never a verified success."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(
        entry_id="inconcl_entry",
        task_name="data_fetch",
        root_cause="schema_error",
        recovery_action=RecoveryAction.FALLBACK,
        recovered_execution=True,
        verification_status=VerificationStatus.INCONCLUSIVE,
        fallback_task_name="backup_fetch",
    ))

    rec = memory.recommend("data_fetch", "schema_error")
    assert rec.suggested_action is None
    assert rec.confidence == ConfidenceLevel.LOW
    assert rec.verified_success_count == 0
    assert rec.failure_count == 1


# ---- 15. Execution failure not counting as recovery success ----

def test_execution_failure_not_counting_as_recovery_success():
    """Rule: A task that failed execution entirely increments failure_count."""
    memory = RecoveryMemory()
    memory.remember(_make_entry(
        entry_id="exec_fail",
        task_name="calc_task",
        root_cause="zero_division",
        recovery_action=RecoveryAction.STOP,
        recovered_execution=False,
        verification_status=VerificationStatus.FAILED,
        terminal_status=RecoveryStatus.FAILED_PERMANENT,
    ))

    rec = memory.recommend("calc_task", "zero_division")
    assert rec.suggested_action is None
    assert rec.confidence == ConfidenceLevel.LOW
    assert rec.verified_success_count == 0
    assert rec.failure_count == 1


# ---- 16. Conflicting outcomes suppress or downgrade recommendation ----

def test_conflicting_outcomes_suppress_or_downgrade_recommendation():
    """Verifies that failures suppress advice or downgrade confidence, and action ties suppress advice."""
    memory = RecoveryMemory()

    # Subcase A: 2 verified successes + 2 failures -> equal/outweighed -> suppressed
    memory.remember(_make_entry(entry_id="v1", task_name="t1", root_cause="rc", recovered_execution=True, verification_status=VerificationStatus.VERIFIED))
    memory.remember(_make_entry(entry_id="v2", task_name="t1", root_cause="rc", recovered_execution=True, verification_status=VerificationStatus.VERIFIED))
    memory.remember(_make_entry(entry_id="f1", task_name="t1", root_cause="rc", recovered_execution=False, verification_status=VerificationStatus.FAILED))
    memory.remember(_make_entry(entry_id="f2", task_name="t1", root_cause="rc", recovered_execution=False, verification_status=VerificationStatus.FAILED))

    rec = memory.recommend("t1", "rc")
    assert rec.suggested_action is None
    assert rec.confidence == ConfidenceLevel.LOW
    assert "Conflicting evidence" in rec.rationale

    # Subcase B: 2 verified successes + 1 failure -> verified outnumbers failures -> downgraded from HIGH to MEDIUM
    memory.clear()
    memory.remember(_make_entry(entry_id="v1", task_name="t2", root_cause="rc", recovered_execution=True, verification_status=VerificationStatus.VERIFIED))
    memory.remember(_make_entry(entry_id="v2", task_name="t2", root_cause="rc", recovered_execution=True, verification_status=VerificationStatus.VERIFIED))
    memory.remember(_make_entry(entry_id="f1", task_name="t2", root_cause="rc", recovered_execution=False, verification_status=VerificationStatus.FAILED))

    rec2 = memory.recommend("t2", "rc")
    assert rec2.suggested_action == RecoveryAction.RETRY
    assert rec2.confidence == ConfidenceLevel.MEDIUM
    assert "Moderate consensus" in rec2.rationale

    # Subcase C: Split consensus between actions (1 RETRY vs 1 FALLBACK) -> tie suppresses advice
    memory.clear()
    memory.remember(_make_entry(entry_id="v_retry", task_name="t3", root_cause="rc",
                                recovery_action=RecoveryAction.RETRY,
                                recovered_execution=True, verification_status=VerificationStatus.VERIFIED))
    memory.remember(_make_entry(entry_id="v_fallback", task_name="t3", root_cause="rc",
                                recovery_action=RecoveryAction.FALLBACK,
                                fallback_task_name="t3_backup",
                                terminal_status=RecoveryStatus.RECOVERED_VIA_FALLBACK,
                                recovered_execution=True, verification_status=VerificationStatus.VERIFIED))

    rec3 = memory.recommend("t3", "rc")
    assert rec3.suggested_action is None
    assert rec3.confidence == ConfidenceLevel.LOW
    assert "tied between multiple actions" in rec3.rationale


# ---- 17. Input validation for max_entries and malformed entries ----

def test_input_validation_max_entries_and_malformed_entries():
    """Verifies that invalid construction parameters and malformed entries raise appropriate exceptions."""
    with pytest.raises(ValueError, match="max_entries must be positive"):
        RecoveryMemory(max_entries=0)

    with pytest.raises(ValueError, match="max_entries must be positive"):
        RecoveryMemory(max_entries=-10)

    memory = RecoveryMemory()
    with pytest.raises(TypeError, match="entry must be a RecoveryMemoryEntry"):
        memory.remember({"entry_id": "bad"})  # type: ignore[arg-type]

    # Invariant: attempt_count < 1
    with pytest.raises(ValueError, match="attempt_count must be an integer >= 1"):
        _make_entry(attempt_count=0)

    # Invariant: recovered_execution=False with VERIFIED
    with pytest.raises(ValueError, match="Unrecovered execution cannot have verification_status VERIFIED"):
        _make_entry(recovered_execution=False, verification_status=VerificationStatus.VERIFIED)

    # Invariant: RECOVERED_VIA_FALLBACK requires fallback_task_name
    with pytest.raises(ValueError, match="RECOVERED_VIA_FALLBACK requires fallback_task_name"):
        _make_entry(
            terminal_status=RecoveryStatus.RECOVERED_VIA_FALLBACK,
            recovery_action=RecoveryAction.FALLBACK,
            fallback_task_name=None,
        )


# ---- 18. Sanitization and validation of task names and labels ----

def test_sanitization_validation_task_names_and_labels():
    """Verifies that whitespace-only or control-character-laden labels are rejected."""
    with pytest.raises(ValueError, match="task_name cannot be empty or whitespace-only"):
        _make_entry(task_name="   ")

    with pytest.raises(ValueError, match="root_cause cannot be empty or whitespace-only"):
        _make_entry(root_cause="")

    with pytest.raises(ValueError, match="task_name contains illegal control characters"):
        _make_entry(task_name="task\ninjection")

    with pytest.raises(ValueError, match="root_cause contains illegal control characters"):
        _make_entry(root_cause="cause\0null")

    memory = RecoveryMemory()
    with pytest.raises(ValueError, match="task_name cannot be empty"):
        memory.find_similar(" ", "valid_cause")

    with pytest.raises(ValueError, match="root_cause cannot be empty"):
        memory.recommend("valid_task", "")


# ---- 19. Internal storage not being externally mutable ----

def test_internal_storage_not_externally_mutable():
    """Verifies that entry instances are frozen and query returns do not expose mutable internals."""
    memory = RecoveryMemory()
    entry = _make_entry(entry_id="freeze_test")
    memory.remember(entry)

    # RecoveryMemoryEntry is a frozen dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.task_name = "mutated_task"  # type: ignore[misc]

    # Modifying the list returned by find_similar does not mutate internal storage
    results = memory.find_similar("payment_task", "network_timeout")
    assert len(results) == 1
    results.clear()

    assert len(memory) == 1
    assert len(memory.find_similar("payment_task", "network_timeout")) == 1


# ---- 20. Concurrent access thread safety ----

def test_concurrent_access_thread_safety():
    """Verifies thread safety under concurrent writes, reads, and clears."""
    memory = RecoveryMemory(max_entries=500)
    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                e = _make_entry(
                    entry_id=f"w{worker_id}_{i}",
                    task_name="shared_task",
                    root_cause="concurrent_load",
                    timestamp=float(i),
                )
                memory.remember(e)
                similar = memory.find_similar("shared_task", "concurrent_load", limit=5)
                assert isinstance(similar, list)
                rec = memory.recommend("shared_task", "concurrent_load")
                assert isinstance(rec, MemoryRecommendation)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(memory) > 0


# ---- 21. Memory cannot execute tasks or call task functions ----

def test_memory_cannot_execute_tasks():
    """Verifies that RecoveryMemory has no runner, executor, or task dispatch capabilities."""
    memory = RecoveryMemory()

    assert not hasattr(memory, "run")
    assert not hasattr(memory, "execute")
    assert not hasattr(memory, "runner")
    assert not hasattr(memory, "_runner")
    assert not hasattr(memory, "dispatch")


# ---- 22. Public imports work without circular imports ----

def test_public_imports_without_circular_imports():
    """Verifies that all memory public types are exported from shar without circular dependency issues."""
    import shar

    assert hasattr(shar, "RecoveryMemory")
    assert hasattr(shar, "RecoveryMemoryEntry")
    assert hasattr(shar, "MemoryRecommendation")

    assert shar.RecoveryMemory is RecoveryMemory
    assert shar.RecoveryMemoryEntry is RecoveryMemoryEntry
    assert shar.MemoryRecommendation is MemoryRecommendation

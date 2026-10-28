"""Controlled recovery memory for the Self-Healing AI Agent Runtime (shar).

This module provides a bounded, deterministic, in-memory store for past recovery
experiences. It captures safe, structured metadata about recovery attempts and their
verification outcomes, allowing downstream components to consult historical evidence
without bypassing safety boundaries or violating data privacy.

Design Principles:
- Advisory only: Recommendations provide historical evidence, never authorization.
- Zero sensitive data: No task arguments, kwargs, raw outputs, or tracebacks are stored.
- Explicit outcome separation: Verified success, unverified success, verification failure,
  and execution failure are tracked in disjoint counters.
- Conservative confidence: A single verified success cannot produce HIGH confidence;
  unverified recoveries cannot produce HIGH confidence; conflicting evidence suppresses advice.
- Deterministic matching: Primary task identity and normalized failure signatures form
  the matching boundary. No fuzzy matching or external dependencies.
- Bounded capacity: Enforces a strict max_entries limit with FIFO eviction.
- Thread safe: All mutations and queries are protected by an internal lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Optional

from shar.logger import setup_logger
from shar.models import (
    ConfidenceLevel,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryStatus,
    VerificationStatus,
)


def _sanitize_label(label: str, field_name: str) -> str:
    """Validates and normalizes string labels (task names, diagnostic causes)."""
    if not isinstance(label, str):
        raise ValueError(f"{field_name} must be a str, got {type(label).__name__}")
    cleaned = label.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    if any(c in cleaned for c in ("\n", "\r", "\t", "\0")):
        raise ValueError(f"{field_name} contains illegal control characters")
    return cleaned


@dataclass(frozen=True)
class RecoveryMemoryEntry:
    """Safe, structured record of a prior recovery workflow and its verified outcome.

    Invariants:
    - attempt_count is 1-based (total attempts executed, >= 1).
    - task_name and root_cause must be non-empty, sanitized strings.
    - If recovered_execution is False, verification_status must NOT be VERIFIED.
    - If terminal_status is RECOVERED_VIA_FALLBACK, fallback_task_name must be present.
    - Stores NO task arguments, raw outputs, credentials, or full tracebacks.
    """

    entry_id: str
    task_name: str
    failure_type: FailureType
    failure_category: FailureCategory
    root_cause: str
    recovery_action: RecoveryAction
    recovered_execution: bool
    verification_status: VerificationStatus
    attempt_count: int
    terminal_status: RecoveryStatus
    exception_type: Optional[str] = None
    fallback_task_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _sanitize_label(self.entry_id, "entry_id")
        _sanitize_label(self.task_name, "task_name")
        _sanitize_label(self.root_cause, "root_cause")

        if self.exception_type is not None:
            _sanitize_label(self.exception_type, "exception_type")

        if self.fallback_task_name is not None:
            _sanitize_label(self.fallback_task_name, "fallback_task_name")

        if not isinstance(self.attempt_count, int) or self.attempt_count < 1:
            raise ValueError(f"attempt_count must be an integer >= 1, got {self.attempt_count}")

        if not isinstance(self.failure_type, FailureType):
            raise ValueError(f"failure_type must be a FailureType enum, got {self.failure_type}")

        if not isinstance(self.failure_category, FailureCategory):
            raise ValueError(f"failure_category must be a FailureCategory enum, got {self.failure_category}")

        if not isinstance(self.recovery_action, RecoveryAction):
            raise ValueError(f"recovery_action must be a RecoveryAction enum, got {self.recovery_action}")

        if not isinstance(self.verification_status, VerificationStatus):
            raise ValueError(f"verification_status must be a VerificationStatus enum, got {self.verification_status}")

        if not isinstance(self.terminal_status, RecoveryStatus):
            raise ValueError(f"terminal_status must be a RecoveryStatus enum, got {self.terminal_status}")

        # Safety Invariant: Execution failure can never produce verified success
        if not self.recovered_execution and self.verification_status == VerificationStatus.VERIFIED:
            raise ValueError("Unrecovered execution cannot have verification_status VERIFIED.")

        # Safety Invariant: Fallback terminal recovery requires explicit fallback task attribution
        if self.terminal_status == RecoveryStatus.RECOVERED_VIA_FALLBACK and not self.fallback_task_name:
            raise ValueError("RecoveryStatus.RECOVERED_VIA_FALLBACK requires fallback_task_name.")


@dataclass(frozen=True)
class MemoryRecommendation:
    """Structured, advisory-only recommendation synthesized from historical memory.

    A recommendation is historical evidence and statistical observation,
    NEVER an authorization to execute an action or bypass safety policies.
    """

    suggested_action: Optional[RecoveryAction]
    confidence: ConfidenceLevel
    sample_size: int
    verified_success_count: int
    unverified_success_count: int
    failure_count: int
    rationale: str


class RecoveryMemory:
    """Thread-safe, bounded, in-memory store for recovery records and recommendations."""

    def __init__(
        self,
        max_entries: int = 1000,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the RecoveryMemory store.

        Args:
            max_entries: Maximum number of entries to retain. Must be positive.
            logger: Custom logger instance. Defaults to 'shar.memory'.
        """
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max_entries = max_entries
        self._logger = logger or setup_logger("shar.memory")
        self._lock = threading.RLock()
        self._entries: list[RecoveryMemoryEntry] = []

    def remember(self, entry: RecoveryMemoryEntry) -> None:
        """Stores a recovery memory entry, evicting the oldest entry if at capacity.

        Args:
            entry: A validated RecoveryMemoryEntry.
        """
        if not isinstance(entry, RecoveryMemoryEntry):
            raise TypeError(f"entry must be a RecoveryMemoryEntry, got {type(entry).__name__}")

        with self._lock:
            if len(self._entries) >= self._max_entries:
                evicted = self._entries.pop(0)
                self._logger.debug(
                    f"RecoveryMemory capacity ({self._max_entries}) reached; evicted entry '{evicted.entry_id}'"
                )

            self._entries.append(entry)
            self._logger.info(
                f"Recorded recovery memory entry '{entry.entry_id}' for task '{entry.task_name}' "
                f"(action={entry.recovery_action.value}, status={entry.terminal_status.value}, "
                f"verification={entry.verification_status.value})",
                extra={
                    "entry_id": entry.entry_id,
                    "task_name": entry.task_name,
                    "action": entry.recovery_action.value,
                    "terminal_status": entry.terminal_status.value,
                    "verification_status": entry.verification_status.value,
                },
            )

    def find_similar(
        self,
        task_name: str,
        root_cause: str,
        exception_type: Optional[str] = None,
        limit: int = 5,
    ) -> list[RecoveryMemoryEntry]:
        """Finds matching historical entries sorted deterministically.

        Matching Rules:
        - Must match task_name and root_cause.
        - If exception_type is specified, entries with identical exception_type
          or unspecified exception_type are matched, with exact matches prioritized.
        - Returns a new list containing up to `limit` entries. Internal storage is not exposed.

        Args:
            task_name: Primary task identity to match.
            root_cause: Diagnostic root cause to match.
            exception_type: Optional normalized exception class name.
            limit: Maximum entries to return. Must be positive.

        Returns:
            List of matching RecoveryMemoryEntry objects.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        clean_task_name = _sanitize_label(task_name, "task_name")
        clean_root_cause = _sanitize_label(root_cause, "root_cause")
        clean_exc = _sanitize_label(exception_type, "exception_type") if exception_type is not None else None

        with self._lock:
            matching: list[RecoveryMemoryEntry] = []
            for entry in self._entries:
                if entry.task_name == clean_task_name and entry.root_cause == clean_root_cause:
                    if clean_exc is not None:
                        if entry.exception_type == clean_exc or entry.exception_type is None:
                            matching.append(entry)
                    else:
                        matching.append(entry)

            def sort_key(e: RecoveryMemoryEntry) -> tuple[int, int, float, str]:
                # Priority 1: Exact exception type match
                exact_exc = 0 if (clean_exc is not None and e.exception_type == clean_exc) else 1
                # Priority 2: Verified success first
                if e.recovered_execution and e.verification_status == VerificationStatus.VERIFIED:
                    ver_score = 0
                elif e.recovered_execution and e.verification_status == VerificationStatus.UNVERIFIED:
                    ver_score = 1
                else:
                    ver_score = 2
                # Priority 3: Newer timestamp (negative timestamp)
                # Priority 4: Stable entry_id tiebreaker
                return (exact_exc, ver_score, -e.timestamp, e.entry_id)

            matching.sort(key=sort_key)
            return list(matching[:limit])

    def recommend(
        self,
        task_name: str,
        root_cause: str,
        exception_type: Optional[str] = None,
    ) -> MemoryRecommendation:
        """Synthesizes an advisory-only recommendation based on historical recovery outcomes.

        Deterministic Consensus & Confidence Rules:
        - sample_size == 0 -> suggested_action = None, confidence = LOW.
        - verified_count == 0, unverified_count > 0, failure_count == 0 ->
          suggested_action = top unverified action, confidence = LOW (unverified cannot confer HIGH or MEDIUM).
        - verified_count == 0, failure_count > 0 ->
          suggested_action = None, confidence = LOW.
        - failure_count >= verified_count ->
          suggested_action = None, confidence = LOW (conflicting failures suppress recommendation).
        - Tie between verified recovery actions ->
          suggested_action = None, confidence = LOW (split consensus).
        - verified_count == 1, failure_count == 0 ->
          suggested_action = action, confidence = MEDIUM (single verified success cannot produce HIGH).
        - verified_count >= 2, failure_count == 0, 100% action agreement ->
          suggested_action = action, confidence = HIGH.
        - verified_count >= 2, failure_count > 0, top_action outnumbers failures ->
          suggested_action = action, confidence = MEDIUM (downgraded due to historical failures).

        Args:
            task_name: Primary task identity to match.
            root_cause: Diagnostic root cause to match.
            exception_type: Optional normalized exception class name.

        Returns:
            A MemoryRecommendation with suggested action, confidence, and outcome statistics.
        """
        clean_task_name = _sanitize_label(task_name, "task_name")
        clean_root_cause = _sanitize_label(root_cause, "root_cause")
        clean_exc = _sanitize_label(exception_type, "exception_type") if exception_type is not None else None

        with self._lock:
            matching = [
                e for e in self._entries
                if e.task_name == clean_task_name and e.root_cause == clean_root_cause
                and (clean_exc is None or e.exception_type == clean_exc or e.exception_type is None)
            ]

        sample_size = len(matching)
        if sample_size == 0:
            return MemoryRecommendation(
                suggested_action=None,
                confidence=ConfidenceLevel.LOW,
                sample_size=0,
                verified_success_count=0,
                unverified_success_count=0,
                failure_count=0,
                rationale=(
                    f"No historical recovery entries found for task '{clean_task_name}' "
                    f"with root cause '{clean_root_cause}'."
                ),
            )

        # 4 distinct outcome buckets
        verified_successes = [
            e for e in matching
            if e.recovered_execution and e.verification_status == VerificationStatus.VERIFIED
        ]
        unverified_successes = [
            e for e in matching
            if e.recovered_execution and e.verification_status == VerificationStatus.UNVERIFIED
        ]
        rejected_or_inconclusive = [
            e for e in matching
            if e.recovered_execution and e.verification_status in (VerificationStatus.FAILED, VerificationStatus.INCONCLUSIVE)
        ]
        execution_failures = [
            e for e in matching
            if not e.recovered_execution
        ]

        verified_count = len(verified_successes)
        unverified_count = len(unverified_successes)
        failure_count = len(rejected_or_inconclusive) + len(execution_failures)

        # Case 1: Zero verified successes
        if verified_count == 0:
            if unverified_count > 0 and failure_count == 0:
                action_counts: dict[RecoveryAction, int] = {}
                for e in unverified_successes:
                    action_counts[e.recovery_action] = action_counts.get(e.recovery_action, 0) + 1
                sorted_actions = sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0].value))
                top_action, _ = sorted_actions[0]
                return MemoryRecommendation(
                    suggested_action=top_action,
                    confidence=ConfidenceLevel.LOW,
                    sample_size=sample_size,
                    verified_success_count=0,
                    unverified_success_count=unverified_count,
                    failure_count=0,
                    rationale=(
                        f"Observed {unverified_count} unverified recovery attempt(s) via {top_action.value}. "
                        "Confidence is LOW because output correctness was not semantically verified."
                    ),
                )
            else:
                return MemoryRecommendation(
                    suggested_action=None,
                    confidence=ConfidenceLevel.LOW,
                    sample_size=sample_size,
                    verified_success_count=0,
                    unverified_success_count=unverified_count,
                    failure_count=failure_count,
                    rationale=(
                        f"All {sample_size} historical attempt(s) failed execution or failed verification. "
                        "No recovery action recommended."
                    ),
                )

        # Case 2: verified_count > 0
        action_counts = {}
        for e in verified_successes:
            action_counts[e.recovery_action] = action_counts.get(e.recovery_action, 0) + 1
        sorted_actions = sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0].value))
        top_action, top_count = sorted_actions[0]

        # Check for tie between verified actions
        if len(sorted_actions) > 1 and sorted_actions[1][1] == top_count:
            return MemoryRecommendation(
                suggested_action=None,
                confidence=ConfidenceLevel.LOW,
                sample_size=sample_size,
                verified_success_count=verified_count,
                unverified_success_count=unverified_count,
                failure_count=failure_count,
                rationale=(
                    f"Conflicting evidence: historical verified successes are tied between multiple actions "
                    f"({top_action.value} vs {sorted_actions[1][0].value}). Action suppressed."
                ),
            )

        # Check if failures equal or outnumber verified successes
        if failure_count >= verified_count:
            return MemoryRecommendation(
                suggested_action=None,
                confidence=ConfidenceLevel.LOW,
                sample_size=sample_size,
                verified_success_count=verified_count,
                unverified_success_count=unverified_count,
                failure_count=failure_count,
                rationale=(
                    f"Conflicting evidence: {failure_count} failure(s) outweigh or equal {verified_count} "
                    f"verified success(es). Action suppressed."
                ),
            )

        # Case 2A: Single verified success cannot produce HIGH confidence
        if verified_count == 1:
            return MemoryRecommendation(
                suggested_action=top_action,
                confidence=ConfidenceLevel.MEDIUM,
                sample_size=sample_size,
                verified_success_count=1,
                unverified_success_count=unverified_count,
                failure_count=failure_count,
                rationale=(
                    f"Single verified recovery via {top_action.value} observed with {failure_count} failure(s) "
                    "(confidence MEDIUM: minimum 2 verified successes required for HIGH)."
                ),
            )

        # Case 2B: verified_count >= 2
        if failure_count == 0 and top_count == verified_count:
            return MemoryRecommendation(
                suggested_action=top_action,
                confidence=ConfidenceLevel.HIGH,
                sample_size=sample_size,
                verified_success_count=verified_count,
                unverified_success_count=unverified_count,
                failure_count=0,
                rationale=(
                    f"Strong consensus: {verified_count} verified recoveries via {top_action.value} with zero failures "
                    "(confidence HIGH)."
                ),
            )
        else:
            return MemoryRecommendation(
                suggested_action=top_action,
                confidence=ConfidenceLevel.MEDIUM,
                sample_size=sample_size,
                verified_success_count=verified_count,
                unverified_success_count=unverified_count,
                failure_count=failure_count,
                rationale=(
                    f"Moderate consensus: {top_count} verified recoveries via {top_action.value} outnumber "
                    f"{failure_count} failure(s) (confidence MEDIUM)."
                ),
            )

    def clear(self) -> None:
        """Empties all entries from the store."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Returns the current number of stored entries."""
        with self._lock:
            return len(self._entries)

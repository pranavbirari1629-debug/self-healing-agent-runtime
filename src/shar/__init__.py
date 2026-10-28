"""Self-Healing AI Agent Runtime (shar).

A lightweight, reliable runtime providing task isolation, failure detection,
and recovery management.
"""

from shar.logger import setup_logger
from shar.models import (
    AttemptRecord,
    ConfidenceLevel,
    Diagnosis,
    DiagnosticContext,
    FailureCategory,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryResult,
    RecoveryStatus,
    Task,
    TaskOutcome,
    TaskResult,
    VerificationResult,
    VerificationStatus,
)
from shar.detector import ClassificationResult, FailureDetector
from shar.diagnosis import DiagnosisEngine
from shar.memory import MemoryRecommendation, RecoveryMemory, RecoveryMemoryEntry
from shar.orchestrator import RecoveryOrchestrator
from shar.policy import RecoveryPolicy
from shar.agent import (
    Agent,
    AgentAction,
    AgentActionType,
    AgentPlanner,
    AgentResult,
    AgentStatus,
    AgentStep,
    ScriptedPlanner,
)
from shar.runner import TaskRunner
from shar.tools import (
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolOrchestratorBridge,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
    UNSET,
)
from shar.verifier import OutputVerifier

__version__ = "0.1.0"

__all__ = [
    # Phase 1
    "Task",
    "TaskResult",
    "TaskOutcome",
    "FailureType",
    "TaskRunner",
    "setup_logger",
    # Phase 2
    "FailureCategory",
    "ConfidenceLevel",
    "Diagnosis",
    "DiagnosticContext",
    "ClassificationResult",
    "FailureDetector",
    "DiagnosisEngine",
    # Phase 3
    "RecoveryAction",
    "RecoveryStatus",
    "RecoveryDecision",
    "AttemptRecord",
    "RecoveryResult",
    "RecoveryPolicy",
    "RecoveryOrchestrator",
    # Phase 4
    "VerificationStatus",
    "VerificationResult",
    "OutputVerifier",
    "RecoveryMemoryEntry",
    "MemoryRecommendation",
    "RecoveryMemory",
    # Phase 6, Step 1
    "ToolParameter",
    "ToolSpec",
    "ToolRegistry",
    "ToolExecutionStatus",
    "ToolExecutionResult",
    "ToolOrchestratorBridge",
    "UNSET",
    # Phase 6, Step 2
    "AgentActionType",
    "AgentStatus",
    "AgentAction",
    "AgentStep",
    "AgentResult",
    "AgentPlanner",
    "ScriptedPlanner",
    "Agent",
    "__version__",
]

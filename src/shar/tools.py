"""Deterministic, permission-aware tool integration layer for the shar runtime.

This module provides the core abstractions for defining, registering, validating,
and executing tools within the Self-Healing Agent Runtime:
- ToolParameter: Immutable typed parameter specification with strict validation.
- ToolSpec: Immutable specification of a tool callable and its recovery constraints.
- ToolRegistry: In-memory registry with deterministic listing and permission filtering.
- ToolExecutionStatus & ToolExecutionResult: Structured result envelopes.
- ToolOrchestratorBridge: Coordinates permission gating, argument validation, Task
  synthesis, and execution through RecoveryOrchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable, Optional, Sequence
import uuid

from shar.logger import setup_logger
from shar.models import (
    FailureType,
    RecoveryResult,
    RecoveryStatus,
    Task,
    VerificationStatus,
)
from shar.orchestrator import RecoveryOrchestrator


class _UnsetType:
    """Sentinel indicating that an optional parameter default was not provided."""

    _instance: Optional[_UnsetType] = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"

    def __bool__(self) -> bool:
        return False


UNSET = _UnsetType()


def _validate_type(value: Any, expected_type: type) -> bool:
    """Validates that value conforms to expected_type without unsafe conversions.

    Explicitly guards against Python's bool-is-a-subclass-of-int behavior:
    `isinstance(True, int)` is True in Python, which would otherwise allow boolean
    flags to masquerade as numeric counts, IDs, or timestamps.
    """
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is float:
        return isinstance(value, (float, int)) and not isinstance(value, bool)
    return isinstance(value, expected_type)


@dataclass(frozen=True)
class ToolParameter:
    """Defines an individual named parameter expected by a Tool.

    Invariants:
    - name must be a valid, non-empty Python identifier.
    - type must be a valid Python type (eval/exec is strictly forbidden).
    - If required is True, default must be UNSET.
    - If required is False, default must be specified (can be explicitly None).
    """

    name: str
    type: type
    description: str = ""
    required: bool = True
    default: Any = UNSET

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolParameter name must be a non-empty string")
        if not self.name.isidentifier():
            raise ValueError(
                f"ToolParameter name '{self.name}' must be a valid Python identifier"
            )
        if not isinstance(self.type, type):
            raise TypeError(
                f"ToolParameter 'type' must be a Python type, got {type(self.type).__name__}"
            )
        if not isinstance(self.description, str):
            raise TypeError("ToolParameter description must be a string")
        if not isinstance(self.required, bool):
            raise TypeError("ToolParameter required must be a boolean")

        if self.required:
            if self.default is not UNSET:
                raise ValueError(
                    f"Required parameter '{self.name}' cannot specify a default value"
                )
        else:
            if self.default is UNSET:
                raise ValueError(
                    f"Optional parameter '{self.name}' must specify a default value "
                    "(use default=None if explicitly nullable)"
                )
            if self.default is not None and not _validate_type(self.default, self.type):
                raise TypeError(
                    f"Default value {self.default!r} for parameter '{self.name}' "
                    f"does not match expected type {self.type.__name__}"
                )


@dataclass(frozen=True)
class ToolSpec:
    """Immutable specification of a tool callable and its operational constraints.

    Safety Guarantees:
    - Conservative defaults: is_idempotent=False, is_read_only=False, safe_to_retry_on_timeout=False.
    - Declaring is_read_only=True enforces is_idempotent=True, but does NOT grant implicit
      permission bypasses or sensitive-data exemptions.
    - Validates numeric boundaries matching core Task validation.
    """

    name: str
    description: str
    callable: Callable[..., Any]
    parameters: tuple[ToolParameter, ...] = field(default_factory=tuple)
    required_permissions: frozenset[str] = field(default_factory=frozenset)
    is_idempotent: bool = False
    is_read_only: bool = False
    safe_to_retry_on_timeout: bool = False
    timeout_seconds: Optional[float] = None
    max_retries: int = 1
    recovery_budget_seconds: Optional[float] = None
    backoff_base_delay: float = 0.05
    backoff_factor: float = 2.0
    backoff_max_delay: float = 2.0
    jitter: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    verifier: Optional[Callable[[Any], bool]] = None
    fallback_tool_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolSpec name must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("ToolSpec description must be a string")
        if not callable(self.callable):
            raise TypeError("ToolSpec callable must be a callable object")

        # Normalize parameters sequence to an immutable tuple
        if isinstance(self.parameters, (list, tuple)):
            norm_params: list[ToolParameter] = []
            for p in self.parameters:
                if not isinstance(p, ToolParameter):
                    raise TypeError(f"Expected ToolParameter instance, got {type(p).__name__}")
                norm_params.append(p)
            object.__setattr__(self, "parameters", tuple(norm_params))
        else:
            raise TypeError("ToolSpec parameters must be a sequence of ToolParameter")

        # Verify parameter uniqueness
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            duplicates = [name for name in param_names if param_names.count(name) > 1]
            raise ValueError(f"Duplicate parameter names in ToolSpec: {sorted(set(duplicates))}")

        # Normalize required_permissions to frozenset
        if isinstance(self.required_permissions, (set, frozenset, list, tuple)):
            for perm in self.required_permissions:
                if not isinstance(perm, str) or not perm.strip():
                    raise ValueError("Permissions must be non-empty strings")
            object.__setattr__(self, "required_permissions", frozenset(self.required_permissions))
        else:
            raise TypeError("ToolSpec required_permissions must be a set or sequence of strings")

        # Numeric safety invariants matching Task.__post_init__
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

        # Semantic invariants
        if self.is_read_only and not self.is_idempotent:
            raise ValueError("Read-only tools must declare is_idempotent=True")

        if self.fallback_tool_name is not None:
            if not isinstance(self.fallback_tool_name, str) or not self.fallback_tool_name.strip():
                raise ValueError("fallback_tool_name must be a non-empty string if provided")
            if self.fallback_tool_name == self.name:
                raise ValueError(
                    f"ToolSpec '{self.name}' cannot designate itself as its own fallback"
                )


class ToolRegistry:
    """In-memory catalog of authorized ToolSpec instances.

    Guarantees:
    - Deterministic duplicate rejection on registration.
    - Prevents accidental mutation of registered specifications.
    - Predictable None return for unregistered tool names.
    - Permission-filtered tool listing sorted deterministically by name.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._logger = logger or setup_logger("shar.tools.registry")

    def register(self, tool: ToolSpec) -> None:
        """Registers a ToolSpec in the registry.

        Raises:
            TypeError: If tool is not a ToolSpec instance.
            ValueError: If a tool with the same name is already registered.
        """
        if not isinstance(tool, ToolSpec):
            raise TypeError(f"Expected ToolSpec instance, got {type(tool).__name__}")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool
        self._logger.info(
            f"Registered tool '{tool.name}' (idempotent={tool.is_idempotent}, "
            f"permissions={sorted(tool.required_permissions)})",
            extra={"tool_name": tool.name},
        )

    def get(self, name: str) -> Optional[ToolSpec]:
        """Looks up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(
        self, caller_permissions: Optional[frozenset[str] | set[str]] = None
    ) -> list[ToolSpec]:
        """Lists registered tools, optionally filtered by caller permissions.

        Returns:
            List of ToolSpec instances sorted alphabetically by name.
        """
        if caller_permissions is None:
            return sorted(self._tools.values(), key=lambda t: t.name)

        perms = frozenset(caller_permissions)
        matching = [
            tool for tool in self._tools.values()
            if tool.required_permissions.issubset(perms)
        ]
        return sorted(matching, key=lambda t: t.name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


class ToolExecutionStatus(str, Enum):
    """Terminal outcome of a tool invocation request."""

    SUCCESS = "success"
    PERMISSION_DENIED = "permission_denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_NOT_FOUND = "tool_not_found"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured envelope capturing the outcome of a tool execution.

    Invariants:
    - is_verified is True iff verification_status is VERIFIED.
    - If status != SUCCESS, is_success must be False.
    - If is_success is False, is_verified must be False.
    - Preserves correlation ID, timing, and full RecoveryResult when available.
    """

    tool_name: str
    status: ToolExecutionStatus
    output: Any = None
    error: Optional[str] = None
    is_success: bool = False
    is_verified: bool = False
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    recovery_result: Optional[RecoveryResult] = None
    correlation_id: str = ""
    execution_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, ToolExecutionStatus):
            raise TypeError(f"status must be a ToolExecutionStatus, got {type(self.status).__name__}")
        if not isinstance(self.verification_status, VerificationStatus):
            raise TypeError(
                f"verification_status must be a VerificationStatus, got {type(self.verification_status).__name__}"
            )
        if self.execution_time_ms < 0:
            raise ValueError(f"execution_time_ms must be non-negative, got {self.execution_time_ms}")

        # Invariant: Non-success cannot have is_success=True
        if self.status != ToolExecutionStatus.SUCCESS and self.is_success:
            raise ValueError(f"Status '{self.status.value}' cannot have is_success=True")

        # Invariant: Unsuccessful execution cannot be verified
        if not self.is_success and self.is_verified:
            raise ValueError("Unsuccessful tool execution cannot have is_verified=True")

        # Invariant: is_verified iff status == VERIFIED
        expected_verified = (self.verification_status == VerificationStatus.VERIFIED)
        if self.is_verified != expected_verified:
            raise ValueError(
                f"is_verified={self.is_verified} must be True if and only if "
                f"verification_status is VerificationStatus.VERIFIED, got {self.verification_status.value}"
            )


class ToolOrchestratorBridge:
    """Bridges tool invocations to RecoveryOrchestrator with safety checks.

    Responsibilities:
    - Pre-execution permission verification for primary and fallback tools.
    - Argument validation: checks parameter presence, defaults, and strict types.
    - Guaranteed zero-invocation on rejected requests (permissions/arguments).
    - Translates ToolSpec to Task without modifying existing runtime code.
    - Maps RecoveryResult and VerificationResult into ToolExecutionResult.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        orchestrator: Optional[RecoveryOrchestrator] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError(f"Expected ToolRegistry, got {type(registry).__name__}")
        self._registry = registry
        self._orchestrator = orchestrator or RecoveryOrchestrator()
        self._logger = logger or setup_logger("shar.tools.bridge")

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        caller_permissions: frozenset[str] | set[str] = frozenset(),
        correlation_id: Optional[str] = None,
    ) -> ToolExecutionResult:
        """Validates and executes a tool through the self-healing orchestrator.

        Args:
            tool_name: The name of the registered tool to execute.
            arguments: Dictionary of arguments matching the tool's parameter specification.
            caller_permissions: Trusted application-provided permission set.
            correlation_id: Optional tracking identifier. Generated if omitted.

        Returns:
            A structured ToolExecutionResult detailing status, output, verification,
            and underlying recovery diagnostics.
        """
        cid = correlation_id or uuid.uuid4().hex
        start_time = time.perf_counter()
        perms = frozenset(caller_permissions)

        # 1. Lookup tool in registry
        tool = self._registry.get(tool_name)
        if tool is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"Tool '{tool_name}' not found in registry"
            self._logger.warning(
                err_msg,
                extra={"correlation_id": cid, "tool_name": tool_name},
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.TOOL_NOT_FOUND,
                error=err_msg,
                correlation_id=cid,
                execution_time_ms=elapsed_ms,
            )

        # 2. Enforce primary tool permissions immediately before execution
        if not tool.required_permissions.issubset(perms):
            missing = sorted(tool.required_permissions - perms)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = (
                f"Caller lacks required permission(s) for tool '{tool_name}': {missing}"
            )
            self._logger.warning(
                err_msg,
                extra={"correlation_id": cid, "tool_name": tool_name},
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.PERMISSION_DENIED,
                error=err_msg,
                correlation_id=cid,
                execution_time_ms=elapsed_ms,
            )

        # 3. Validate input arguments
        val_error, validated_kwargs = self._validate_arguments(tool, arguments)
        if val_error is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._logger.warning(
                val_error,
                extra={"correlation_id": cid, "tool_name": tool_name},
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.INVALID_ARGUMENTS,
                error=val_error,
                correlation_id=cid,
                execution_time_ms=elapsed_ms,
            )

        # 4. Resolve fallback tool and enforce fallback security boundaries
        fallback_task: Optional[Task] = None
        if tool.fallback_tool_name is not None:
            fb_tool = self._registry.get(tool.fallback_tool_name)
            if fb_tool is None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = (
                    f"Configured fallback tool '{tool.fallback_tool_name}' "
                    "not found in registry"
                )
                self._logger.warning(
                    err_msg,
                    extra={"correlation_id": cid, "tool_name": tool_name},
                )
                return ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.INVALID_ARGUMENTS,
                    error=err_msg,
                    correlation_id=cid,
                    execution_time_ms=elapsed_ms,
                )

            # Prevent fallback cycles
            if fb_tool.name == tool.name:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = f"Fallback tool '{fb_tool.name}' creates a self-cycle"
                return ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.INVALID_ARGUMENTS,
                    error=err_msg,
                    correlation_id=cid,
                    execution_time_ms=elapsed_ms,
                )

            # Independent permission check for fallback tool (prevents privilege escalation)
            if not fb_tool.required_permissions.issubset(perms):
                missing_fb = sorted(fb_tool.required_permissions - perms)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = (
                    f"Caller lacks required permission(s) for fallback tool "
                    f"'{fb_tool.name}': {missing_fb}"
                )
                self._logger.warning(
                    err_msg,
                    extra={"correlation_id": cid, "tool_name": tool_name},
                )
                return ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.PERMISSION_DENIED,
                    error=err_msg,
                    correlation_id=cid,
                    execution_time_ms=elapsed_ms,
                )

            # Validate fallback argument compatibility
            fb_err, fb_kwargs = self._validate_arguments(fb_tool, arguments)
            if fb_err is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = (
                    f"Incompatible arguments for fallback tool '{fb_tool.name}': {fb_err}"
                )
                self._logger.warning(
                    err_msg,
                    extra={"correlation_id": cid, "tool_name": tool_name},
                )
                return ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.INVALID_ARGUMENTS,
                    error=err_msg,
                    correlation_id=cid,
                    execution_time_ms=elapsed_ms,
                )

            # Construct fallback Task (fallback=None strictly bounds delegation depth to 1)
            fallback_task = self._build_task(fb_tool, fb_kwargs, fallback=None)

        # 5. Build primary Task and execute via RecoveryOrchestrator
        primary_task = self._build_task(tool, validated_kwargs, fallback=fallback_task)
        recovery_result = self._orchestrator.execute(primary_task, correlation_id=cid)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 6. Map RecoveryResult to ToolExecutionResult
        return self._map_recovery_result(tool.name, recovery_result, cid, elapsed_ms)

    def _validate_arguments(
        self, tool: ToolSpec, arguments: dict[str, Any]
    ) -> tuple[Optional[str], dict[str, Any]]:
        """Validates arguments against a tool's parameter specification.

        Returns:
            Tuple of (error_message, validated_kwargs). error_message is None if valid.
        """
        if not isinstance(arguments, dict):
            return "Arguments must be provided as a dictionary", {}

        param_map = {p.name: p for p in tool.parameters}

        # Check for unknown parameters
        unknown_keys = set(arguments.keys()) - set(param_map.keys())
        if unknown_keys:
            return (
                f"Unknown parameter(s) for tool '{tool.name}': {sorted(unknown_keys)}",
                {},
            )

        # Check for missing required parameters
        missing_required = [
            p.name for p in tool.parameters
            if p.required and p.name not in arguments
        ]
        if missing_required:
            return (
                f"Missing required parameter(s) for tool '{tool.name}': {sorted(missing_required)}",
                {},
            )

        # Validate types and populate defaults
        validated: dict[str, Any] = {}
        for param in tool.parameters:
            if param.name in arguments:
                val = arguments[param.name]
                if not _validate_type(val, param.type):
                    return (
                        f"Invalid type for parameter '{param.name}' in tool '{tool.name}': "
                        f"expected {param.type.__name__}, got {type(val).__name__}",
                        {},
                    )
                validated[param.name] = val
            else:
                # Optional parameter: populate configured default
                validated[param.name] = param.default

        return None, validated

    def _build_task(
        self,
        tool: ToolSpec,
        kwargs: dict[str, Any],
        fallback: Optional[Task],
    ) -> Task:
        """Synthesizes an existing core Task from a ToolSpec and validated arguments."""
        return Task(
            name=tool.name,
            callable=tool.callable,
            args=(),
            kwargs=kwargs,
            timeout_seconds=tool.timeout_seconds,
            max_retries=tool.max_retries,
            validator=tool.validator,
            fallback=fallback,
            metadata={"tool_name": tool.name, "is_read_only": tool.is_read_only},
            is_idempotent=tool.is_idempotent,
            safe_to_retry_on_timeout=tool.safe_to_retry_on_timeout,
            recovery_budget_seconds=tool.recovery_budget_seconds,
            backoff_base_delay=tool.backoff_base_delay,
            backoff_factor=tool.backoff_factor,
            backoff_max_delay=tool.backoff_max_delay,
            jitter=tool.jitter,
            verifier=tool.verifier,
        )

    def _map_recovery_result(
        self,
        tool_name: str,
        recovery_result: RecoveryResult,
        correlation_id: str,
        elapsed_ms: float,
    ) -> ToolExecutionResult:
        """Converts an orchestrated RecoveryResult into a structured ToolExecutionResult."""
        final_task_result = recovery_result.final_result
        verification = recovery_result.verification
        ver_status = (
            verification.status
            if verification is not None
            else VerificationStatus.UNVERIFIED
        )

        if recovery_result.is_success:
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.SUCCESS,
                output=final_task_result.output,
                error=None,
                is_success=True,
                is_verified=(ver_status == VerificationStatus.VERIFIED),
                verification_status=ver_status,
                recovery_result=recovery_result,
                correlation_id=correlation_id,
                execution_time_ms=elapsed_ms,
            )

        # Execution failed: map failure type without exposing tracebacks or secrets
        if (
            final_task_result.failure_type == FailureType.TIMEOUT
            or recovery_result.status == RecoveryStatus.FAILED_TIMEOUT_UNSAFE
        ):
            status = ToolExecutionStatus.TIMEOUT
            err_msg = f"Tool '{tool_name}' timed out during execution"
        else:
            status = ToolExecutionStatus.EXECUTION_FAILED
            exc_type = final_task_result.exception_type or "ExecutionError"
            exc_msg = final_task_result.exception_message or "Execution failed"
            err_msg = f"Tool '{tool_name}' failed with {exc_type}: {exc_msg}"

        return ToolExecutionResult(
            tool_name=tool_name,
            status=status,
            output=None,
            error=err_msg,
            is_success=False,
            is_verified=False,
            verification_status=ver_status,
            recovery_result=recovery_result,
            correlation_id=correlation_id,
            execution_time_ms=elapsed_ms,
        )

"""Deterministic, bounded agent coordination layer for the shar runtime.

This module provides the core abstractions for orchestrating multi-step goal execution
using registered tools through the self-healing ToolOrchestratorBridge:
- AgentActionType: Discriminator for planner decisions (TOOL_CALL, FINISH, FAIL).
- AgentStatus: Terminal outcome status for an agent execution.
- AgentAction: Immutable action descriptor emitted by a planner.
- AgentStep: Immutable snapshot of deliberation and action outcome.
- AgentResult: Comprehensive envelope capturing goal completion and history.
- AgentPlanner: Protocol defining the planner interface.
- ScriptedPlanner: Deterministic planner for testing and fixed-sequence workflows.
- Agent: Bounded coordinator managing goal execution, limits, and safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Callable, Optional, Protocol, Sequence
import uuid

from shar.logger import setup_logger
from shar.tools import ToolExecutionResult, ToolOrchestratorBridge, ToolSpec


class AgentActionType(str, Enum):
    """Discriminator indicating the nature of a planner's step decision."""

    TOOL_CALL = "tool_call"
    FINISH = "finish"
    FAIL = "fail"


class AgentStatus(str, Enum):
    """Terminal outcome classifications for an agent execution."""

    COMPLETED = "completed"
    FAILED = "failed"
    STEP_LIMIT_EXCEEDED = "step_limit_exceeded"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    PLANNING_ERROR = "planning_error"


@dataclass(frozen=True)
class AgentAction:
    """Action decision emitted by an AgentPlanner.

    Invariants:
    - action_type must be an AgentActionType enum.
    - TOOL_CALL requires a non-empty tool_name and a dictionary of arguments.
    - arguments dictionary is defensively copied to prevent external mutation.
    """

    action_type: AgentActionType
    tool_name: Optional[str] = None
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    thought: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, AgentActionType):
            raise TypeError(
                f"action_type must be an AgentActionType, got {type(self.action_type).__name__}"
            )

        if not isinstance(self.thought, str):
            raise TypeError("thought must be a string")

        if self.action_type == AgentActionType.TOOL_CALL:
            if not isinstance(self.tool_name, str) or not self.tool_name.strip():
                raise ValueError("TOOL_CALL action requires a non-empty tool_name")
            if not isinstance(self.arguments, dict):
                raise TypeError("TOOL_CALL action arguments must be a dict")
            # Defensive copy of arguments to prevent mutation
            object.__setattr__(self, "arguments", dict(self.arguments))
        else:
            if not isinstance(self.arguments, dict):
                raise TypeError("arguments must be a dict")
            object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class AgentStep:
    """Historical record of an individual planning and execution step.

    Invariants:
    - step_number must be an integer >= 1.
    - action must be an AgentAction instance.
    - tool_result must be None or a ToolExecutionResult.
    - elapsed_ms must be non-negative.
    """

    step_number: int
    action: AgentAction
    tool_result: Optional[ToolExecutionResult] = None
    elapsed_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.step_number, int) or self.step_number < 1:
            raise ValueError(f"step_number must be an integer >= 1, got {self.step_number}")
        if not isinstance(self.action, AgentAction):
            raise TypeError(f"action must be an AgentAction, got {type(self.action).__name__}")
        if self.tool_result is not None and not isinstance(self.tool_result, ToolExecutionResult):
            raise TypeError(
                f"tool_result must be a ToolExecutionResult, got {type(self.tool_result).__name__}"
            )
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms must be non-negative, got {self.elapsed_ms}")


@dataclass(frozen=True)
class AgentResult:
    """Structured envelope capturing the terminal outcome of an agent execution.

    Invariants:
    - status must be an AgentStatus enum.
    - is_success must be True if and only if status is COMPLETED.
    - Unsuccessful executions cannot be marked is_verified=True.
    - is_verified represents goal-level verification, distinct from tool-level verification.
    """

    goal: str
    status: AgentStatus
    output: Any = None
    steps: tuple[AgentStep, ...] = field(default_factory=tuple)
    total_steps: int = 0
    total_tool_calls: int = 0
    total_elapsed_ms: float = 0.0
    correlation_id: str = ""
    is_success: bool = False
    is_verified: bool = False
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AgentStatus):
            raise TypeError(f"status must be an AgentStatus, got {type(self.status).__name__}")

        if isinstance(self.steps, (list, tuple)):
            object.__setattr__(self, "steps", tuple(self.steps))
        else:
            raise TypeError("steps must be a sequence of AgentStep")

        if self.total_steps < 0:
            raise ValueError(f"total_steps must be non-negative, got {self.total_steps}")
        if self.total_tool_calls < 0:
            raise ValueError(f"total_tool_calls must be non-negative, got {self.total_tool_calls}")
        if self.total_elapsed_ms < 0:
            raise ValueError(f"total_elapsed_ms must be non-negative, got {self.total_elapsed_ms}")

        # Status and success consistency
        if self.status == AgentStatus.COMPLETED and not self.is_success:
            raise ValueError("AgentStatus.COMPLETED must have is_success=True")
        if self.status != AgentStatus.COMPLETED and self.is_success:
            raise ValueError(f"Status '{self.status.value}' cannot have is_success=True")

        # Verification invariant: only successful completions can be verified
        if not self.is_success and self.is_verified:
            raise ValueError("Unsuccessful agent execution cannot be marked is_verified=True")


class AgentPlanner(Protocol):
    """Protocol defining the interface for an agent planning component."""

    def plan(
        self,
        goal: str,
        available_tools: Sequence[ToolSpec],
        history: Sequence[AgentStep],
    ) -> AgentAction:
        """Determines the next action based on the goal, available tools, and history."""
        ...


class ScriptedPlanner:
    """Deterministic, zero-dependency planner for testing and fixed-sequence workflows.

    Supports either a static list of AgentActions or an injected decision callable.
    Guarantees predictable termination when actions are exhausted.
    """

    def __init__(
        self,
        actions_or_callable: Sequence[AgentAction]
        | Callable[[str, Sequence[ToolSpec], Sequence[AgentStep]], AgentAction],
    ) -> None:
        if callable(actions_or_callable):
            self._logic: Optional[
                Callable[[str, Sequence[ToolSpec], Sequence[AgentStep]], AgentAction]
            ] = actions_or_callable
            self._actions: Optional[list[AgentAction]] = None
        elif isinstance(actions_or_callable, (list, tuple)):
            norm_actions: list[AgentAction] = []
            for a in actions_or_callable:
                if not isinstance(a, AgentAction):
                    raise TypeError(f"Expected AgentAction instance, got {type(a).__name__}")
                norm_actions.append(a)
            self._logic = None
            self._actions = norm_actions
        else:
            raise TypeError("Expected sequence of AgentAction or decision callable")

    def plan(
        self,
        goal: str,
        available_tools: Sequence[ToolSpec],
        history: Sequence[AgentStep],
    ) -> AgentAction:
        if self._logic is not None:
            action = self._logic(goal, available_tools, history)
            if not isinstance(action, AgentAction):
                raise TypeError(
                    f"Planner callable returned {type(action).__name__}, expected AgentAction"
                )
            return action

        if self._actions is None or len(self._actions) == 0:
            return AgentAction(
                action_type=AgentActionType.FAIL,
                thought="Scripted action sequence exhausted without reaching a conclusion",
            )

        return self._actions.pop(0)


class Agent:
    """Bounded, deterministic agent coordinator.

    Responsibilities:
    - Iterates deliberation cycles up to max_steps.
    - Enforces tool-call limits strictly before dispatching tool calls.
    - Exposes only tools permitted for the caller when presenting tools to planner.
    - Dispatches tool calls exclusively through ToolOrchestratorBridge.
    - Never modifies caller permissions based on planner output.
    - Records comprehensive history without leaking sensitive payloads.
    - Keeps tool-output verification decoupled from goal-level verification.
    """

    def __init__(
        self,
        bridge: ToolOrchestratorBridge,
        planner: AgentPlanner,
        max_steps: int = 10,
        max_tool_calls: int = 10,
        goal_verifier: Optional[Callable[[Any], bool]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not isinstance(bridge, ToolOrchestratorBridge):
            raise TypeError(f"Expected ToolOrchestratorBridge, got {type(bridge).__name__}")
        if not hasattr(planner, "plan") or not callable(getattr(planner, "plan")):
            raise TypeError("planner must implement AgentPlanner with a callable plan() method")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be an integer >= 1, got {max_steps}")
        if not isinstance(max_tool_calls, int) or max_tool_calls < 1:
            raise ValueError(f"max_tool_calls must be an integer >= 1, got {max_tool_calls}")

        self._bridge = bridge
        self._planner = planner
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._goal_verifier = goal_verifier
        self._logger = logger or setup_logger("shar.agent")

    def run(
        self,
        goal: str,
        caller_permissions: frozenset[str] | set[str] = frozenset(),
        correlation_id: Optional[str] = None,
    ) -> AgentResult:
        """Executes the agent deliberation loop toward a user goal.

        Args:
            goal: Human-readable task or objective string.
            caller_permissions: Trusted application permissions granted to this run.
            correlation_id: Optional tracking identifier. Generated if omitted.

        Returns:
            A structured AgentResult detailing status, output, history, and diagnostics.
        """
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("Goal must be a non-empty string")

        perms = frozenset(caller_permissions)
        cid = correlation_id or uuid.uuid4().hex
        start_time = time.perf_counter()

        self._logger.info(
            f"Starting agent execution for goal: {goal!r}",
            extra={"correlation_id": cid, "goal": goal},
        )

        steps: list[AgentStep] = []
        tool_call_count = 0

        while len(steps) < self._max_steps:
            step_number = len(steps) + 1
            step_start = time.perf_counter()

            # 1. Query tools visible to caller permissions (never expose forbidden tools to planner)
            available_tools = self._bridge._registry.list_tools(caller_permissions=perms)

            # 2. Consult planner under strict exception containment
            try:
                action = self._planner.plan(goal, available_tools, tuple(steps))
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = f"Planner raised an exception: {type(exc).__name__}: {exc}"
                self._logger.error(
                    err_msg,
                    extra={"correlation_id": cid, "step_number": step_number},
                    exc_info=True,
                )
                return AgentResult(
                    goal=goal,
                    status=AgentStatus.PLANNING_ERROR,
                    output=None,
                    steps=tuple(steps),
                    total_steps=len(steps),
                    total_tool_calls=tool_call_count,
                    total_elapsed_ms=elapsed_ms,
                    correlation_id=cid,
                    is_success=False,
                    is_verified=False,
                    error=err_msg,
                )

            # 3. Validate planner output structure
            if not isinstance(action, AgentAction):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                err_msg = (
                    f"Planner returned invalid type '{type(action).__name__}', expected AgentAction"
                )
                self._logger.error(
                    err_msg,
                    extra={"correlation_id": cid, "step_number": step_number},
                )
                return AgentResult(
                    goal=goal,
                    status=AgentStatus.PLANNING_ERROR,
                    output=None,
                    steps=tuple(steps),
                    total_steps=len(steps),
                    total_tool_calls=tool_call_count,
                    total_elapsed_ms=elapsed_ms,
                    correlation_id=cid,
                    is_success=False,
                    is_verified=False,
                    error=err_msg,
                )

            # 4. Handle FINISH action
            if action.action_type == AgentActionType.FINISH:
                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                step_record = AgentStep(
                    step_number=step_number,
                    action=action,
                    tool_result=None,
                    elapsed_ms=step_elapsed,
                )
                steps.append(step_record)
                total_elapsed = (time.perf_counter() - start_time) * 1000.0

                # Goal-level verification: distinct from tool-level verification
                is_verified = False
                if self._goal_verifier is not None:
                    try:
                        verdict = self._goal_verifier(action.output)
                        is_verified = (verdict is True)
                    except Exception:
                        is_verified = False

                self._logger.info(
                    f"Agent finished goal successfully (verified={is_verified})",
                    extra={"correlation_id": cid, "total_steps": len(steps)},
                )
                return AgentResult(
                    goal=goal,
                    status=AgentStatus.COMPLETED,
                    output=action.output,
                    steps=tuple(steps),
                    total_steps=len(steps),
                    total_tool_calls=tool_call_count,
                    total_elapsed_ms=total_elapsed,
                    correlation_id=cid,
                    is_success=True,
                    is_verified=is_verified,
                    error=None,
                )

            # 5. Handle FAIL action
            if action.action_type == AgentActionType.FAIL:
                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                step_record = AgentStep(
                    step_number=step_number,
                    action=action,
                    tool_result=None,
                    elapsed_ms=step_elapsed,
                )
                steps.append(step_record)
                total_elapsed = (time.perf_counter() - start_time) * 1000.0

                fail_msg = action.thought or "Planner indicated goal failure"
                self._logger.warning(
                    f"Agent planner failed goal: {fail_msg}",
                    extra={"correlation_id": cid, "total_steps": len(steps)},
                )
                return AgentResult(
                    goal=goal,
                    status=AgentStatus.FAILED,
                    output=None,
                    steps=tuple(steps),
                    total_steps=len(steps),
                    total_tool_calls=tool_call_count,
                    total_elapsed_ms=total_elapsed,
                    correlation_id=cid,
                    is_success=False,
                    is_verified=False,
                    error=fail_msg,
                )

            # 6. Handle TOOL_CALL action
            if action.action_type == AgentActionType.TOOL_CALL:
                # Enforce tool-call limit strictly BEFORE invoking the tool
                if tool_call_count >= self._max_tool_calls:
                    step_elapsed = (time.perf_counter() - step_start) * 1000.0
                    step_record = AgentStep(
                        step_number=step_number,
                        action=action,
                        tool_result=None,
                        elapsed_ms=step_elapsed,
                    )
                    steps.append(step_record)
                    total_elapsed = (time.perf_counter() - start_time) * 1000.0

                    err_msg = (
                        f"Tool-call limit of {self._max_tool_calls} reached; "
                        f"refusing tool '{action.tool_name}'"
                    )
                    self._logger.warning(
                        err_msg,
                        extra={"correlation_id": cid, "tool_name": action.tool_name},
                    )
                    return AgentResult(
                        goal=goal,
                        status=AgentStatus.TOOL_CALL_LIMIT_EXCEEDED,
                        output=None,
                        steps=tuple(steps),
                        total_steps=len(steps),
                        total_tool_calls=tool_call_count,
                        total_elapsed_ms=total_elapsed,
                        correlation_id=cid,
                        is_success=False,
                        is_verified=False,
                        error=err_msg,
                    )

                # Tool call authorized under limits: increment counter and execute via bridge
                tool_call_count += 1
                assert action.tool_name is not None
                tool_result = self._bridge.execute_tool(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                    caller_permissions=perms,
                    correlation_id=cid,
                )

                step_elapsed = (time.perf_counter() - step_start) * 1000.0
                step_record = AgentStep(
                    step_number=step_number,
                    action=action,
                    tool_result=tool_result,
                    elapsed_ms=step_elapsed,
                )
                steps.append(step_record)

        # 7. Loop completed without reaching FINISH or FAIL: step limit exceeded
        total_elapsed = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Step limit of {self._max_steps} reached without completing goal"
        self._logger.warning(
            err_msg,
            extra={"correlation_id": cid, "total_steps": len(steps)},
        )
        return AgentResult(
            goal=goal,
            status=AgentStatus.STEP_LIMIT_EXCEEDED,
            output=None,
            steps=tuple(steps),
            total_steps=len(steps),
            total_tool_calls=tool_call_count,
            total_elapsed_ms=total_elapsed,
            correlation_id=cid,
            is_success=False,
            is_verified=False,
            error=err_msg,
        )

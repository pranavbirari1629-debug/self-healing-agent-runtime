"""Deterministic unit and integration tests for Phase 6 Tool Integration in shar.tools."""

import time
import pytest

from shar import (
    FailureCategory,
    FailureType,
    RecoveryOrchestrator,
    RecoveryResult,
    RecoveryStatus,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolOrchestratorBridge,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
    UNSET,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# 1. ToolParameter Validation Tests
# ---------------------------------------------------------------------------

def test_tool_parameter_valid_required():
    param = ToolParameter(name="user_id", type=int, description="User identifier")
    assert param.name == "user_id"
    assert param.type is int
    assert param.description == "User identifier"
    assert param.required is True
    assert param.default is UNSET


def test_tool_parameter_valid_optional_with_default():
    param = ToolParameter(name="limit", type=int, required=False, default=10)
    assert param.name == "limit"
    assert param.required is False
    assert param.default == 10


def test_tool_parameter_valid_optional_nullable():
    param = ToolParameter(name="filter_query", type=str, required=False, default=None)
    assert param.name == "filter_query"
    assert param.required is False
    assert param.default is None


def test_tool_parameter_empty_or_whitespace_name():
    with pytest.raises(ValueError, match="non-empty string"):
        ToolParameter(name="", type=str)
    with pytest.raises(ValueError, match="non-empty string"):
        ToolParameter(name="   ", type=str)


def test_tool_parameter_invalid_identifier_name():
    with pytest.raises(ValueError, match="valid Python identifier"):
        ToolParameter(name="123invalid", type=int)
    with pytest.raises(ValueError, match="valid Python identifier"):
        ToolParameter(name="has-dash", type=int)
    with pytest.raises(ValueError, match="valid Python identifier"):
        ToolParameter(name="has space", type=int)


def test_tool_parameter_invalid_type():
    with pytest.raises(TypeError, match="must be a Python type"):
        ToolParameter(name="query", type="str")  # string instead of type


def test_tool_parameter_required_cannot_have_default():
    with pytest.raises(ValueError, match="cannot specify a default value"):
        ToolParameter(name="user_id", type=int, required=True, default=42)


def test_tool_parameter_optional_must_specify_default():
    with pytest.raises(ValueError, match="must specify a default value"):
        ToolParameter(name="limit", type=int, required=False)


def test_tool_parameter_default_type_mismatch():
    with pytest.raises(TypeError, match="does not match expected type"):
        ToolParameter(name="count", type=int, required=False, default="not_an_int")


# ---------------------------------------------------------------------------
# 2. ToolSpec Validation Tests
# ---------------------------------------------------------------------------

def test_tool_spec_valid():
    spec = ToolSpec(
        name="calculate_sum",
        description="Adds two integers",
        callable=lambda a, b: a + b,
        parameters=(
            ToolParameter(name="a", type=int),
            ToolParameter(name="b", type=int),
        ),
        required_permissions=frozenset({"compute"}),
        is_idempotent=True,
    )
    assert spec.name == "calculate_sum"
    assert spec.is_idempotent is True
    assert spec.is_read_only is False
    assert spec.max_retries == 1
    assert "compute" in spec.required_permissions


def test_tool_spec_defaults_are_conservative():
    spec = ToolSpec(
        name="raw_action",
        description="Raw action",
        callable=lambda: None,
    )
    assert spec.is_idempotent is False
    assert spec.is_read_only is False
    assert spec.safe_to_retry_on_timeout is False
    assert spec.max_retries == 1
    assert spec.parameters == ()
    assert spec.required_permissions == frozenset()


def test_tool_spec_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        ToolSpec(name="", description="desc", callable=lambda: None)


def test_tool_spec_non_callable():
    with pytest.raises(TypeError, match="callable object"):
        ToolSpec(name="bad", description="desc", callable="not_callable")


def test_tool_spec_duplicate_parameters():
    with pytest.raises(ValueError, match="Duplicate parameter names"):
        ToolSpec(
            name="dup_tool",
            description="desc",
            callable=lambda: None,
            parameters=(
                ToolParameter(name="item", type=str),
                ToolParameter(name="item", type=int),
            ),
        )


def test_tool_spec_read_only_must_be_idempotent():
    with pytest.raises(ValueError, match="Read-only tools must declare is_idempotent=True"):
        ToolSpec(
            name="read_data",
            description="Read query",
            callable=lambda: "data",
            is_read_only=True,
            is_idempotent=False,
        )


def test_tool_spec_self_fallback_prohibited():
    with pytest.raises(ValueError, match="cannot designate itself as its own fallback"):
        ToolSpec(
            name="loop_tool",
            description="desc",
            callable=lambda: None,
            fallback_tool_name="loop_tool",
        )


def test_tool_spec_numeric_boundaries():
    with pytest.raises(ValueError, match="max_retries"):
        ToolSpec(name="bad_retries", description="desc", callable=lambda: None, max_retries=-1)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ToolSpec(name="bad_timeout", description="desc", callable=lambda: None, timeout_seconds=-0.5)
    with pytest.raises(ValueError, match="backoff_max_delay"):
        ToolSpec(
            name="bad_delay",
            description="desc",
            callable=lambda: None,
            backoff_base_delay=1.0,
            backoff_max_delay=0.5,
        )


# ---------------------------------------------------------------------------
# 3. ToolRegistry Tests
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    registry = ToolRegistry()
    tool = ToolSpec(name="echo", description="Echo", callable=lambda text: text)
    registry.register(tool)

    assert "echo" in registry
    assert len(registry) == 1
    assert registry.get("echo") is tool
    assert registry.get("non_existent") is None


def test_registry_rejects_duplicate():
    registry = ToolRegistry()
    tool1 = ToolSpec(name="echo", description="Echo 1", callable=lambda: 1)
    tool2 = ToolSpec(name="echo", description="Echo 2", callable=lambda: 2)

    registry.register(tool1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool2)


def test_registry_list_tools_unfiltered_and_filtered():
    registry = ToolRegistry()
    t_public = ToolSpec(
        name="public_info",
        description="Public",
        callable=lambda: "pub",
    )
    t_admin = ToolSpec(
        name="admin_reboot",
        description="Admin",
        callable=lambda: "reboot",
        required_permissions=frozenset({"admin"}),
    )
    t_user = ToolSpec(
        name="read_profile",
        description="Profile",
        callable=lambda: "profile",
        required_permissions=frozenset({"read_profile"}),
    )

    registry.register(t_admin)
    registry.register(t_public)
    registry.register(t_user)

    # Unfiltered lists all sorted by name
    all_tools = registry.list_tools()
    assert [t.name for t in all_tools] == ["admin_reboot", "public_info", "read_profile"]

    # Filtered by user permission: sees public and user, but not admin
    user_tools = registry.list_tools(caller_permissions=frozenset({"read_profile"}))
    assert [t.name for t in user_tools] == ["public_info", "read_profile"]

    # Empty permissions: sees only tools requiring zero permissions
    empty_tools = registry.list_tools(caller_permissions=frozenset())
    assert [t.name for t in empty_tools] == ["public_info"]


# ---------------------------------------------------------------------------
# 4. Argument Validation & Type Checking Tests (Bool-as-Int Guard)
# ---------------------------------------------------------------------------

def test_bridge_rejects_bool_for_int():
    registry = ToolRegistry()
    call_count = 0

    def work(count: int):
        nonlocal call_count
        call_count += 1
        return count * 2

    tool = ToolSpec(
        name="doubler",
        description="Doubles an integer",
        callable=work,
        parameters=(ToolParameter(name="count", type=int),),
    )
    registry.register(tool)
    bridge = ToolOrchestratorBridge(registry)

    # Calling with bool True when expected type is int
    result = bridge.execute_tool("doubler", arguments={"count": True})
    assert result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.is_success is False
    assert "Invalid type for parameter 'count'" in result.error
    assert "expected int, got bool" in result.error
    assert call_count == 0  # Callable was NEVER invoked!


def test_bridge_rejects_missing_required_arguments():
    registry = ToolRegistry()
    call_count = 0

    def greet(name: str):
        nonlocal call_count
        call_count += 1
        return f"Hello, {name}"

    registry.register(ToolSpec(
        name="greet",
        description="Greeting",
        callable=greet,
        parameters=(ToolParameter(name="name", type=str),),
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("greet", arguments={})
    assert result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.is_success is False
    assert "Missing required parameter" in result.error
    assert call_count == 0


def test_bridge_rejects_unknown_arguments():
    registry = ToolRegistry()
    call_count = 0

    def ping():
        nonlocal call_count
        call_count += 1
        return "pong"

    registry.register(ToolSpec(
        name="ping",
        description="Ping",
        callable=ping,
        parameters=(),
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("ping", arguments={"extra_flag": 123})
    assert result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.is_success is False
    assert "Unknown parameter(s)" in result.error
    assert call_count == 0


def test_bridge_populates_optional_defaults():
    registry = ToolRegistry()

    def query(text: str, limit: int = 10, offset: int = 0):
        return {"text": text, "limit": limit, "offset": offset}

    registry.register(ToolSpec(
        name="query",
        description="Search",
        callable=query,
        parameters=(
            ToolParameter(name="text", type=str),
            ToolParameter(name="limit", type=int, required=False, default=10),
            ToolParameter(name="offset", type=int, required=False, default=0),
        ),
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("query", arguments={"text": "search terms"})
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.output == {"text": "search terms", "limit": 10, "offset": 0}


# ---------------------------------------------------------------------------
# 5. Permission Enforcement Tests
# ---------------------------------------------------------------------------

def test_bridge_rejects_unauthorized_execution():
    registry = ToolRegistry()
    call_count = 0

    def delete_db():
        nonlocal call_count
        call_count += 1
        return "dropped"

    registry.register(ToolSpec(
        name="delete_db",
        description="Destructive",
        callable=delete_db,
        required_permissions=frozenset({"db:drop"}),
    ))
    bridge = ToolOrchestratorBridge(registry)

    # Caller has no permissions
    result = bridge.execute_tool("delete_db", arguments={}, caller_permissions=frozenset())
    assert result.status == ToolExecutionStatus.PERMISSION_DENIED
    assert result.is_success is False
    assert "db:drop" in result.error
    assert call_count == 0  # Callable was NEVER invoked!


def test_bridge_allows_authorized_execution():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="delete_db",
        description="Destructive",
        callable=lambda: "dropped",
        required_permissions=frozenset({"db:drop"}),
        is_idempotent=True,
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool(
        "delete_db", arguments={}, caller_permissions=frozenset({"db:drop", "other"})
    )
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.output == "dropped"


def test_bridge_tool_not_found():
    registry = ToolRegistry()
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("missing_tool", arguments={})
    assert result.status == ToolExecutionStatus.TOOL_NOT_FOUND
    assert result.is_success is False
    assert "missing_tool" in result.error


# ---------------------------------------------------------------------------
# 6. Output Verification & Result Invariant Tests
# ---------------------------------------------------------------------------

def test_bridge_unverified_execution():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="raw_calc",
        description="Unverified",
        callable=lambda x: x + 1,
        parameters=(ToolParameter(name="x", type=int),),
        verifier=None,
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("raw_calc", arguments={"x": 5})
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.output == 6
    assert result.verification_status == VerificationStatus.UNVERIFIED
    assert result.is_verified is False  # Must NOT mark unverified as verified!


def test_bridge_verified_execution():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="auth_check",
        description="Verified auth",
        callable=lambda: {"auth": True, "token": "xyz"},
        verifier=lambda out: isinstance(out, dict) and out.get("auth") is True,
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("auth_check", arguments={})
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.verification_status == VerificationStatus.VERIFIED
    assert result.is_verified is True


def test_bridge_failed_verification():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="corrupt_check",
        description="Fails verifier",
        callable=lambda: {"auth": False},
        verifier=lambda out: out.get("auth") is True,
    ))
    bridge = ToolOrchestratorBridge(registry)

    result = bridge.execute_tool("corrupt_check", arguments={})
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True  # Execution succeeded
    assert result.verification_status == VerificationStatus.FAILED  # Verification failed
    assert result.is_verified is False


# ---------------------------------------------------------------------------
# 7. Self-Healing Retries & Non-Idempotent Tool Safety Tests
# ---------------------------------------------------------------------------

def test_bridge_retries_transient_failure_for_idempotent_tool():
    registry = ToolRegistry()
    attempts = 0

    def flaky_fetch():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Transient network glitch")
        return {"data": "fresh_data"}

    registry.register(ToolSpec(
        name="fetch_data",
        description="Fetches data with retry",
        callable=flaky_fetch,
        is_idempotent=True,
        max_retries=2,
        backoff_base_delay=0.001,
        verifier=lambda out: out.get("data") == "fresh_data",
    ))

    # Fast test runner with no-op sleep
    orchestrator = RecoveryOrchestrator(sleep_fn=lambda _: None)
    bridge = ToolOrchestratorBridge(registry, orchestrator=orchestrator)

    result = bridge.execute_tool("fetch_data", arguments={})
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.output == {"data": "fresh_data"}
    assert result.is_verified is True
    assert attempts == 2
    assert result.recovery_result is not None
    assert result.recovery_result.status == RecoveryStatus.RECOVERED_VIA_RETRY


def test_bridge_does_not_retry_non_idempotent_tool():
    registry = ToolRegistry()
    attempts = 0

    def charge_card():
        nonlocal attempts
        attempts += 1
        raise ConnectionError("Interrupted connection")

    registry.register(ToolSpec(
        name="charge_card",
        description="Payment action",
        callable=charge_card,
        is_idempotent=False,  # NOT safe to retry!
        max_retries=3,
    ))

    bridge = ToolOrchestratorBridge(registry)
    result = bridge.execute_tool("charge_card", arguments={})

    assert result.status == ToolExecutionStatus.EXECUTION_FAILED
    assert result.is_success is False
    assert result.is_verified is False
    assert attempts == 1  # Exactly ONE attempt; no retries performed
    assert result.recovery_result is not None
    assert result.recovery_result.status == RecoveryStatus.FAILED_NON_IDEMPOTENT


# ---------------------------------------------------------------------------
# 8. Timeout Safety Tests
# ---------------------------------------------------------------------------

def test_bridge_handles_timeout_safely():
    registry = ToolRegistry()

    def slow_worker():
        time.sleep(0.3)
        return "finished"

    registry.register(ToolSpec(
        name="slow_task",
        description="Slow task",
        callable=slow_worker,
        timeout_seconds=0.05,
        safe_to_retry_on_timeout=False,
        max_retries=2,
    ))

    bridge = ToolOrchestratorBridge(registry)
    result = bridge.execute_tool("slow_task", arguments={})

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.is_success is False
    assert result.is_verified is False
    assert result.recovery_result is not None
    assert result.recovery_result.status == RecoveryStatus.FAILED_TIMEOUT_UNSAFE


# ---------------------------------------------------------------------------
# 9. Fallback Tool Delegation & Fallback Permissions Tests
# ---------------------------------------------------------------------------

def test_bridge_fallback_success():
    registry = ToolRegistry()

    def primary_failing(url: str):
        raise ValueError("Invalid endpoint configuration")  # Permanent failure

    def fallback_working(url: str):
        return {"cached": True, "url": url}

    registry.register(ToolSpec(
        name="cached_fetch",
        description="Fallback cache fetcher",
        callable=fallback_working,
        parameters=(ToolParameter(name="url", type=str),),
        is_idempotent=True,
    ))

    registry.register(ToolSpec(
        name="network_fetch",
        description="Primary network fetcher",
        callable=primary_failing,
        parameters=(ToolParameter(name="url", type=str),),
        is_idempotent=True,
        fallback_tool_name="cached_fetch",
    ))

    bridge = ToolOrchestratorBridge(registry)
    result = bridge.execute_tool("network_fetch", arguments={"url": "https://example.com"})

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.is_success is True
    assert result.output == {"cached": True, "url": "https://example.com"}
    assert result.recovery_result is not None
    assert result.recovery_result.status == RecoveryStatus.RECOVERED_VIA_FALLBACK


def test_bridge_rejects_fallback_if_caller_lacks_fallback_permissions():
    """Prevents privilege escalation through fallbacks."""
    registry = ToolRegistry()
    call_count = 0

    def primary(cmd: str):
        nonlocal call_count
        call_count += 1
        return "primary_out"

    def fallback(cmd: str):
        return "escalated_out"

    registry.register(ToolSpec(
        name="admin_tool",
        description="High privilege fallback",
        callable=fallback,
        parameters=(ToolParameter(name="cmd", type=str),),
        required_permissions=frozenset({"admin"}),
    ))

    registry.register(ToolSpec(
        name="user_tool",
        description="Normal tool that configures admin fallback",
        callable=primary,
        parameters=(ToolParameter(name="cmd", type=str),),
        required_permissions=frozenset({"user"}),
        fallback_tool_name="admin_tool",
    ))

    bridge = ToolOrchestratorBridge(registry)

    # Caller only has "user" permission, not "admin"
    result = bridge.execute_tool(
        "user_tool", arguments={"cmd": "status"}, caller_permissions=frozenset({"user"})
    )
    assert result.status == ToolExecutionStatus.PERMISSION_DENIED
    assert result.is_success is False
    assert "admin_tool" in result.error
    assert call_count == 0  # Neither tool callable was invoked!


def test_bridge_rejects_incompatible_fallback_arguments():
    registry = ToolRegistry()

    registry.register(ToolSpec(
        name="specialized_fallback",
        description="Requires specialized secret_key",
        callable=lambda key: key,
        parameters=(ToolParameter(name="secret_key", type=str),),
    ))

    registry.register(ToolSpec(
        name="generic_tool",
        description="Only accepts simple query",
        callable=lambda query: query,
        parameters=(ToolParameter(name="query", type=str),),
        fallback_tool_name="specialized_fallback",
    ))

    bridge = ToolOrchestratorBridge(registry)

    # Passing query is incompatible with specialized_fallback's required secret_key
    result = bridge.execute_tool("generic_tool", arguments={"query": "hello"})
    assert result.status == ToolExecutionStatus.INVALID_ARGUMENTS
    assert result.is_success is False
    assert "Incompatible arguments for fallback tool" in result.error


# ---------------------------------------------------------------------------
# 10. Correlation ID & Public API Exports Tests
# ---------------------------------------------------------------------------

def test_bridge_correlation_id_propagation():
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="action",
        description="Action",
        callable=lambda: "ok",
    ))
    bridge = ToolOrchestratorBridge(registry)

    # Custom correlation ID
    res1 = bridge.execute_tool("action", arguments={}, correlation_id="trace-xyz-999")
    assert res1.correlation_id == "trace-xyz-999"
    assert res1.recovery_result is not None
    assert res1.recovery_result.correlation_id == "trace-xyz-999"

    # Generated correlation ID
    res2 = bridge.execute_tool("action", arguments={})
    assert res2.correlation_id != ""
    assert res2.recovery_result is not None
    assert res2.recovery_result.correlation_id == res2.correlation_id


def test_public_api_exports():
    import shar

    for symbol in (
        "ToolParameter",
        "ToolSpec",
        "ToolRegistry",
        "ToolExecutionStatus",
        "ToolExecutionResult",
        "ToolOrchestratorBridge",
        "UNSET",
    ):
        assert hasattr(shar, symbol), f"shar is missing export '{symbol}'"

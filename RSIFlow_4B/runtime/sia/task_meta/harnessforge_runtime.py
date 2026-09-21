"""Execute immutable HarnessForge candidate bundles with the pinned runtime.

This module is deliberately an adapter, not a second harness implementation.
Candidate ``builder.py`` and its Planning, Action, and Memory providers execute
against the real ``Agents`` implementation vendored under
``upstream/HarnessForge_4B``.  The only local responsibilities are translating
the Task model/environment protocols and returning the rollout evidence shape
consumed by :mod:`sia.task_meta.pipeline_execution`.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from sia.task_meta.harnessforge_manifest import HarnessBundleManifest
from sia.task_meta.task_client import TaskContextBudgetExceeded


_IMPORT_LOCK = threading.RLock()


class HarnessForgeBudgetExceeded(RuntimeError):
    """The adapter rejected a call before dispatch because its fixed budget ended."""


def _upstream_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "upstream" / "HarnessForge_4B"
    required = (
        root / "Agents" / "agents.py",
        root / "harness" / "module_action" / "base_action.py",
        root / "harness_runtime.py",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Pinned HarnessForge runtime is incomplete: {root}")
    return root


def _assert_pinned_module(module: Any) -> None:
    source = getattr(module, "__file__", None)
    if not source:
        raise ImportError(f"HarnessForge native module has no source path: {module!r}")
    try:
        Path(source).resolve().relative_to(_upstream_root())
    except ValueError as exc:
        raise ImportError(
            f"Refusing non-pinned HarnessForge module {module.__name__}: {source}"
        ) from exc


def _assert_factory_shared_module(module: Any) -> None:
    """Require the official active package, keeping ``harness`` fallback-only."""

    source = Path(getattr(module, "__file__", "")).resolve()
    preferred = (_upstream_root() / "harness_factory").resolve()
    try:
        source.relative_to(preferred)
    except ValueError as exc:
        raise ImportError(
            f"HarnessForge shared module did not load from harness_factory: {source}"
        ) from exc


def _install_json_repair_fallback() -> bool:
    """Provide strict JSON parsing when the optional upstream helper is absent.

    Production HarnessForge environments install ``json-repair``.  The Task
    runtime's CPU validation environment intentionally has fewer optional
    packages; valid JSON must still exercise the exact upstream agent classes.
    Malformed JSON remains a normal agent parse failure in this fallback.
    """

    try:
        importlib.import_module("json_repair")
        return False
    except ModuleNotFoundError:
        fallback = types.ModuleType("json_repair")
        fallback.loads = json.loads  # type: ignore[attr-defined]
        sys.modules["json_repair"] = fallback
        return True


@contextmanager
def _activated_candidate(
    manifest: HarnessBundleManifest,
) -> Iterator[tuple[Path, str]]:
    """Materialize and import one bundle without reusing another candidate."""

    upstream = _upstream_root()
    with tempfile.TemporaryDirectory(prefix="sia_harnessforge_") as temporary:
        temporary_root = Path(temporary)
        package_name = f"_sia_hf_{manifest.bundle_sha256[:24]}"
        candidate_root = manifest.materialize(temporary_root / package_name)
        added_paths = [
            temporary_root,
            upstream,
            upstream / "harness_factory",
            upstream / "harness",
        ]
        native_prefixes = (
            "Agents",
            "harness_runtime",
            "module_action",
            "module_planning",
            "module_memory",
        )
        displaced_modules = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if any(name == prefix or name.startswith(prefix + ".") for prefix in native_prefixes)
        }
        old_path = list(sys.path)
        old_package = os.environ.get("HARNESS_PACKAGE")
        old_package_root = os.environ.get("HARNESS_PACKAGE_ROOT")
        fallback_installed = False
        try:
            os.environ["HARNESS_PACKAGE"] = "harness_factory"
            os.environ["HARNESS_PACKAGE_ROOT"] = str(upstream)
            for path in reversed(added_paths):
                value = str(path)
                while value in sys.path:
                    sys.path.remove(value)
                sys.path.insert(0, value)
            fallback_installed = _install_json_repair_fallback()
            importlib.invalidate_caches()
            yield candidate_root, package_name
        finally:
            for module_name in sorted(
                (name for name in sys.modules if name == package_name or name.startswith(package_name + ".")),
                reverse=True,
            ):
                sys.modules.pop(module_name, None)
            for module_name in list(sys.modules):
                if any(
                    module_name == prefix or module_name.startswith(prefix + ".")
                    for prefix in native_prefixes
                ):
                    sys.modules.pop(module_name, None)
            sys.modules.update(displaced_modules)
            if fallback_installed:
                sys.modules.pop("json_repair", None)
            sys.path[:] = old_path
            if old_package is None:
                os.environ.pop("HARNESS_PACKAGE", None)
            else:
                os.environ["HARNESS_PACKAGE"] = old_package
            if old_package_root is None:
                os.environ.pop("HARNESS_PACKAGE_ROOT", None)
            else:
                os.environ["HARNESS_PACKAGE_ROOT"] = old_package_root
            importlib.invalidate_caches()


def _plain_role(value: Any) -> str:
    role = getattr(value, "value", value)
    role = str(role)
    return {"tool-response": "user", "tool-call": "assistant"}.get(role, role)


def _plain_content(value: Any) -> Any:
    if not isinstance(value, list):
        return copy.deepcopy(value)
    text_parts: list[str] = []
    non_text: list[Any] = []
    for item in value:
        if isinstance(item, Mapping) and item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        else:
            non_text.append(copy.deepcopy(item))
    if not non_text:
        return "".join(text_parts)
    return copy.deepcopy(value)


def _transport_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transported: list[dict[str, Any]] = []
    for message in messages:
        normalized = {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if key not in {"role", "content"}
        }
        normalized["role"] = _plain_role(message.get("role", "user"))
        normalized["content"] = _plain_content(message.get("content", ""))
        transported.append(normalized)
    return transported


def _operation_from_stack() -> str:
    """Name the native operation without altering upstream call signatures."""

    for frame_info in inspect.stack(context=0)[2:16]:
        function = frame_info.function
        module = str(frame_info.frame.f_globals.get("__name__", ""))
        if function == "provide_final_answer":
            return "budget_finalization"
        if function == "topology_initialize":
            return "planning"
        if function == "adaptation":
            return "summary"
        if "memory_module" in module:
            if "prune" in function:
                return "memory_prune"
            return "memory_extract"
        if module == "Agents.agents" and function == "step":
            return "action"
    return "harnessforge_model"


def _assistant_payload(response: Any) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(response, Mapping):
        raise ValueError("Task model response must be an object")
    assistant = response.get("message", response.get("assistant", response))
    if not isinstance(assistant, Mapping):
        raise ValueError("Task model returned no assistant message")
    role = _plain_role(assistant.get("role", "assistant"))
    if role != "assistant":
        raise ValueError("Task model returned a non-assistant message")
    result = {
        key: copy.deepcopy(value)
        for key, value in assistant.items()
        if key in {"role", "content", "reasoning_content", "tool_calls"}
    }
    result["role"] = "assistant"
    result.setdefault("content", "")
    return result, response


class _ModelAdapter:
    def __init__(
        self,
        model_callable: Any,
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        max_model_calls: int,
    ) -> None:
        self._model_callable = model_callable
        self._seed = seed
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_model_calls = max_model_calls
        self.calls: list[dict[str, Any]] = []
        self.budget_exhausted = False
        self.context_budget_exhausted = False
        self.model_id = "task_model_callable"
        self._last_counts = {"input_token_count": 0, "output_token_count": 0}

    def __call__(self, messages: list[dict[str, Any]], **_: Any) -> Any:
        if self.context_budget_exhausted:
            raise TaskContextBudgetExceeded('Task context budget already exhausted; no generation dispatched')
        if len(self.calls) >= self._max_model_calls:
            self.budget_exhausted = True
            raise HarnessForgeBudgetExceeded("Maximum Task model calls reached")
        actual = _transport_messages(messages)
        operation = _operation_from_stack()
        call_seed = self._seed + len(self.calls)
        record: dict[str, Any] = {
            "call_id": len(self.calls),
            "operation": operation,
            "messages": copy.deepcopy(actual),
            "seed": call_seed,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "status": "started",
            "response_received": False,
        }
        self.calls.append(record)
        started = time.monotonic()
        try:
            raw_response = self._model_callable(
                actual,
                tools=None,
                seed=call_seed,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            record["response_received"] = True
            assistant, response = _assistant_payload(raw_response)
            usage = copy.deepcopy(response.get("usage"))
            record.update(
                assistant=copy.deepcopy(assistant),
                usage=usage,
                binding=copy.deepcopy(response.get("binding")),
                finish_reason=response.get("finish_reason"),
                status="completed",
            )
            if isinstance(usage, Mapping):
                self._last_counts = {
                    "input_token_count": int(usage.get("prompt_tokens") or 0),
                    "output_token_count": int(usage.get("completion_tokens") or 0),
                }
            models = importlib.import_module("Agents.models")
            value = copy.deepcopy(assistant)
            raw_tool_calls = value.get("tool_calls") or []
            normalized_calls = []
            for index, tool_call in enumerate(raw_tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                function = tool_call.get("function") or {}
                if not isinstance(function, Mapping) or not function.get("name"):
                    continue
                normalized_calls.append(
                    models.ChatMessageToolCall(
                        function=models.ChatMessageToolCallDefinition(
                            name=str(function["name"]),
                            arguments=copy.deepcopy(function.get("arguments", {})),
                            description=function.get("description"),
                        ),
                        id=str(tool_call.get("id", f"call_{index}")),
                        type=str(tool_call.get("type", "function")),
                    )
                )
            return models.ChatMessage(
                role="assistant",
                content=value.get("content"),
                reasoning_content=value.get("reasoning_content"),
                tool_calls=normalized_calls or None,
                raw=copy.deepcopy(raw_response),
            )
        except TaskContextBudgetExceeded as exc:
            self.context_budget_exhausted = True
            record.update(status="rejected", error_type=type(exc).__name__,
                          generation_dispatched=False, usage_complete=True)
            raise
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["wall_time_seconds"] = time.monotonic() - started

    def get_token_counts(self) -> dict[str, int]:
        return dict(self._last_counts)


def _normalize_schema(
    schema: Mapping[str, Any],
) -> tuple[str, str, dict[str, dict[str, Any]], dict[str, Any]]:
    function = schema.get("function", schema)
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise ValueError("Environment tool schemas must declare a function name")
    parameters = function.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Invalid parameters for environment tool {function['name']}")
    properties = parameters.get("properties") or {}
    if not isinstance(properties, Mapping):
        raise ValueError(f"Invalid properties for environment tool {function['name']}")
    required = set(parameters.get("required") or [])
    inputs: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in properties.items():
        if not isinstance(raw_name, str) or not isinstance(raw_spec, Mapping):
            raise ValueError(f"Invalid input schema for environment tool {function['name']}")
        item = copy.deepcopy(dict(raw_spec))
        item.setdefault("type", "any")
        item.setdefault("description", f"Argument {raw_name}")
        if raw_name not in required:
            item["nullable"] = True
        inputs[raw_name] = item
    validation_schema = copy.deepcopy(dict(parameters))
    validation_schema.setdefault("type", "object")
    validation_schema.setdefault("additionalProperties", False)
    return str(function["name"]), str(function.get("description", "")), inputs, validation_schema


class _ToolRecorder:
    def __init__(self, environment: Any, max_tool_calls: int) -> None:
        self.environment = environment
        self.max_tool_calls = max_tool_calls
        self.calls: list[dict[str, Any]] = []
        self.budget_exhausted = False
        self._lock = threading.Lock()

    def invoke(self, name: str, arguments: dict[str, Any], schema: dict[str, Any]) -> Any:
        with self._lock:
            if len(self.calls) >= self.max_tool_calls:
                self.budget_exhausted = True
                raise HarnessForgeBudgetExceeded("Maximum Task tool calls reached")
            record = {
                "call_id": len(self.calls),
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "status": "started",
            }
            self.calls.append(record)
            try:
                from jsonschema import ValidationError, validate

                try:
                    validate(instance=arguments, schema=schema)
                except ValidationError as exc:
                    record.update(status="task_error", error_type="ValidationError")
                    raise ValueError(f"Invalid arguments for environment tool {name}: {exc.message}") from exc
                observation = self.environment.step(name, arguments)
                record.update(observation=copy.deepcopy(observation), status="completed")
                return observation
            except (TypeError, ValueError) as exc:
                record.update(status="task_error", error_type=type(exc).__name__)
                raise
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                raise


def _environment_tools(environment: Any, recorder: _ToolRecorder) -> list[Any]:
    schemas = environment.tools() if callable(environment.tools) else environment.tools
    if not isinstance(schemas, list):
        raise TypeError("Environment tools must be a list of function schemas")
    agents_tools = importlib.import_module("Agents.tools")
    _assert_pinned_module(agents_tools)
    tools: list[Any] = []
    seen: set[str] = set()
    for schema in schemas:
        if not isinstance(schema, Mapping):
            raise TypeError("Environment tool schemas must be objects")
        name, description, inputs, validation_schema = _normalize_schema(schema)
        if name == "final_answer":
            raise ValueError("Environment cannot override the fixed final_answer tool")
        if name in seen:
            raise ValueError(f"Duplicate environment tool: {name}")
        seen.add(name)

        def initialize(
            self: Any,
            *,
            _name: str = name,
            _description: str = description,
            _inputs: dict[str, dict[str, Any]] = inputs,
            _schema: dict[str, Any] = validation_schema,
            _recorder: _ToolRecorder = recorder,
        ) -> None:
            self.name = _name
            self.description = _description
            self.inputs = copy.deepcopy(_inputs)
            self.output_type = "any"
            self.skip_forward_signature_validation = True
            self._recorder = _recorder
            self._validation_schema = copy.deepcopy(_schema)
            agents_tools.Tool.__init__(self)

        def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                if len(args) != 1 or kwargs or not isinstance(args[0], dict):
                    raise TypeError(f"{self.name} expects one JSON argument object")
                kwargs = args[0]
            return self._recorder.invoke(self.name, dict(kwargs), self._validation_schema)

        tool_class = type(
            f"EnvironmentTool_{len(tools)}",
            (agents_tools.Tool,),
            {
                "__module__": __name__,
                "__init__": initialize,
                "forward": forward,
                "skip_forward_signature_validation": True,
            },
        )
        tools.append(tool_class())
    return tools


_STRICT_BENCH_TYPES = frozenset(
    {"toolhop", "api_bank", "restbench", "mixed", "mixed_agent", "mixeddata", "taubench"}
)


def _optional_native_tool(module_name: str, class_name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate an optional pinned helper without inventing a local substitute."""

    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None
    _assert_pinned_module(module)
    tool_class = getattr(module, class_name, None)
    if tool_class is None:
        return None
    return tool_class(*args, **kwargs)


def _native_action_helpers(model: _ModelAdapter) -> dict[str, Any]:
    """Mirror the complete native helper surface built by pinned ``CoreAgent``."""

    agents_tools = importlib.import_module("Agents.tools")
    _assert_pinned_module(agents_tools)
    helpers = {
        "web_tool": _optional_native_tool("module_action.search_tools", "WebSearchTool"),
        "crawl_tool": _optional_native_tool(
            "module_action.search_tools", "CrawlPageTool", model=model
        ),
        "vector_tool": agents_tools.VectorSimilarityRetrieve(memory=None, model=model),
        "reasoning_tool": agents_tools.Reasoning(model=model),
        "process_tool": agents_tools.Process(agent=None),
        "end_process_tool": agents_tools.EndProcess(agent=None),
        "delete_memory_tool": agents_tools.DeleteMemory(agent=None),
        "expert_parallel_tool": _optional_native_tool(
            "module_action.cosight_tool", "ExpertParallelTool", model=model, agents=[]
        ),
        "camv_tool": _optional_native_tool(
            "module_action.cosight_tool", "CAMVTool", model=model
        ),
        "executor_tool": agents_tools.Executor(agent=None),
        "refine_tool": agents_tools.Refine(agent=None),
    }
    return helpers


def _action_context_evidence(context: Any, agent: Any, helpers: Mapping[str, Any]) -> dict[str, Any]:
    raw_agent_tools = getattr(agent, "tools", {})
    if isinstance(raw_agent_tools, Mapping):
        agent_tool_names = sorted(str(name) for name in raw_agent_tools)
    else:
        agent_tool_names = sorted(
            str(getattr(tool, "name"))
            for tool in (raw_agent_tools or [])
            if getattr(tool, "name", None)
        )
    helper_evidence = {}
    for field_name, tool in helpers.items():
        helper_evidence[field_name] = None if tool is None else {
            "name": str(getattr(tool, "name", "")),
            "type": f"{type(tool).__module__}.{type(tool).__name__}",
        }
    bound_fields = {
        field_name: getattr(tool, "agent", None) is not None
        for field_name, tool in helpers.items()
        if field_name in {
            "process_tool",
            "end_process_tool",
            "delete_memory_tool",
            "executor_tool",
            "refine_tool",
        }
        and tool is not None
    }
    bench_tools = list(getattr(context, "bench_tools", None) or [])
    strict = bool(getattr(context, "strict_bench_tools", False))
    return {
        "bench_type": getattr(context, "bench_type", None),
        "strict_bench_tools": strict,
        "helpers": helper_evidence,
        "bound_agent_references": bound_fields,
        "vector_memory_bound": (
            helpers["vector_tool"] is not None
            and getattr(helpers["vector_tool"], "memory", None) is getattr(agent, "memory", None)
        ),
        "reasoning_permitted": bool(helpers["reasoning_tool"] is not None and not (strict and bench_tools)),
        "agent_tools": agent_tool_names,
    }

def _candidate_memory_system(builder: Any) -> str | None:
    value = getattr(builder, "DEFAULT_MEMORY_SYSTEM", None)
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "off", "disable", "disabled", "null"}:
        return None
    return normalized


def _memory_config(
    model: _ModelAdapter, storage_root: Path, memory_system: str
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    normalized_system = memory_system.strip().lower()
    provider_root = (storage_root / normalized_system).resolve()
    try:
        provider_root.relative_to(storage_root.resolve())
    except ValueError as exc:
        raise ValueError("HarnessForge memory system escaped its storage scope") from exc
    try:
        memory_types = importlib.import_module("module_memory.memory_types")
        memory_config = importlib.import_module("module_memory.config")
        # Pinned factory seeds every matching candidate-local provider with the
        # lightweight defaults, including evolved systems with custom names.
        # ``storage_root`` is already scoped by bundle and replica, so it is the
        # equivalent of the factory's applied harness storage namespace.
        config.update(
            memory_config.get_memory_config(memory_types.MemoryType.LIGHTWEIGHT_MEMORY)
        )
    except (AttributeError, ImportError):
        pass
    config.update(
        model=model,
        storage_dir=str(provider_root),
        longterm_memory_path=str(provider_root / "longterm_memory.json"),
        # Storage is scoped by the executor; native retrieval/extraction settings remain intact.
        write_only=False,
    )
    return config


def _build_memory(
    package_name: str, model: _ModelAdapter, storage_root: Path, memory_system: str
) -> Any:
    provider_module = importlib.import_module(f"{package_name}.memory_module.provider")
    declared_system = str(getattr(provider_module, "MEMORY_SYSTEM", "")).strip()
    if not declared_system or declared_system.lower() != memory_system.strip().lower():
        raise ValueError(
            "HarnessForge candidate MemoryProvider does not match DEFAULT_MEMORY_SYSTEM"
        )
    provider_class = getattr(provider_module, "MemoryProvider", None)
    if provider_class is None:
        raise ValueError("HarnessForge memory provider does not export MemoryProvider")
    provider = provider_class(
        config=_memory_config(model, storage_root, memory_system)
    )
    required = ("initialize", "provide_memory", "take_in_memory")
    missing = [name for name in required if not callable(getattr(provider, name, None))]
    if missing:
        raise TypeError(f"HarnessForge MemoryProvider missing interfaces: {missing}")
    if provider.initialize() is False:
        raise RuntimeError("HarnessForge MemoryProvider.initialize() returned False")
    return provider


@contextmanager
def _memory_scope(storage_root: str | Path | None) -> Iterator[Path | None]:
    if storage_root is None:
        yield None
        return
    root = Path(storage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"HarnessForge memory scope cannot be a symlink: {root}")
    lock_path = root / ".rollout.lock"
    with lock_path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield root
        finally:
            if "fcntl" in locals():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _agent_trajectory(agent: Any) -> list[dict[str, Any]]:
    if agent is None:
        return []
    memory_steps = importlib.import_module("Agents.memory")
    _assert_pinned_module(memory_steps)
    trajectory: list[dict[str, Any]] = []
    for step in getattr(getattr(agent, "memory", None), "steps", []) or []:
        if isinstance(step, memory_steps.TaskStep):
            continue
        if isinstance(step, memory_steps.PlanningStep):
            trajectory.append(
                {
                    "name": "plan",
                    "value": step.plan,
                    "think": step.plan_think,
                    "cot_think": step.plan_reasoning,
                    "memory_guidance": getattr(step, "memory_guidance", None),
                }
            )
        elif isinstance(step, memory_steps.SummaryStep):
            trajectory.append(
                {
                    "name": "summary",
                    "value": step.summary,
                    "cot_think": step.summary_reasoning,
                }
            )
        elif isinstance(step, memory_steps.ActionStep):
            trajectory.append(
                {
                    "name": "action",
                    "tool_calls": [
                        tool_call.dict() for tool_call in (step.tool_calls or [])
                    ],
                    "obs": step.observations,
                    "think": step.action_think,
                    "cot_think": step.action_reasoning,
                    "memory_guidance": getattr(step, "memory_guidance", None),
                    "subagent_trajectories": getattr(step, "subagent_trajectories", None),
                }
            )
    return trajectory


def _ingest_memory(
    provider: Any,
    agent: Any,
    *,
    task_prompt: str,
    final_answer: Any,
    evaluation: Any,
    execution_error: str | None,
    seed: int,
    storage_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectory = _agent_trajectory(agent)
    verification = getattr(evaluation, "verification", {}) or {}
    infrastructure_error = bool(getattr(evaluation, "infrastructure_error", False))
    is_correct = verification.get("success") is True and not infrastructure_error
    receipt: dict[str, Any] = {
        "status": "started",
        "storage_root": str(storage_root),
        "trajectory_steps": len(trajectory),
        "is_correct": is_correct,
        "serialization": "exclusive_replica_scope",
    }
    try:
        harness_runtime = importlib.import_module("harness_runtime")
        memory_types = harness_runtime.get_memory_types_module(memory_provider=provider)
        trajectory_data = memory_types.TrajectoryData(
            query=task_prompt,
            trajectory=trajectory,
            result=final_answer,
            metadata={
                "item_index": seed,
                "status": (
                    "success"
                    if execution_error is None and evaluation is not None and not infrastructure_error
                    else "error"
                ),
                "is_correct": is_correct,
                "task_success": is_correct,
                "full_query": task_prompt,
                "reward": getattr(evaluation, "reward", None),
            },
        )
        success, message = provider.take_in_memory(trajectory_data)
        receipt.update(
            status="completed" if success else "rejected",
            success=bool(success),
            message=str(message),
        )
        errors = [] if success else [{"operation": "memory_ingest", "error": str(message)}]
        return receipt, errors
    except Exception as exc:
        receipt.update(status="failed", success=False, error_type=type(exc).__name__)
        return receipt, [{"operation": "memory_ingest", "error_type": type(exc).__name__}]


def _classify_error(exc: BaseException) -> tuple[str, str, bool]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    names = {type(item).__name__ for item in chain}
    if "HarnessForgeBudgetExceeded" in names or "_BudgetExhausted" in names:
        return "budget_exhausted", str(exc), False
    if "TaskInfrastructureError" in names:
        return "infrastructure", str(exc), True
    if "TaskContextBudgetExceeded" in names:
        return "context_budget_exhausted", str(exc), False
    if any("JSON" in name or "Generation" in name for name in names):
        return "parse_error", str(exc), False
    return "candidate_error", f"{type(exc).__name__}: {exc}", False


def _conversations(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for call in calls:
        if call.get("status") != "completed" or not isinstance(call.get("assistant"), dict):
            continue
        result.append(
            {
                "call_id": call["call_id"],
                "operation": call["operation"],
                "messages": [*copy.deepcopy(call["messages"]), copy.deepcopy(call["assistant"])],
                "sft_supervision": "final_assistant",
            }
        )
    return result


def run_harnessforge(
    manifest: HarnessBundleManifest,
    model_callable: Any,
    environment: Any,
    task_prompt: str,
    artifacts_text: str = "",
    seed: int = 42,
    *,
    artifact_sources: list[dict[str, Any]] | None = None,
    memory_storage_root: str | Path | None = None,
    bench_type: str | None = None,
    max_model_calls: int = 128,
    max_tool_calls: int = 128,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_steps: int = 40,
) -> dict[str, Any]:
    """Run one Task rollout through a materialized native HarnessForge agent."""

    if not isinstance(manifest, HarnessBundleManifest):
        raise TypeError("run_harnessforge requires a validated HarnessBundleManifest")
    for name, value in {
        "max_model_calls": max_model_calls,
        "max_tool_calls": max_tool_calls,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(task_prompt, str) or not isinstance(artifacts_text, str):
        raise TypeError("HarnessForge task and artifact prompts must be strings")
    if bench_type is not None and (not isinstance(bench_type, str) or not bench_type.strip()):
        raise TypeError("bench_type must be None or a non-empty string")

    model = _ModelAdapter(
        model_callable,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        max_model_calls=max_model_calls,
    )
    recorder = _ToolRecorder(environment, max_tool_calls)
    final_answer: Any = None
    error_type: str | None = None
    error: str | None = None
    infrastructure_failure = False
    agent: Any = None
    evaluation: Any = None
    memory_receipt: dict[str, Any] | None = None
    memory_errors: list[dict[str, Any]] = []
    inference_call_count = 0
    native_module_paths: dict[str, str] = {}
    agent_contract: dict[str, str] = {}
    action_context: dict[str, Any] = {}
    memory_system: str | None = None

    try:
        with _memory_scope(memory_storage_root) as persistent_memory_root:
            with _IMPORT_LOCK:
                with _activated_candidate(manifest) as (candidate_root, package_name):
                    memory: Any = None
                    runtime_memory_root = persistent_memory_root or (candidate_root / ".runtime_memory")
                    prompt = task_prompt
                    if artifacts_text:
                        prompt += "\n\nCurrent unverified reusable artifacts:\n" + artifacts_text
                    try:
                        builder = importlib.import_module(f"{package_name}.builder")
                        build_agent = getattr(builder, "build_agent_from_context", None)
                        if not callable(build_agent):
                            raise ValueError(
                                "HarnessForge builder.py does not export build_agent_from_context"
                            )

                        action_base = importlib.import_module("module_action.base_action")
                        agents = importlib.import_module("Agents.agents")
                        harness_runtime = importlib.import_module("harness_runtime")
                        memory_system = _candidate_memory_system(builder)
                        if memory_system is not None:
                            memory = _build_memory(package_name, model, runtime_memory_root, memory_system)
                        native_modules = (
                            action_base,
                            importlib.import_module("module_planning.base_planning"),
                            importlib.import_module("module_memory.base_memory"),
                            agents,
                            harness_runtime,
                        )
                        for native_module in native_modules:
                            _assert_pinned_module(native_module)
                        for shared_module in native_modules[:3]:
                            _assert_factory_shared_module(shared_module)
                        native_module_paths = {
                            module.__name__: str(Path(module.__file__).resolve())
                            for module in native_modules
                        }
                        bench_tools = _environment_tools(environment, recorder)
                        helper_tools = _native_action_helpers(model)
                        normalized_bench_type = str(bench_type or "").strip().lower()
                        strict_bench_tools = normalized_bench_type in _STRICT_BENCH_TYPES
                        context = action_base.ActionContext(
                            model=model,
                            summary_interval=5,
                            prompts_type=None,
                            max_steps=max_steps,
                            planning_system=str(
                                getattr(builder, "PLANNING_SYSTEM", "generic_planning")
                            ),
                            action_system=str(getattr(builder, "ACTION_SYSTEM", "single_react")),
                            memory_provider=memory,
                            project_root=candidate_root,
                            bench_type=bench_type,
                            kwargs={},
                            bench_tools=bench_tools,
                            strict_bench_tools=strict_bench_tools,
                            **helper_tools,
                        )
                        agent = build_agent(context)
                        if agent is None:
                            raise TypeError("HarnessForge build_agent_from_context returned None")
                        missing_attrs = [
                            name
                            for name in ("planning_system", "action_system", "harness_name")
                            if not hasattr(agent, name)
                        ]
                        if missing_attrs:
                            raise TypeError(
                                f"HarnessForge agent missing expected attrs: {missing_attrs}"
                            )
                        agent_contract = {
                            "type": f"{type(agent).__module__}.{type(agent).__name__}",
                            **{name: str(getattr(agent, name)) for name in
                               ("planning_system", "action_system", "harness_name")},
                        }
                        action_context = _action_context_evidence(context, agent, helper_tools)
                        final_answer = agent.run(prompt)
                    except Exception as exc:
                        error_type, error, infrastructure_failure = _classify_error(exc)

                    inference_call_count = len(model.calls)
                    failed_calls = [
                        call
                        for call in model.calls[:inference_call_count]
                        if call.get("status") == "failed"
                    ]
                    transport_failure = next(
                        (call for call in failed_calls if not call.get("response_received")), None
                    )
                    if transport_failure is not None:
                        error_type = "infrastructure"
                        error = str(
                            transport_failure.get("error_type") or "Task model transport failed"
                        )
                        infrastructure_failure = True
                    if model.context_budget_exhausted and not infrastructure_failure:
                        error_type = 'context_budget_exhausted'
                        error = 'Task exceeded the fixed context budget; rejected before generation'
                    if (model.budget_exhausted or recorder.budget_exhausted) and error_type is None:
                        error_type = "budget_exhausted"
                        error = "Maximum Task model/tool calls reached"
                    failed_tools = [
                        call for call in recorder.calls if call.get("status") == "failed"
                    ]
                    if failed_tools:
                        error_type = "infrastructure"
                        error = str(
                            failed_tools[0].get("error_type")
                            or "Task environment operation failed"
                        )
                        infrastructure_failure = True

                    if final_answer is None:
                        final_answer = ""
                    elif isinstance(final_answer, str):
                        # AgentText belongs to a temporary, dynamically loaded module.
                        final_answer = str(final_answer)
                    else:
                        final_answer = json.dumps(final_answer, ensure_ascii=False)
                    if not infrastructure_failure:
                        try:
                            evaluation = environment.evaluate(final_answer)
                        except Exception as exc:
                            error_type = "infrastructure"
                            error = f"{type(exc).__name__}: {exc}"
                            infrastructure_failure = True
                        else:
                            if bool(getattr(evaluation, "infrastructure_error", False)):
                                error_type = "infrastructure"
                                error = str(
                                    getattr(evaluation, "error_type", None)
                                    or "Task evaluator infrastructure failure"
                                )
                                infrastructure_failure = True

                    pending_official = (getattr(evaluation, "verification", {}) or {}).get("status") == "pending_official"
                    if error_type == "context_budget_exhausted" and not infrastructure_failure and not pending_official:
                        from dataclasses import replace
                        # A task exceeding its fixed budget is a completed failure,
                        # never a reason to remove its denominator or train on it.
                        metrics = dict(evaluation.metrics)
                        for key in ("exact_match", "f1", "pass", "task_success"):
                            if key in metrics:
                                metrics[key] = 0.0
                        evaluation = replace(evaluation, reward=0.0, metrics=metrics,
                            verification={**evaluation.verification, "status": "completed",
                                          "success": False, "termination_reason": error_type},
                            error_type=error_type,
                            details={**evaluation.details, "task_budget_exhausted": True})

                    if memory is not None and pending_official:
                        # Official scoring has not happened: never infer a success/
                        # failure label or learn result-based memory from this turn.
                        memory_receipt = {
                            "status": "skipped",
                            "reason": "pending_official_evaluation",
                        }
                    if memory is not None and not pending_official:
                        budget_before_ingest = model.budget_exhausted
                        calls_before_ingest = len(model.calls)
                        memory_receipt, memory_errors = _ingest_memory(
                            memory,
                            agent,
                            task_prompt=prompt,
                            final_answer=final_answer,
                            evaluation=evaluation,
                            execution_error=error_type,
                            seed=seed,
                            storage_root=runtime_memory_root,
                        )
                        post_calls = model.calls[calls_before_ingest:]
                        for call in post_calls:
                            call["post_evaluation"] = True
                        if (
                            (model.budget_exhausted and not budget_before_ingest)
                            or any(call.get("status") == "failed" for call in post_calls)
                        ):
                            memory_receipt["status"] = "degraded"
                            memory_errors.append(
                                {
                                    "operation": "memory_ingest",
                                    "error_type": "model_call_failed_or_budget_exhausted",
                                }
                            )
    except Exception as exc:
        if error_type is None:
            error_type, error, infrastructure_failure = _classify_error(exc)

    conversations = _conversations(model.calls[:inference_call_count])
    artifact_chars = len(artifacts_text)
    result = {
        "messages": conversations[-1]["messages"] if conversations else [],
        "model_calls": model.calls,
        "sft_conversations": conversations,
        "tool_calls": recorder.calls,
        "final_answer": final_answer,
        "notes": [],
        "error_type": error_type,
        "error": error,
        "memory_errors": memory_errors,
        "memory_receipt": memory_receipt,
        "bundle_sha256": manifest.bundle_sha256,
        "seed_sha256": manifest.bundle_sha256,
        "effective_harness_sha256": manifest.bundle_sha256,
        "native_module_paths": native_module_paths,
        "agent_contract": agent_contract,
        "action_context": action_context,
        "memory_system": memory_system,
        "steps": int(getattr(agent, "step_number", 0) or 0),
        "artifact_chars_used": artifact_chars,
        "artifact_chars_omitted": 0,
        "artifact_sources_used": len(artifact_sources or []),
        "infrastructure_failure": infrastructure_failure,
    }
    # Match persisted JSON evidence before crossing a process boundary. Native
    # subclasses must not retain imports from an unloaded candidate. No default=str:
    # unsupported evidence stays an explicit serialization error. The evaluator
    # result is a host-owned object, not candidate evidence.
    result = json.loads(json.dumps(result, ensure_ascii=False))
    result["_evaluation"] = evaluation
    return result


__all__ = ["HarnessForgeBudgetExceeded", "run_harnessforge"]

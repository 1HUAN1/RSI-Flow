"""Execute a validated Task H graph against one caller-bound Task model.

The graph can select context, arrange declared primitives and request bounded
recovery. It never supplies a model endpoint, filesystem operation or evaluator.
All state is rollout-local; generated notes are returned to the controller.
"""
from __future__ import annotations

import copy
import json
import time

from sia.task_meta.meta_harness.runtime import condition
from sia.task_meta.seed import _BudgetExhausted, _hash, _message, _render, _tool_specs, _transport

from .policy import MAX_TRANSITIONS


def run_harness(spec, model_callable, environment, task_prompt, artifacts_text="", seed=42, *, artifact_sources=None):
    """Run one finite graph; reset, trusted evaluation and asset commit stay outside."""
    from . import legacy_view, validate_harness

    spec = validate_harness(copy.deepcopy(spec))
    legacy = legacy_view(spec)
    parts = spec["parts"]
    prompts, config, budget = legacy["prompts"], legacy["context"], spec["budget"]
    tools = _tool_specs(environment)
    tool_names = {tool["name"] for tool in tools}
    raw_tools = environment.tools() if callable(environment.tools) else environment.tools
    tool_schemas = {tool.get("function", tool)["name"]: tool.get("function", tool).get("parameters", {})
                    for tool in raw_tools}
    roles = {role["name"]: role for role in parts["tools"]["roles"]}
    calls, tool_calls, memory_errors, shortterm, step_summaries, events = [], [], [], [], [], []
    memory = [_message("system", parts["input"]["action_system"]),
              _message("user", _render(parts["input"]["task_template"], task=task_prompt))]
    final_answer, candidate, error_type, error = None, None, None, None
    last_context, memory_counter, last_provision = "", 1, -999
    step_number, transitions, active_node = 0, 0, None
    guidance, requested, observations, remembered_calls = None, [], [], []
    needs_repair, review, tool_success, evidence_count = False, "", True, 0
    last_tool_error, last_check_passed, last_role, tool_success_count = "", None, None, 0
    terminal = False
    terminal_nodes = set()

    def scope(item=None):
        return {"state": {"has_final": final_answer is not None, "has_candidate": candidate is not None,
                          "can_continue": step_number <= legacy["planning"]["max_steps"],
                          "needs_repair": needs_repair, "tool_success": tool_success,
                          "evidence_count": evidence_count, "step": step_number, "candidate": candidate,
                          "step_number": step_number, "last_tool_error": last_tool_error,
                          "tool_success_count": tool_success_count, "memory_count": len(shortterm),
                          "last_check_passed": last_check_passed, "last_role": last_role,
                          "ready_to_submit": candidate is not None and not needs_repair,
                          "remaining_model_calls": budget["max_model_calls"] - len(calls),
                          "remaining_tool_calls": budget["max_tool_calls"] - len(tool_calls)}, "item": item}

    def event(kind, **values):
        events.append({"event_id": len(events), "kind": kind, "node": active_node,
                       "step": step_number, **copy.deepcopy(values)})

    # The caller provides a complete read-only A_in. Selection precedes the
    # prompt character budget; no generated note becomes another rollout's input.
    sources = copy.deepcopy(artifact_sources) if artifact_sources is not None else (
        [{"path": "inline_artifacts", "content": artifacts_text, "sha256": _hash(artifacts_text)}]
        if artifacts_text else [])
    if not isinstance(sources, list) or any(not isinstance(row, dict) or not isinstance(row.get("content"), str)
                                            or not isinstance(row.get("path"), str) for row in sources):
        raise ValueError("Artifact sources require controller-owned path/content records")
    asset_policy = parts["memory"]["assets"]
    selected = [row for row in sources if condition(asset_policy["filter"], scope(row))][:asset_policy["max_items"]]
    selected_text = "\n".join(row["content"] for row in selected)
    available_chars = len("\n".join(row["content"] for row in sources))
    artifact_chars = min(len(selected_text), config["artifact_char_limit"]) if config["use_artifacts"] else 0
    if artifact_chars:
        memory.append(_message("user", "Current unverified reusable artifacts:\n" + selected_text[:artifact_chars]))
    event("asset_selection", selected=[{key: row.get(key) for key in ("path", "sha256")} for row in selected],
          input_count=len(sources), selected_count=len(selected), chars_used=artifact_chars,
          chars_omitted=max(0, available_chars - artifact_chars))

    def history_view():
        policy = parts["memory"]["history"]
        required = {index for index, message in enumerate(memory) if message["role"] in policy["required_roles"]}
        eligible = {index for index, message in enumerate(memory)
                    if condition(policy["filter"], scope({**message, "index": index}))} | required
        kept = set(required)
        chars = sum(len(str(memory[index].get("content", ""))) for index in kept)
        max_messages = policy["max_messages"] if policy["max_messages"] is not None else len(memory)
        max_chars = policy["max_chars"] if policy["max_chars"] is not None else sum(len(str(row.get("content", ""))) for row in memory)
        if len(kept) > max_messages or chars > max_chars:
            raise _BudgetExhausted("Required Task context exceeds the declared history budget")
        for index in sorted(eligible - required, reverse=True):
            length = len(str(memory[index].get("content", "")))
            if len(kept) < max_messages and chars + length <= max_chars:
                kept.add(index)
                chars += length
        indices = sorted(kept)
        event("history_selection", selected_indices=indices,
              omitted_indices=[index for index in range(len(memory)) if index not in kept], chars=chars)
        return [copy.deepcopy(memory[index]) for index in indices]

    def complete(operation, messages, *, role=None):
        if len(calls) >= budget["max_model_calls"]:
            raise _BudgetExhausted("Maximum Task model calls reached")
        actual, request_seed = _transport(messages), seed + len(calls)
        record = {"call_id": len(calls), "operation": operation, "role": role, "node": active_node,
                  "messages": copy.deepcopy(actual), "tools": None, "seed": request_seed,
                  "max_tokens": budget["max_tokens"], "temperature": budget["temperature"], "status": "started"}
        calls.append(record)
        started = time.monotonic()
        event("model_request", call_id=record["call_id"], operation=operation, role=role)
        try:
            response = model_callable(actual, tools=None, seed=request_seed, max_tokens=budget["max_tokens"],
                                      temperature=budget["temperature"])
            assistant = response.get("message", response.get("assistant", response))
            if not isinstance(assistant, dict) or assistant.get("role", "assistant") != "assistant":
                raise ValueError("Task model returned no assistant message")
            assistant = {key: copy.deepcopy(value) for key, value in assistant.items()
                         if key in {"role", "content", "tool_calls", "reasoning_content"}}
            assistant.setdefault("role", "assistant")
            record.update({"assistant": assistant, "usage": response.get("usage"),
                           "binding": response.get("binding"), "status": "completed"})
            usage = response.get("usage") or {}
            record["usage_complete"] = all(type(usage.get(key)) is int and usage[key] >= 0
                                           for key in ("prompt_tokens", "completion_tokens"))
            record["unknown_usage_calls"] = int(not record["usage_complete"])
            for key in ("chat_template_kwargs", "finish_reason", "request_id", "model_ref_requested", "model_ref_response"):
                if key in response:
                    record[key] = copy.deepcopy(response[key])
            event("model_response", call_id=record["call_id"], status="completed")
            return assistant
        except _BudgetExhausted:
            # The controller rejected this attempt before dispatch or a receipt.
            calls.pop()
            events[-1].update(kind="model_budget_exhausted", dispatched=False)
            raise
        except Exception as exc:
            record.update({"status": "failed", "error_type": type(exc).__name__,
                           "usage_complete": False, "unknown_usage_calls": 1})
            event("model_response", call_id=record["call_id"], status="failed", error_type=type(exc).__name__)
            raise RuntimeError("Task model operation failed") from exc
        finally:
            record["wall_time_seconds"] = time.monotonic() - started

    def previous_summary():
        return "\n".join(f"Step {step}: {summary}" for step, summary in step_summaries) or "No previous steps (this is the first step)."

    def extract_memory():
        nonlocal last_context, memory_counter, last_provision
        if not config["shortterm_enabled"]:
            return None
        memory_counter += 1
        enum_roles = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT", "tool-response": "TOOL_RESPONSE"}
        context = "\n".join("MessageRole." + enum_roles[item["role"]] + ": " + item["content"] for item in history_view())
        delta = context.strip()
        if last_context and delta.startswith(last_context.strip()):
            delta = delta[len(last_context.strip()):].strip() or context.strip()
        last_context = context
        if len(delta.strip()) >= 50:
            prompt = _render(prompts["memory_extract"], task=task_prompt, previous_steps=previous_summary(),
                             current_memory="\n".join("- " + item for item in shortterm) or "(No memory items yet)", context_delta=delta)
            response = complete("memory_extract", [_message("user", prompt)])
            try:
                extracted = json.loads(response.get("content", ""))
                summary = extracted.get("step_summary", "").strip()
                if summary:
                    step_summaries.append((memory_counter, summary))
                for value in extracted.get("key_extracts", []):
                    item = str(value).strip()
                    if len(item) <= 10 or item in shortterm:
                        continue
                    shortterm.append(item)
                    if len(shortterm) > config["max_shortterm_items"]:
                        prune = _render(prompts["memory_prune"], task=task_prompt, previous_steps=previous_summary(),
                                        memory_items="\n".join(f"{i}. {text}" for i, text in enumerate(shortterm, 1)),
                                        max_shortterm_items=config["max_shortterm_items"])
                        selected_memory = complete("memory_prune", [_message("user", prune)])
                        try:
                            indices = json.loads(selected_memory.get("content", ""))
                            if not isinstance(indices, list) or not indices:
                                raise ValueError("No memory indices")
                            keep = {index - 1 for index in indices if type(index) is int and 1 <= index <= len(shortterm)}
                            shortterm[:] = [text for index, text in enumerate(shortterm) if index in keep][:config["max_shortterm_items"]]
                        except (ValueError, TypeError):
                            shortterm[:] = shortterm[-config["max_shortterm_items"]:]
            except (ValueError, TypeError, AttributeError) as exc:
                memory_errors.append({"call_id": calls[-1]["call_id"], "error_type": type(exc).__name__})
        if memory_counter - last_provision >= config["shortterm_interval"] and shortterm:
            last_provision = memory_counter
            content = "**Key Information & Constraints:**\n" + "\n".join(f"{i}. {item}" for i, item in enumerate(shortterm, 1))
            return "---Memory System Guidance---\n" + content + "\n---End Memory---"
        return None

    def parse_action(assistant):
        value = json.loads(assistant.get("content", ""))
        if not isinstance(value, dict):
            raise ValueError("Action response must be a JSON object")
        action_calls = value.get("tools")
        if action_calls is None and isinstance(value.get("answer"), str):
            action_calls = [{"name": "final_answer", "arguments": {"answer": value["answer"]}}]
        if not isinstance(action_calls, list) or not action_calls:
            raise ValueError("Action response requires a nonempty tools list")
        for call in action_calls:
            if (not isinstance(call, dict) or call.get("name") not in tool_names
                    or not isinstance(call.get("arguments"), dict)):
                raise ValueError("Action contains unknown tools or non-object arguments")
        return action_calls[:parts["tools"]["action"]["max_tools_per_step"]]

    def role_call(name, *, operation="role"):
        nonlocal last_role
        role = roles[name]
        last_role = name
        instruction = _render(role["instruction"], task=task_prompt, candidate=candidate,
                              observations="\n\n".join(observations), review=review, step=step_number,
                              error=last_tool_error, schema=json.dumps(tools, ensure_ascii=False),
                              history=json.dumps(history_view(), ensure_ascii=False))
        request = [*history_view(), _message("user", instruction)]
        if operation == "submission_review":
            request.append(_message("user", json.dumps({"candidate": candidate, "observations": observations,
                "required_response": {"needs_repair": "boolean", "feedback": "optional string"}}, ensure_ascii=False)))
        return complete(operation + ":" + name, request, role=name)

    def inspect_candidate():
        nonlocal final_answer, needs_repair, review, last_check_passed
        if candidate is None:
            return
        answer_format = parts["submission"]["answer_format"]
        rendered_candidate = candidate
        if answer_format["mode"] == "json_key":
            parsed = json.loads(candidate) if isinstance(candidate, str) else candidate
            if not isinstance(parsed, dict) or answer_format["key"] not in parsed:
                raise ValueError("Final answer does not contain the configured JSON key")
            rendered_candidate = parsed[answer_format["key"]]
        elif answer_format["mode"] == "last_line" and isinstance(candidate, str):
            rendered_candidate = candidate.splitlines()[-1] if candidate.splitlines() else ""
        failures = []
        for check in parts["submission"]["checks"]:
            kind = check["kind"]
            if kind == "nonempty":
                passed = rendered_candidate is not None and bool(str(rendered_candidate).strip())
            elif kind == "tool_success":
                passed = tool_success and tool_success_count >= check.get("minimum", 1)
            elif kind == "evidence_count":
                passed = evidence_count >= check.get("minimum", 1)
            elif kind == "model_review":
                response = role_call(check["role"], operation="submission_review")
                value = json.loads(response.get("content", ""))
                if not isinstance(value, dict) or type(value.get("needs_repair")) is not bool:
                    raise ValueError("Submission review must return JSON with boolean needs_repair")
                passed = not value["needs_repair"]
                review = str(value.get("feedback", ""))
            else:
                raise ValueError("Unsupported submission check")
            event("submission_check", name=check["name"], check=kind, passed=passed,
                  evidence_source="public_task_state_only")
            if not passed:
                failures.append(check["name"])
        needs_repair = bool(failures)
        last_check_passed = not needs_repair
        if needs_repair:
            review = "Checks requiring repair: " + ", ".join(failures) + ("\n" + review if review else "")
            memory.append(_message("user", _render(parts["submission"]["repair_prompt"],
                task=task_prompt, candidate=candidate, review=review, observations="\n\n".join(observations), step=step_number,
                error=last_tool_error, schema=json.dumps(tools, ensure_ascii=False),
                history=json.dumps(history_view(), ensure_ascii=False))))
            # Always preserve actual failed checks even if an evolved prompt
            # omits its review placeholder.
            memory.append(_message("user", review))
        else:
            final_answer = rendered_candidate

    def execute_tools():
        nonlocal candidate, terminal, requested, tool_success, evidence_count, last_tool_error, tool_success_count
        if terminal:
            raise ValueError("Cannot execute tools after the environment terminated")
        pending, requested = requested, []
        if guidance:
            memory.append(_message("user", guidance))
        for call in pending:
            name, arguments = call["name"], copy.deepcopy(call["arguments"])
            remembered_calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
            if name == "final_answer":
                candidate = arguments.get("answer", arguments)
                observations.append(str(candidate))
                break
            for attempt in range(parts["tools"]["recovery"]["max_retries"] + 1):
                if len(tool_calls) >= budget["max_tool_calls"]:
                    raise _BudgetExhausted("Maximum Task tool calls reached")
                entry = {"name": name, "arguments": copy.deepcopy(arguments), "step": step_number,
                         "attempt": attempt, "node": active_node}
                tool_calls.append(entry)
                task_error = None
                try:
                    if parts["tools"]["validate_arguments"]:
                        import jsonschema
                        try:
                            jsonschema.validate(arguments, tool_schemas[name])
                        except jsonschema.ValidationError as exc:
                            raise ValueError("Tool argument validation failed: " + exc.message) from exc
                    observation = environment.step(name, arguments)
                    entry.update({"observation": copy.deepcopy(observation), "status": "completed"})
                    observations.append(f"Results for tool call '{name}' with arguments '{arguments}':\n{str(observation).strip()}")
                    # Only actual returned retrieval records are evidence. A
                    # successful calculator or a final answer is not evidence.
                    if isinstance(observation, dict):
                        if observation.get("error") or observation.get("runtime_error") is True:
                            task_error = str(observation.get("error") or observation.get("observation") or "tool_runtime_error")
                            entry["task_tool_failure"] = True
                        if task_error is None:
                            records = observation.get("results", observation.get("evidence", []))
                            if isinstance(records, list):
                                evidence_count += len(records)
                        if observation.get("terminal") is True or observation.get("terminated") is True:
                            candidate = observation.get("final_answer", observation.get("observation", ""))
                            terminal = True
                            event("environment_terminal", tool=name, observation=observation)
                    if task_error is None:
                        tool_success_count += 1
                        break
                except (ValueError, TypeError) as exc:
                    entry.update({"status": "task_error", "error_type": type(exc).__name__, "error": str(exc)})
                    task_error = str(exc)
                    observations.append(f"Error for tool call '{name}' with arguments '{arguments}':\n{exc}")
                last_tool_error = task_error
                event("tool_error", tool=name, arguments=arguments, error=task_error, attempt=attempt)
                if terminal or attempt == parts["tools"]["recovery"]["max_retries"]:
                    tool_success = False
                    break
                prompt = _render(parts["tools"]["recovery"]["prompt"], task=task_prompt,
                                 error=task_error, observations="\n\n".join(observations), step=step_number,
                                 candidate=candidate, review=review, schema=json.dumps(tool_schemas[name]),
                                 history=json.dumps(history_view(), ensure_ascii=False))
                prompt += "\n" + json.dumps({"tool": name, "arguments": arguments, "error": task_error,
                    "required_response": {"arguments": "object; only repair this tool's arguments"}}, ensure_ascii=False)
                response = complete("tool_repair", [*history_view(), _message("user", prompt)])
                repair = json.loads(response.get("content", ""))
                if not isinstance(repair, dict) or set(repair) != {"arguments"} or not isinstance(repair["arguments"], dict):
                    raise ValueError("Tool recovery must only return the same tool's repaired arguments")
                arguments = repair["arguments"]
                remembered_calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
            if terminal:
                break

    graph = parts["control"]["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    active_node = graph["entry"]
    try:
        while True:
            if transitions >= MAX_TRANSITIONS:
                raise _BudgetExhausted("Maximum Task graph transitions reached")
            transitions += 1
            node = nodes[active_node]
            kind = node["kind"]
            if terminal:
                if kind not in {"inspect", "commit", "branch", "stop"} or kind in terminal_nodes:
                    event("terminal_guard", blocked_primitive=kind)
                    if final_answer is None:
                        error_type, error = "submission_rejected", "Terminated environment cannot be retried"
                    break
                terminal_nodes.add(kind)
            enabled = condition(node.get("when", True), scope())
            event("transition", primitive=kind, enabled=enabled)
            if enabled:
                if kind == "stop":
                    break
                if final_answer is not None and kind not in {"commit", "branch"}:
                    # A graph cannot obtain a second submission or extra calls
                    # after an accepted final answer.
                    break
                if kind == "prepare":
                    if step_number == 0 and legacy["planning"]["enabled"]:
                        plan = complete("planning", [_message("system", prompts["planning_initial"]),
                            _message("user", _render(prompts["planning_task"], task=task_prompt))])
                        memory.extend([_message("user", "Now, begin your planning analysis for this task!"),
                                       _message("assistant", "[PLAN]:\n" + plan.get("content", "").strip())])
                        step_number += 1
                    elif step_number and step_number % legacy["planning"]["summary_interval"] == 0:
                        summary = complete("summary", [_message("system", _render(prompts["summary_pre"], task=task_prompt, step=step_number)),
                            *[item for item in history_view() if item["role"] != "system"],
                            _message("user", _render(prompts["summary_post"], task=task_prompt, step=step_number))])
                        memory.extend([_message("user", "Now, summarize and analysis the task completion status and provide recommendations for next steps!"),
                                       _message("assistant", "[SUMMARY]:\n" + summary.get("content", "").strip())])
                        step_number += 1
                elif kind == "memory":
                    guidance = extract_memory()
                elif kind == "act":
                    instruction = _render(parts["input"]["action_step"], task=task_prompt,
                                          tool_functions_json=json.dumps(tools, ensure_ascii=False, indent=2))
                    sections = {"history": history_view(), "guidance": [_message("user", guidance)] if guidance else [],
                                "action": [_message("user", instruction)]}
                    request = [message for section in parts["input"]["section_order"] for message in sections[section]]
                    retries = parts["tools"]["action"]["parse_retries"]
                    for attempt in range(retries + 1):
                        assistant = complete("action" if attempt == 0 else "action_repair", request)
                        try:
                            requested = parse_action(assistant)
                            break
                        except (ValueError, TypeError) as exc:
                            if attempt == retries:
                                raise ValueError("Unparseable Task action after the declared repair budget") from exc
                            request = [*request, assistant, _message("user", "Repair your action: " + str(exc) + ". Return strict JSON using only the listed tools.")]
                    observations, remembered_calls, candidate = [], [], None
                    tool_success = True
                elif kind == "tools":
                    execute_tools()
                elif kind == "inspect":
                    inspect_candidate()
                elif kind == "commit":
                    memory.extend([_message("assistant", "Calling tools:\n" + str({"tools": remembered_calls})),
                                   _message("tool-response", "Tool calling observation:\n" + "\n\n".join(observations))])
                    step_number += 1
                elif kind == "finalize":
                    assistant = complete("budget_finalization", [_message("system", prompts["final_pre"]),
                                         *[item for item in history_view() if item["role"] != "system"],
                                         _message("user", _render(prompts["final_post"], task=task_prompt))])
                    candidate = json.loads(assistant.get("content", "")).get("answer", "")
                    inspect_candidate()
                elif kind == "role":
                    role = roles[node["role"]]
                    assistant = role_call(role["name"])
                    text = assistant.get("content", "").strip()
                    if role["result"] == "memory":
                        if text and text not in shortterm:
                            shortterm.append(text)
                            shortterm[:] = shortterm[-config["max_shortterm_items"]:]
                    elif role["result"] == "review":
                        value = json.loads(text)
                        if not isinstance(value, dict) or type(value.get("needs_repair")) is not bool:
                            raise ValueError("Review role must return JSON with boolean needs_repair")
                        needs_repair, review = value["needs_repair"], str(value.get("feedback", ""))
                        memory.append(_message("assistant", "[REVIEW]:\n" + text))
                    else:
                        memory.append(_message("assistant", "[" + role["result"].upper() + "]:\n" + text))
                elif kind != "branch":
                    raise ValueError("Unsupported Task graph primitive")
            following = node["next"]
            if isinstance(following, list):
                following = next(edge["to"] for edge in following if condition(edge["when"], scope()))
            event("edge", target=following)
            active_node = following
        if final_answer is not None:
            if parts["submission"]["strip_final_answer"] and isinstance(final_answer, str):
                final_answer = final_answer.strip()
            event("submission", answer=final_answer, count=1)
    except _BudgetExhausted as exc:
        error_type, error = "budget_exhausted", str(exc)
    except ValueError as exc:
        final_answer = None
        error_type, error = "parse_error", str(exc)
    except Exception as exc:
        error_type, error = "infrastructure", type(exc).__name__
    successful_calls = [call for call in calls if call["status"] == "completed"]
    conversations = [{**{key: copy.deepcopy(value) for key, value in call.items()
                         if key in {"call_id", "operation", "role", "node", "tools", "binding", "chat_template_kwargs",
                                    "seed", "max_tokens", "temperature", "request_id"}},
                      "messages": [*copy.deepcopy(call["messages"]), copy.deepcopy(call["assistant"])],
                      "sft_supervision": "final_assistant"} for call in successful_calls]
    return {"messages": conversations[-1]["messages"] if conversations else [], "model_calls": calls,
            "sft_conversations": conversations, "tool_calls": tool_calls, "final_answer": final_answer,
            "notes": list(shortterm) if config["generate_artifacts"] else [], "error_type": error_type, "error": error,
            "memory_errors": memory_errors, "seed_sha256": _hash(spec), "steps": step_number,
            "artifact_chars_used": artifact_chars, "artifact_chars_omitted": max(0, available_chars - artifact_chars),
            "final_context": memory, "infrastructure_failure": error_type == "infrastructure", "events": events,
            "harness_identity": {"schema_version": 2, "content_sha256": _hash(spec)},
            "transitions": transitions, "submission_count": sum(event["kind"] == "submission" for event in events)}

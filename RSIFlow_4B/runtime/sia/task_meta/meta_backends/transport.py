"""Trusted credential forwarding, identity checks and API budget enforcement.

Codex produces every model request. This layer never generates a decision or
rewrites tool semantics; unsupported server-history paths fail closed.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import math
import os
import re
import socketserver
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def persist_bytes(path, data):
    """Publish durable evidence atomically, before acknowledging delivery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def completed_response_events(response):
    """Frame an actual complete provider response for the pinned Codex SSE reader.

    No output, identity, usage or completion is inferred from a partial response.
    Added/done items preserve the provider's original content and call IDs.
    """
    if not isinstance(response, dict) or response.get("status") != "completed" or not response.get("id"):
        raise ValueError("Buffered provider response is not explicitly completed")
    if not isinstance(response.get("output"), list) or not response["output"]:
        raise ValueError("Buffered provider response has no output items")
    usage = response.get("usage") or {}
    if type(usage.get("output_tokens")) is not int or usage["output_tokens"] < 0:
        raise ValueError("Buffered provider response has no valid usage")
    # Some gateways encode an optional reasoning status as an empty string.
    # Normalize that metadata only after the whole response is explicitly completed.
    response = dict(response)
    response["output"] = [dict(item, status=None) if isinstance(item, dict) and item.get("type") == "reasoning" and item.get("status") == "" else item for item in response["output"]]
    for item in response["output"]:
        if not isinstance(item, dict) or not item.get("id") or item.get("status") not in {None, "completed"}:
            raise ValueError("Buffered provider output item is incomplete or unbound")
    yield {"type": "response.created", "response": {"id": response["id"], "model": response.get("model")}}
    for index, item in enumerate(response["output"]):
        yield {"type": "response.output_item.added", "output_index": index, "item": item}
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    yield {"type": "response.completed", "response": response}


def buffered_stream_response(response, maximum, deadline, raw_path, record, receive_bytes):
    """Keep native SSE evidence; deliver only the provider's complete response."""
    size = 0
    fields = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    record["upstream_events_file"] = raw_path.name
    counts = record.setdefault("upstream_event_counts", {})
    with raw_path.open("xb", buffering=0) as raw:
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Upstream stream exceeded the existing request deadline")
                line = response.readline(maximum - size + 1)
                if not line:
                    if size == 0:
                        # Delivery ended before any output or tool call reached Codex.
                        # Provider computation/charge is unknown and remains reserved.
                        record['terminal_failure'] = {'code': 'empty_stream_eof', 'usage': None,
                            'provider_completion_unknown': True, 'delivered_bytes': 0}
                        raise TerminalProviderFailure('Provider closed an empty stream; usage unknown')
                    raise ValueError("Upstream stream ended without a complete response")
                raw.write(line)
                size += len(line)
                record["upstream_stream_bytes"] = size
                if size > maximum:
                    raise ValueError("Upstream stream exceeds the resource budget")
                receive_bytes(len(line))
                if line.startswith(b"data:"):
                    fields.append(line[5:].strip())
                if line.strip() or not fields:
                    continue
                data = b"\n".join(fields)
                fields.clear()
                if data == b"[DONE]":
                    raise ValueError("Upstream DONE arrived without a complete response")
                event = json.loads(data)
                kind = event.get("type", "untyped")
                counts[kind] = counts.get(kind, 0) + 1
                failed = event.get("response") or {}
                error = failed.get("error") or {}
                if kind == "response.failed" and failed.get("status") == "failed" and error.get("code") in {"server_error", "upstream_error", "rate_limit_exceeded"}:
                    record["terminal_failure"] = {"code": error["code"], "response_id": failed.get("id"), "usage": failed.get("usage")}
                    raise TerminalProviderFailure("Provider explicitly terminated: " + error["code"])
                if (kind == 'response.failed' and failed.get('status') == 'failed'
                        and error.get('type') == 'api_error' and error.get('code') is None):
                    record['terminal_failure'] = {'code': 'provider_api_error', 'response_id': failed.get('id'),
                        'usage': failed.get('usage'), 'provider_completion_unknown': False}
                    raise TerminalProviderFailure('Provider explicitly terminated: api_error without code')
                top_error = event.get("error") or {}
                if kind == 'error' and event.get('code') in {
                        'do_request_failed', 'server_error', 'upstream_error', 'rate_limit_exceeded', 'stream_read_error'}:
                    record['terminal_failure'] = {'code': event['code'], 'event_type': 'error',
                        'usage': None, 'provider_completion_unknown': True}
                    raise TerminalProviderFailure('Provider explicitly terminated delivery: ' + event['code'])
                if kind == "error" and top_error.get("code") == "server_error" and top_error.get("type") == "server_error":
                    record["terminal_failure"] = {"code": "server_error", "event_type": "error",
                                                  "usage": None, "provider_completion_unknown": True}
                    raise TerminalProviderFailure("Provider explicitly terminated delivery: server_error; usage unknown")
                if kind == "error" and top_error.get("code") == "rate_limit_exceeded" and top_error.get("type") == "too_many_requests":
                    record["terminal_failure"] = {"code": "rate_limit_exceeded", "event_type": "error",
                        "usage": None, "provider_completion_unknown": True}
                    raise TerminalProviderFailure("Provider explicitly terminated delivery: rate_limit_exceeded; usage unknown")
                if kind == "error" and top_error.get("type") == "upstream_error" and top_error.get("code") == "stream_read_error":
                    record["terminal_failure"] = {"code": "stream_read_error", "event_type": "error", "usage": None, "provider_completion_unknown": True}
                    raise TerminalProviderFailure("Provider explicitly terminated delivery: stream_read_error; usage unknown")
                if kind in {"error", "response.failed", "response.incomplete"}:
                    raise ValueError("Upstream reported failed or incomplete response")
                if kind == "response.completed":
                    value = event.get("response")
                    list(completed_response_events(value))
                    return json.dumps(value, ensure_ascii=False).encode()
        finally:
            os.fsync(raw.fileno())


class TerminalProviderFailure(ValueError):
    pass


def valid_attempt_chain(records, accepted_models):
    successes = {r["ordinal"]: r for r in records if r.get("completed") and r.get("returned_model") in accepted_models}
    if not successes:
        return False
    for r in records:
        if r.get("ordinal") in successes:
            continue
        recovered = successes.get(r.get("recovered_by_ordinal"), {})
        if not (r.get("provider_outcome") == "failed" and r.get("terminal_failure")
                and r.get("durable_request_id") and recovered.get("durable_request_id") == r["durable_request_id"]
                and recovered["ordinal"] > r["ordinal"]):
            return False
    return True


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Provider redirects are not permitted")


class ResponsesTransport:
    def __init__(self, config, secret, operation_budget=None, evidence_dir=None):
        self.config = config
        self._secret = secret
        self.requests = []
        self.output_tokens = 0
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.server = None
        self.error = None
        self.operation_budget = operation_budget
        self.completed_item_status = {}
        self.provider_item_aliases = {}
        self.replay_aliases = []
        self.completed_call_ids = set()
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None

    def validate_body(self, body):
        if not isinstance(body, dict) or body.get("model") != self.config.model:
            raise ValueError("Exact registered model is required; no model fallback")
        if body.get("previous_response_id") or body.get("conversation"):
            raise ValueError("Server-side conversation state is unsupported")
        if body.get("store") is True:
            raise ValueError("Responses must use store=false")
        if body.get("background") or body.get("models") or body.get("route"):
            raise ValueError("Background execution/model routing unsupported")
        body = dict(body)
        if self.config.response_delivery == "buffered_json":
            body["stream"] = self.config.provider == "autodl"
        body["store"] = False
        maximum = self.config.budget.max_output_tokens_per_request
        remaining = self.config.budget.max_total_output_tokens - self.output_tokens
        requested = body.get("max_output_tokens", maximum)
        if type(requested) is not int or requested <= 0 or remaining <= 0:
            raise ValueError("Invalid or exhausted output token budget")
        body["max_output_tokens"] = min(requested, maximum, remaining)
        if self.operation_budget:
            body["max_output_tokens"] = self.operation_budget.output_limit(body["max_output_tokens"])
        if self.config.provider == "openrouter":
            policy = {"allow_fallbacks": self.config.allow_provider_fallback}
            if self.config.provider_order:
                policy["order"] = self.config.provider_order
            body["provider"] = policy
            if (self.config.model == "deepseek/deepseek-v4.1-flash"
                    and self.config.g_output_delivery == "candidate_file_all" and body.get("tools")):
                # Do not constrain intermediate tool turns to the final receipt schema.
                # Native Codex and the controller still validate the final file/hash/schema.
                text_options = dict(body.get("text") or {})
                text_options.pop("format", None)
                if text_options:
                    body["text"] = text_options
                else:
                    body.pop("text", None)
                if not any(item.get("type") == "function_call_output" for item in body.get("input", []) if isinstance(item, dict)):
                    body["tool_choice"] = "required"
        elif "provider" in body or self.config.provider_order or self.config.allow_provider_fallback:
            raise ValueError("Provider routing is only supported on OpenRouter")
        if self.config.provider == "autodl" and self.config.model in {"DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash", "gpt-5.6-sol"} and self.config.g_output_delivery == "candidate_file_all" and body.get("tools"):
            text_options = dict(body.get("text") or {})
            text_options.pop("format", None)
            if text_options:
                body["text"] = text_options
            else:
                body.pop("text", None)
            if not any(item.get("type") == "function_call_output" for item in body.get("input", []) if isinstance(item, dict)):
                body["tool_choice"] = "required"
        if self.config.provider == "autodl" and isinstance(body.get("input"), list):
            # Codex's native ResponseItem omits status on replay. Restore only
            # metadata witnessed in this operation's completed provider response.
            items = []
            self.replay_aliases = []
            for item in body["input"]:
                item = dict(item)
                status = self.completed_item_status.get((item.get("type"), item.get("id")))
                if item.get("type") == "function_call_output" and item.get("call_id") in self.completed_call_ids and "output" in item:
                    # This marks a complete result message, not command success.
                    status = "completed"
                if status and "status" not in item:
                    item["status"] = status
                original = item.get("id")
                alias = self.provider_item_aliases.get((item.get("type"), original))
                if alias:
                    item["id"] = alias
                    self.replay_aliases.append({"type": item["type"], "original": original, "alias": alias})
                items.append(item)
            body["input"] = items
        return body

    def observe(self, record, event):
        event_type = event.get("type", "untyped")
        counts = record.setdefault("event_counts", {})
        counts[event_type] = counts.get(event_type, 0) + 1
        response = event.get("response", event)
        if not isinstance(response, dict):
            return
        model = response.get("model")
        if model is not None:
            record["observed_model"] = model
            if model not in self.config.accepted_response_models:
                raise ValueError("Provider returned a different model")
            record["returned_model"] = model
        for key in ("id", "provider", "service_tier"):
            if response.get(key) is not None:
                record[key] = response[key]
        if self.config.provider == "autodl" and event.get("type") == "response.completed":
            for item in response.get("output", []):
                original = item.get("id", "")
                prefix = {"reasoning": "rs_", "message": "msg_", "function_call": "fc_"}.get(item.get("type"))
                # Alias only generic IDs actually returned by this provider; preserve
                # raw responses, reasoning, call_id, tool arguments and tool results.
                if prefix and original.startswith("item_"):
                    self.provider_item_aliases[(item["type"], original)] = prefix + original[5:]
                if item.get("id") and item.get("status") == "completed":
                    self.completed_item_status[(item.get("type"), item["id"])] = "completed"
                if item.get("type") == "function_call" and item.get("call_id"):
                    self.completed_call_ids.add(item["call_id"])
        if isinstance(response.get("usage"), dict):
            record["usage"] = response["usage"]
            cost = response["usage"].get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
                record["api_cost_usd"] = cost
        if response.get("status") in {"incomplete", "failed"} or event.get("type") in {"error", "response.failed", "response.incomplete"}:
            raise ValueError("Provider reported failed or truncated output")
        if event.get("type") == "response.completed":
            record["completed"] = True

    def start(self, socket_path):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply_receipt(self, path, body_hash):
                receipt = json.loads(path.read_text())
                if receipt["body_sha256"] != body_hash:
                    self.send_error(409, "Request identity mismatch")
                    return
                state = receipt["state"]
                if state != "completed":
                    data = json.dumps({"state": state}).encode()
                    self.send_response(202 if state == "pending" else 502)
                    self.send_header("Content-Type", "application/json")
                else:
                    data = path.with_suffix(".sse").read_bytes()
                    if hashlib.sha256(data).hexdigest() != receipt["response_sha256"]:
                        self.send_error(409, "Persisted response identity mismatch")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def do_GET(self):
                key = self.path.removeprefix("/api/v1/receipts/")
                body_hash = self.headers.get("X-RSI-Body-SHA256", "")
                if (outer.evidence_dir is None or not self.path.startswith("/api/v1/receipts/")
                        or not re.fullmatch("[0-9a-f]{32}", key)
                        or not re.fullmatch("[0-9a-f]{64}", body_hash)):
                    self.send_error(400)
                    return
                path = outer.evidence_dir / "receipts" / (key + ".json")
                if not path.exists():
                    self.send_error(404, "No persisted request; inference was not resubmitted")
                    return
                self.reply_receipt(path, body_hash)

            def do_POST(self):
                with outer.lock:
                    receipt_path = None
                    receipt = None
                    record = {"ordinal": len(outer.requests), "returned_model": None, "usage": None,
                              "api_cost_usd": None, "completed": False}
                    outer.requests.append(record)
                    try:
                        budget = outer.config.budget
                        if len(outer.requests) > budget.max_requests or time.monotonic() - outer.started >= budget.wall_time_seconds:
                            raise ValueError("Meta API operation budget exhausted")
                        length = int(self.headers.get("Content-Length", "0"))
                        if self.path != "/api/v1/responses" or not 0 < length <= 16000000:
                            raise ValueError("Undeclared transport endpoint/request size")
                        raw_body = self.rfile.read(length)
                        key = self.headers.get("X-RSI-Request-ID")
                        if key is not None:
                            if outer.evidence_dir is None or not re.fullmatch("[0-9a-f]{32}", key):
                                raise ValueError("Invalid durable request identity")
                            if outer.config.response_delivery != "buffered_json":
                                raise ValueError("Durable delivery requires the registered buffered response mode")
                            body_hash = hashlib.sha256(raw_body).hexdigest()
                            if self.headers.get("X-RSI-Body-SHA256") != body_hash:
                                raise ValueError("Durable request body mismatch")
                            receipt_path = outer.evidence_dir / "receipts" / (key + ".json")
                            if receipt_path.exists():
                                outer.requests.pop()  # Retrieval does not reserve another provider call.
                                self.reply_receipt(receipt_path, body_hash)
                                return
                            receipt = {"state": "pending", "request_id": key,
                                       "body_sha256": body_hash, "ordinal": record["ordinal"]}
                            persist_bytes(receipt_path, json.dumps(receipt).encode())
                            record["durable_request_id"] = key
                        body = outer.validate_body(json.loads(raw_body))
                        record["tool_types"] = sorted({tool.get("type", "unknown") for tool in body.get("tools", [])})
                        record["replay_id_aliases"] = list(outer.replay_aliases)
                        record["reasoning"] = body.get("reasoning")
                        record["response_delivery"] = outer.config.response_delivery
                        record["upstream_stream"] = body.get("stream")
                        record["text_format"] = body.get("text", {}).get("format", {}).get("type")
                        record["tool_choice"] = body.get("tool_choice")
                        record["max_output_tokens"] = body["max_output_tokens"]
                        record["store"] = body["store"]
                        record["previous_response_id"] = body.get("previous_response_id")
                        record["input_items"] = len(body.get("input", []))
                        record["input_item_shapes"] = [{"type": x.get("type"), "keys": sorted(x), "status": x.get("status")}
                                                       for x in body.get("input", []) if isinstance(x, dict)]
                        request = urllib.request.Request(outer.config.base_url.rstrip("/") + "/responses", json.dumps(body).encode(),
                            {"Authorization": "Bearer " + outer._secret, "Content-Type": "application/json"})
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
                        timeout = min(outer.config.response_timeout_seconds, budget.wall_time_seconds)
                        if outer.operation_budget:
                            outer.operation_budget.reserve_request()
                            timeout = min(timeout, outer.operation_budget.remaining_seconds())
                        for retry_index in range(5):
                            try:
                                request_deadline = time.monotonic() + timeout
                                with opener.open(request, timeout=timeout) as response:
                                    record["http_status"] = response.status
                                    record["request_id"] = response.headers.get("x-request-id")
                                    content_type = response.headers.get("Content-Type", "application/json")
                                    if outer.config.response_delivery == "buffered_json":
                                        streamed = "text/event-stream" in content_type
                                        if streamed:
                                            if outer.evidence_dir is None:
                                                raise ValueError("Stream buffering requires durable evidence")
                                            data = buffered_stream_response(response, budget.max_event_bytes, request_deadline,
                                                outer.evidence_dir / (str(record["ordinal"]).zfill(3) + ".upstream.sse"), record,
                                                outer.operation_budget.receive_bytes if outer.operation_budget else lambda count: None)
                                        else:
                                            data = response.read(budget.max_event_bytes + 1)
                                        record["response_bytes"] = len(data)
                                        record["response_content_type"] = content_type
                                        if len(data) > budget.max_event_bytes:
                                            raise ValueError("Buffered response exceeds resource budget")
                                        if outer.evidence_dir is not None:
                                            outer.evidence_dir.mkdir(parents=True, exist_ok=True)
                                            filename = f"{record['ordinal']:03d}.json"
                                            persist_bytes(outer.evidence_dir / filename, data)
                                            record["provider_response_file"] = filename
                                            record["provider_response_sha256"] = hashlib.sha256(data).hexdigest()
                                        if outer.operation_budget and not streamed:
                                            outer.operation_budget.receive_bytes(len(data))
                                        value = json.loads(data)
                                        events = list(completed_response_events(value))
                                        for event in events:
                                            outer.observe(record, event)
                                        if record["returned_model"] not in outer.config.accepted_response_models:
                                            raise ValueError("Missing buffered provider model identity")
                                        used = value["usage"]["output_tokens"]
                                        outer.output_tokens += used
                                        if outer.output_tokens > budget.max_total_output_tokens:
                                            raise ValueError("Meta output token budget exceeded")
                                        if outer.operation_budget:
                                            outer.operation_budget.finish_request(used)
                                        record["accounting_completed"] = True
                                        frames = []
                                        for sequence, event in enumerate(events):
                                            event["sequence_number"] = sequence
                                            frames.append(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
                                        wire = b"".join(frames)
                                        if receipt_path is not None:
                                            persist_bytes(receipt_path.with_suffix(".sse"), wire)
                                            receipt.update(state="completed", response_sha256=hashlib.sha256(wire).hexdigest(),
                                                           provider_response_sha256=record["provider_response_sha256"],
                                                           usage=record["usage"], model=record["returned_model"])
                                            persist_bytes(receipt_path, json.dumps(receipt).encode())
                                        for previous in outer.requests:
                                            if previous.get("durable_request_id") == key and previous.get("terminal_failure") and previous.get("provider_outcome") == "failed":
                                                previous["recovered_by_ordinal"] = record["ordinal"]
                                        record["delivery_started"] = True
                                        self.send_response(response.status)
                                        self.send_header("Content-Type", "text/event-stream")
                                        self.send_header("Content-Length", str(len(wire)))
                                        self.end_headers()
                                        self.wfile.write(wire)
                                        self.wfile.flush()
                                        return
                                    self.send_response(response.status)
                                    self.send_header("Content-Type", content_type)
                                    self.end_headers()
                                    size = 0
                                    if "text/event-stream" in content_type:
                                        for line in response:
                                            size += len(line)
                                            if outer.operation_budget:
                                                outer.operation_budget.receive_bytes(len(line))
                                            if size > budget.max_event_bytes or time.monotonic() - outer.started > budget.wall_time_seconds:
                                                raise ValueError("Meta stream resource budget exhausted")
                                            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                                                outer.observe(record, json.loads(line[5:]))
                                            self.wfile.write(line)
                                            self.wfile.flush()
                                        record["stream_bytes"] = size
                                        record["stream_eof"] = True
                                    else:
                                        data = response.read(budget.max_event_bytes + 1)
                                        if outer.operation_budget:
                                            outer.operation_budget.receive_bytes(len(data))
                                        if len(data) > budget.max_event_bytes:
                                            raise ValueError("Meta response exceeds budget")
                                        outer.observe(record, json.loads(data))
                                        record["completed"] = True
                                        self.wfile.write(data)
                                if not record["completed"]:
                                    raise ValueError("Provider stream ended without completion; receipt retained, no automatic retry")
                                usage = record["usage"] or {}
                                used = usage.get("output_tokens")
                                if type(used) is not int or used < 0:
                                    raise ValueError("Provider usage unavailable; cannot enforce cumulative token budget")
                                outer.output_tokens += used
                                if outer.output_tokens > budget.max_total_output_tokens:
                                    raise ValueError("Meta output token budget exceeded")
                                if record["returned_model"] not in outer.config.accepted_response_models or not record["completed"]:
                                    raise ValueError("Missing provider identity or completion evidence")
                                if outer.operation_budget:
                                    outer.operation_budget.finish_request(used)
                                break
                            except (TerminalProviderFailure, urllib.error.HTTPError) as provider_failure:
                                if isinstance(provider_failure, urllib.error.HTTPError):
                                    if provider_failure.code not in {429, 502, 503, 504}:
                                        raise
                                    status = provider_failure.code
                                    record["http_status"] = status
                                    record["terminal_failure"] = {"code": f"http_{status}", "usage": None,
                                        "provider_completion_unknown": status != 429, "request_rejected": status == 429}
                                    # No successful response or native tool output was delivered.
                                    # Preserve conservative unknown usage and retry within the same operation.
                                record["error_type"] = "terminal_provider_failure"
                                record["provider_outcome"] = "failed"
                                failure_path = outer.evidence_dir / (str(record["ordinal"]).zfill(3) + ".terminal_failure.json")
                                persist_bytes(failure_path, json.dumps(record).encode())
                                if outer.operation_budget:
                                    if record.get("http_status") == 429:
                                        outer.operation_budget.finish_rejected_request(str(failure_path))
                                    else:
                                        outer.operation_budget.finish_failed_request(body["max_output_tokens"], str(failure_path))
                                persist_bytes(outer.evidence_dir.parent / "transport_records.json", json.dumps(outer.requests).encode())
                                if retry_index == 4 or len(outer.requests) >= budget.max_requests:
                                    raise
                                pause = min(30 * (retry_index + 1), 60)
                                if outer.operation_budget and outer.operation_budget.remaining_seconds() <= pause + 1:
                                    raise
                                time.sleep(pause)
                                body = outer.validate_body(json.loads(raw_body))
                                request.data = json.dumps(body).encode()
                                record = {"ordinal": len(outer.requests), "returned_model": None, "usage": None,
                                          "api_cost_usd": None, "completed": False, "retry_of": key,
                                          "retry_index": retry_index + 1, "max_output_tokens": body["max_output_tokens"],
                                          "response_delivery": outer.config.response_delivery, "upstream_stream": body.get("stream"),
                                          "reasoning": body.get("reasoning"), "durable_request_id": key}
                                outer.requests.append(record)
                                timeout = min(outer.config.response_timeout_seconds, budget.wall_time_seconds)
                                if outer.operation_budget:
                                    outer.operation_budget.reserve_request()
                                    timeout = min(timeout, outer.operation_budget.remaining_seconds())
                    except Exception as exc:
                        if record.get("delivery_started") and isinstance(exc, OSError):
                            # Client delivery can be retried with GET; provider work cannot.
                            record["delivery_error"] = type(exc).__name__
                            return
                        if isinstance(exc, ValueError):
                            message = str(exc).replace(outer._secret, "[REDACTED]")
                            record["validation_error"] = message[:512]
                        # Preserve bounded validation details, including OpenRouter's
                        # nested upstream error, after removing echoed request fields
                        # and credentials. The outer message alone often hides 422s.
                        status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                        if isinstance(exc, urllib.error.HTTPError):
                            try:
                                problem = json.loads(exc.read(32768))
                                details = problem.get("error", problem)
                                if isinstance(details, dict):
                                    safe = {key: details[key] for key in ("code", "type", "param", "message")
                                            if isinstance(details.get(key), (str, int))}
                                    metadata = details.get("metadata")
                                    if isinstance(metadata, dict):
                                        raw = metadata.get("raw")
                                        if isinstance(raw, str):
                                            try:
                                                raw = json.loads(raw)
                                            except ValueError:
                                                pass
                                        def without_request_data(value):
                                            if isinstance(value, dict):
                                                return {k: without_request_data(v) for k, v in value.items()
                                                        if k.lower() not in {"input", "messages", "headers", "authorization",
                                                                             "api_key", "token", "request", "body"}}
                                            if isinstance(value, list):
                                                return [without_request_data(v) for v in value[:16]]
                                            return value
                                        if isinstance(raw, (str, dict, list)):
                                            safe["upstream"] = without_request_data(raw)
                                        if isinstance(metadata.get("provider_name"), str):
                                            safe["provider_name"] = metadata["provider_name"]
                                    encoded = json.dumps(safe, ensure_ascii=False)
                                    for name, value in os.environ.items():
                                        if len(value) >= 8 and name.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
                                            encoded = encoded.replace(value, "[REDACTED]")
                                    encoded = encoded.replace(outer._secret, "[REDACTED]")
                                    encoded = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", encoded)
                                    encoded = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~-]+", "Bearer [REDACTED]", encoded)
                                    record["provider_error"] = encoded[:4096]
                            except (OSError, ValueError, TypeError):
                                record["provider_error"] = "unavailable_or_non_json"
                        record["error_type"] = "rate_limit" if status == 429 else "authentication" if status in {401, 403} else type(exc).__name__
                        record["http_status"] = status or record.get("http_status")
                        outer.error = record["error_type"]
                        if receipt_path is not None and receipt is not None:
                            receipt.update(state="failed", error_type=record["error_type"],
                                           provider_outcome="unknown" if not record.get("completed") else "received")
                            persist_bytes(receipt_path, json.dumps(receipt).encode())
                        with contextlib.suppress(OSError):
                            self.send_error(status or 502, "Meta transport rejected the request or upstream response")
                    finally:
                        if outer.evidence_dir is not None:
                            persist_bytes(outer.evidence_dir.parent / "transport_records.json", json.dumps(outer.requests).encode())

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            # Keep the already-dispatched provider request alive during an SSH
            # disconnect. Its existing HTTP/operation timeout remains authoritative.
            daemon_threads = False
        self.server = Server(str(socket_path), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._secret = ""

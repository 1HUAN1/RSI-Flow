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
    for item in response["output"]:
        if not isinstance(item, dict) or not item.get("id") or item.get("status") not in {None, "completed"}:
            raise ValueError("Buffered provider output item is incomplete or unbound")
    yield {"type": "response.created", "response": {"id": response["id"], "model": response.get("model")}}
    for index, item in enumerate(response["output"]):
        yield {"type": "response.output_item.added", "output_index": index, "item": item}
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    yield {"type": "response.completed", "response": response}


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
            body["stream"] = False
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
        elif "provider" in body or self.config.provider_order or self.config.allow_provider_fallback:
            raise ValueError("Provider routing is only supported on OpenRouter")
        if self.config.provider == "autodl" and isinstance(body.get("input"), list):
            # Codex's native ResponseItem omits status on replay. Restore only
            # metadata witnessed in this operation's completed provider response.
            items = []
            for item in body["input"]:
                item = dict(item)
                status = self.completed_item_status.get((item.get("type"), item.get("id")))
                if item.get("type") == "function_call_output" and item.get("call_id") in self.completed_call_ids and "output" in item:
                    # This marks a complete result message, not command success.
                    status = "completed"
                if status and "status" not in item:
                    item["status"] = status
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

            def do_POST(self):
                with outer.lock:
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
                        body = outer.validate_body(json.loads(self.rfile.read(length)))
                        record["tool_types"] = sorted({tool.get("type", "unknown") for tool in body.get("tools", [])})
                        record["reasoning"] = body.get("reasoning")
                        record["response_delivery"] = outer.config.response_delivery
                        record["upstream_stream"] = body.get("stream")
                        record["text_format"] = body.get("text", {}).get("format", {}).get("type")
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
                        with opener.open(request, timeout=timeout) as response:
                            record["http_status"] = response.status
                            record["request_id"] = response.headers.get("x-request-id")
                            content_type = response.headers.get("Content-Type", "application/json")
                            if outer.config.response_delivery == "buffered_json":
                                data = response.read(budget.max_event_bytes + 1)
                                record["response_bytes"] = len(data)
                                record["response_content_type"] = content_type
                                if outer.operation_budget:
                                    outer.operation_budget.receive_bytes(len(data))
                                if len(data) > budget.max_event_bytes:
                                    raise ValueError("Buffered response exceeds resource budget")
                                if outer.evidence_dir is not None:
                                    outer.evidence_dir.mkdir(parents=True, exist_ok=True)
                                    filename = f"{record['ordinal']:03d}.json"
                                    (outer.evidence_dir / filename).write_bytes(data)
                                    record["provider_response_file"] = filename
                                    record["provider_response_sha256"] = hashlib.sha256(data).hexdigest()
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
                                self.send_response(response.status)
                                self.send_header("Content-Type", "text/event-stream")
                                self.end_headers()
                                for sequence, event in enumerate(events):
                                    event["sequence_number"] = sequence
                                    self.wfile.write(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
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
                    except Exception as exc:
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
                        with contextlib.suppress(OSError):
                            self.send_error(status or 502, "Meta transport rejected the request or upstream response")

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
        self.server = Server(str(socket_path), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._secret = ""

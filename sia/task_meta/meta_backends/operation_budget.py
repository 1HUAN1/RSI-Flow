"""Trusted cumulative limits shared by every Codex stage in one G operation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from sia.task_meta.meta_harness.bundle import atomic_json

from .contracts import BackendUnavailable


class OperationBudget:
    """Reserve before transport side effects; never reset limits between stages."""

    def __init__(self, path: Path, limits):
        self.path, self.limits = Path(path), limits
        self.lock = threading.RLock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["limits"] != limits.model_dump():
                raise ValueError("Operation resource ceilings changed during recovery")
        else:
            self.state = {"limits": limits.model_dump(), "started_at": time.time(),
                          "requests": 0, "output_tokens": 0, "network_bytes": 0,
                          "files": {}, "pending_requests": 0}
            self._save()

    def _save(self):
        atomic_json(self.path, self.state)

    def remaining_seconds(self):
        remaining = self.limits.wall_time_seconds - max(0.0, time.time() - self.state["started_at"])
        if remaining <= 0:
            raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Cumulative Meta operation wall-time limit reached")
        return remaining

    def output_limit(self, requested):
        with self.lock:
            self.remaining_seconds()
            remaining = self.limits.max_total_output_tokens - self.state["output_tokens"]
            if type(requested) is not int or requested <= 0 or remaining <= 0:
                raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Cumulative Meta output-token limit reached")
            return min(requested, remaining, self.limits.max_output_tokens_per_request)

    def reserve_request(self):
        with self.lock:
            self.remaining_seconds()
            if self.state["pending_requests"]:
                raise BackendUnavailable("META_OPERATION_PENDING", "An upstream request has no confirmed usage/completion; reconciliation is required")
            if self.state["requests"] >= self.limits.max_requests:
                raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Cumulative Meta API request limit reached")
            self.state["requests"] += 1
            self.state["pending_requests"] += 1
            self._save()

    def finish_request(self, used):
        with self.lock:
            if type(used) is not int or used < 0 or not self.state["pending_requests"]:
                raise BackendUnavailable("META_OPERATION_PENDING", "Missing or unbound provider output-token accounting")
            self.state["output_tokens"] += used
            self.state["pending_requests"] -= 1
            self._save()
            if self.state["output_tokens"] > self.limits.max_total_output_tokens:
                raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Provider exceeded the cumulative output-token ceiling")

    def receive_bytes(self, count):
        with self.lock:
            self.state["network_bytes"] += count
            self._save()
            self.remaining_seconds()
            if self.state["network_bytes"] > self.limits.max_event_bytes:
                raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Cumulative response-stream limit reached")

    def check_files(self, stage_id, workspace_bytes, event_bytes):
        with self.lock:
            self.state["files"][stage_id] = {"workspace_bytes": workspace_bytes, "event_bytes": event_bytes}
            self._save()
            self.remaining_seconds()
            if (sum(s["workspace_bytes"] for s in self.state["files"].values()) > self.limits.max_workspace_bytes
                    or sum(s["event_bytes"] for s in self.state["files"].values()) > self.limits.max_event_bytes):
                raise BackendUnavailable("META_OPERATION_BUDGET_EXHAUSTED", "Cumulative stage workspace/event limit reached")

"""Local protocol fixtures only; never experimental model evidence."""
import copy
import http.client
import io
import json
import socket
from unittest.mock import patch

import pytest

from sia.task_meta.meta_backends.contracts import MetaBackendConfig, MetaBudget
from sia.task_meta.meta_backends.operation_budget import OperationBudget
from sia.task_meta.meta_backends.transport import ResponsesTransport, completed_response_events


RESPONSE = {"id": "test_override_response", "status": "completed", "model": "DeepSeek-V4-Flash-0731",
            "usage": {"input_tokens": 20, "output_tokens": 11}, "output": [
                {"id": "test_override_reasoning", "type": "reasoning", "status": "completed", "summary": []},
                {"id": "test_override_call", "type": "function_call", "status": "completed",
                 "name": "exec_command", "call_id": "test_override_id", "arguments": '{"cmd":"true"}'}]}


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("usage", None), ("output", []), ("id", None)])
def test_partial_response_cannot_be_framed_as_completed(field, value):
    response = copy.deepcopy(RESPONSE)
    response[field] = value
    with pytest.raises(ValueError):
        list(completed_response_events(response))


@pytest.mark.parametrize("valid", [True, False])
def test_buffered_transport_preserves_real_items_and_pending_accounting(tmp_path, valid):
    config = MetaBackendConfig(provider="autodl", model="DeepSeek-V4-Flash", response_delivery="buffered_json",
                               budget=MetaBudget(max_total_output_tokens=100))
    ledger = OperationBudget(tmp_path / "budget.json", config.budget)
    transport = ResponsesTransport(config, "test_override_secret", ledger, tmp_path / "responses")
    path = str(tmp_path / "t.sock")
    seen = []

    class Reply(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json", "x-request-id": "test_override_request"}

    class Opener:
        def open(self, request, timeout):
            seen.append(json.loads(request.data))
            return Reply(json.dumps(RESPONSE).encode() if valid else b"")

    class Connection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect(path)

    transport.start(path)
    try:
        with patch("urllib.request.build_opener", return_value=Opener()):
            connection = Connection("localhost")
            connection.request("POST", "/api/v1/responses", json.dumps({"model": config.model, "stream": True,
                               "input": [], "reasoning": {"effort": "high"}, "tools": []}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
            connection.close()
        assert seen[0]["stream"] is False
        assert seen[0]["reasoning"] == {"effort": "high"}
        assert ledger.state["requests"] == 1
        if valid:
            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith(b"data: ")]
            assert [e["item"] for e in events if e["type"] == "response.output_item.done"] == RESPONSE["output"]
            assert events[-1]["response"] == RESPONSE
            assert ledger.state["pending_requests"] == 0
            assert ledger.state["output_tokens"] == 11
            assert json.loads((tmp_path / "responses/000.json").read_text()) == RESPONSE
        else:
            assert response.status == 502
            assert ledger.state["pending_requests"] == 1
            assert ledger.state["output_tokens"] == 0
            assert transport.error
        assert len(seen) == 1  # no automatic replay after an empty response
    finally:
        transport.close()

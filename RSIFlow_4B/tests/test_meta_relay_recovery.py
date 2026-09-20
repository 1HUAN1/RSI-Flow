import json
import socket

import pytest

from sia.task_meta.meta_backends.contracts import BackendUnavailable
from sia.task_meta.meta_backends.remote_execution import _recover_stale_relay


def test_orphaned_relay_socket_is_recovered(tmp_path):
    path = tmp_path / "rsi_meta_stale.sock"
    relay = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    relay.bind(str(path))
    relay.close()

    _recover_stale_relay(path, tmp_path)

    assert not path.exists()
    audit = json.loads((tmp_path / "remote_relay_recovery.json").read_text())
    assert audit["reason"] == "stale_socket_connection_refused"


def test_active_relay_socket_is_preserved(tmp_path):
    path = tmp_path / "rsi_meta_active.sock"
    relay = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    relay.bind(str(path))
    relay.listen()
    try:
        with pytest.raises(BackendUnavailable, match="active Meta relay"):
            _recover_stale_relay(path, tmp_path)
        assert path.exists()
    finally:
        relay.close()
        path.unlink(missing_ok=True)

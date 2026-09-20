"""Credential-free HTTP-to-Unix bridge, executed inside the isolation boundary.

This is transport only. Send each inference request once; reconnect using only
read-only receipt retrieval. No API token or inference retry lives here.
"""

import http.client
import http.server
import hashlib
import socket
import subprocess
import sys
import threading
import time
import uuid


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/transport.sock")


class Bridge(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if self.path != "/api/v1/responses" or not 0 < size <= 16000000:
            self.send_error(400)
            return
        body = self.rfile.read(size)
        key = uuid.uuid4().hex
        headers = {"Content-Type": "application/json", "X-RSI-Request-ID": key,
                   "X-RSI-Body-SHA256": hashlib.sha256(body).hexdigest()}
        deadline = time.monotonic() + 3600  # Worker enforces any smaller operation deadline.
        method, path = "POST", self.path
        while time.monotonic() < deadline:
            conn = UnixConnection("localhost", timeout=min(15, deadline - time.monotonic()))
            try:
                conn.request(method, path, body if method == "POST" else None, headers)
                response = conn.getresponse()
                # Buffer before forwarding: a partial response is never appended
                # to a second copy after reconnecting.
                data = response.read(512000001)
                if len(data) > 512000000:
                    self.send_error(502, "Persisted response exceeds transport ceiling")
                    return
                if response.status != 202:
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    return
            except (OSError, http.client.HTTPException):
                pass
            finally:
                conn.close()
            # Even when the first POST's delivery is unknown, never resend it.
            method, path = "GET", "/api/v1/receipts/" + key
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        self.send_error(504, "Receipt retrieval deadline reached; provider request not resubmitted")


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 18443), Bridge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {"PATH": "/usr/bin:/bin", "HOME": "/home/meta", "CODEX_HOME": "/codex_home",
           "LANG": "C.UTF-8", "TMPDIR": "/tmp"}
    try:
        code = subprocess.call(sys.argv[1:], env=env)
    finally:
        server.shutdown()
    sys.exit(code)

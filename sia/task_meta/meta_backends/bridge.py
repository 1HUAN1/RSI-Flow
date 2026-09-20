"""Credential-free HTTP-to-Unix bridge, executed inside the isolation boundary.

This is transport only. No API token, model inference, or retry policy lives here.
"""

import http.client
import http.server
import socket
import subprocess
import sys
import threading


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(3600)
        self.sock.connect("/transport.sock")


class Bridge(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if self.path != "/api/v1/responses" or not 0 < size <= 16000000:
            self.send_error(400)
            return
        conn = UnixConnection("localhost", timeout=3600)
        try:
            conn.request("POST", self.path, self.rfile.read(size), {"Content-Type": "application/json"})
            response = conn.getresponse()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.end_headers()
            while chunk := response.read1(16384):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            conn.close()


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

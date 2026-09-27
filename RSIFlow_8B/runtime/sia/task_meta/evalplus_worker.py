"""Tiny stdlib worker. Only current test arguments arrive; no expected answers."""

import contextlib
import json
import signal
import sys
import time

from codec import UnsupportedWireValue, decode, encode


class CandidateTimeout(Exception):
    pass


def timeout_handler(*_):
    raise CandidateTimeout()


scope, function = {}, None
wire_in, wire_out = sys.stdin, sys.stdout
class QuietOutput:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


with contextlib.nullcontext(QuietOutput()) as quiet:
    for line in wire_in:
        response = {"status": "error", "error_type": "invalid_request"}
        try:
            request = json.loads(line)
            signal.signal(signal.SIGALRM, timeout_handler)
            if request["operation"] == "initialize":
                signal.setitimer(signal.ITIMER_REAL, request["timeout"])
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    exec(compile(request["code"], "<candidate>", "exec"), scope)
                    function = scope[request["entry_point"]]
                    if not callable(function):
                        raise TypeError("entry point is not callable")
                response = {"status": "ready"}
            elif request["operation"] == "call" and function is not None:
                args = decode(request["arguments"])
                signal.setitimer(signal.ITIMER_REAL, request["timeout"])
                started = time.monotonic()
                with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                    result = function(*args)
                    if request.get("output_not_none"):
                        result = result if isinstance(result, bool) else result is not None
                elapsed = time.monotonic() - started
                signal.setitimer(signal.ITIMER_REAL, 0)
                response = {"status": "ok", "result": encode(result), "arguments_after": encode(args), "seconds": elapsed}
            else:
                raise ValueError("invalid operation")
        except UnsupportedWireValue:
            response = {"status": "unsupported_wire_value"}
        except CandidateTimeout:
            response = {"status": "timeout"}
        except BaseException as exc:
            # Never send arbitrary exception strings, traceback frames, or source.
            response = {"status": "error", "error_type": type(exc).__name__[:100]}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        wire_out.write(json.dumps(response, allow_nan=False) + "\n")
        wire_out.flush()

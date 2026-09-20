"""Bounded, non-executable wire values for isolated function calls.

Only exact built-in values are supported. No pickle, imports, constructors,
arbitrary object hooks, cycles or shared-container aliases. IEEE non-finite
floats use explicit string tags; the interned empty tuple is preserved.
"""

import base64
import math

MAX_NODES = 200000
MAX_DEPTH = 80
MAX_STRING = 2000000


class UnsupportedWireValue(ValueError):
    pass


def encode(value):
    seen, count = set(), [0]

    def visit(value, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_NODES:
            raise UnsupportedWireValue("wire structure budget exceeded")
        kind = type(value)
        if value is None or kind is bool:
            return value
        if kind is str:
            if len(value) > MAX_STRING:
                raise UnsupportedWireValue("wire string budget exceeded")
            return value
        if kind is int:
            if value.bit_length() > 8192:
                raise UnsupportedWireValue("wire integer budget exceeded")
            return {"t": "int", "v": str(value)}
        if kind is float:
            if not math.isfinite(value):
                return {'t': 'special_float', 'v': value.hex()}
            return {"t": "float", "v": value.hex()}
        if kind is complex:
            return {'t': 'complex', 'v': [visit(value.real, depth + 1), visit(value.imag, depth + 1)]}
        if kind is bytes:
            if len(value) > MAX_STRING:
                raise UnsupportedWireValue("wire bytes budget exceeded")
            return {"t": "bytes", "v": base64.b64encode(value).decode("ascii")}
        if kind not in {list, tuple, dict, set, frozenset}:
            raise UnsupportedWireValue("non-builtin return/input value unsupported")
        if kind is tuple and not value:
            return {'t': 'tuple', 'v': []}  # CPython's shared immutable singleton.
        if id(value) in seen:
            raise UnsupportedWireValue("cycles/shared container aliases unsupported")
        seen.add(id(value))
        if kind is dict:
            return {"t": "dict", "v": [[visit(k, depth + 1), visit(v, depth + 1)] for k, v in value.items()]}
        return {"t": kind.__name__, "v": [visit(x, depth + 1) for x in value]}

    return visit(value, 0)


def decode(value):
    count = [0]

    def visit(value, depth):
        count[0] += 1
        if depth > MAX_DEPTH or count[0] > MAX_NODES:
            raise UnsupportedWireValue("wire structure budget exceeded")
        if value is None or type(value) is bool:
            return value
        if type(value) is str:
            if len(value) > MAX_STRING:
                raise UnsupportedWireValue("wire string budget exceeded")
            return value
        if type(value) is not dict or set(value) != {"t", "v"}:
            raise UnsupportedWireValue("invalid tagged wire value")
        kind, payload = value["t"], value["v"]
        if kind == 'special_float':
            if type(payload) is not str or payload not in {'nan', 'inf', '-inf'}:
                raise UnsupportedWireValue('invalid IEEE special value')
            return float(payload)
        if kind == 'complex':
            if type(payload) is not list or len(payload) != 2:
                raise UnsupportedWireValue('invalid complex payload')
            parts = [visit(part, depth + 1) for part in payload]
            if any(type(part) is not float for part in parts):
                raise UnsupportedWireValue('invalid complex components')
            return complex(*parts)
        if kind in {"int", "float", "bytes"}:
            if type(payload) is not str or len(payload) > MAX_STRING * 2:
                raise UnsupportedWireValue("invalid scalar payload")
            try:
                output = int(payload) if kind == "int" else float.fromhex(payload) if kind == "float" else base64.b64decode(payload, validate=True)
            except (ValueError, OverflowError) as exc:
                raise UnsupportedWireValue("invalid scalar encoding") from exc
            encode(output)  # Identical bounds in both directions.
            return output
        if kind not in {"list", "tuple", "set", "frozenset", "dict"} or type(payload) is not list:
            raise UnsupportedWireValue("invalid container payload")
        if kind == "dict":
            result = {}
            for pair in payload:
                if type(pair) is not list or len(pair) != 2:
                    raise UnsupportedWireValue("invalid dictionary pair")
                key, item = (visit(x, depth + 1) for x in pair)
                try:
                    if key in result:
                        raise UnsupportedWireValue("duplicate dictionary key")
                    result[key] = item
                except TypeError as exc:
                    raise UnsupportedWireValue("unhashable dictionary key") from exc
            return result
        result = [visit(x, depth + 1) for x in payload]
        try:
            return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](result)
        except TypeError as exc:
            raise UnsupportedWireValue("unhashable set item") from exc

    return visit(value, 0)

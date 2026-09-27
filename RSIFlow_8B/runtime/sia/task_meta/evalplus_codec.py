"""Bounded, non-executable wire values for isolated function calls.

Only exact built-in values are supported. No pickle, imports, constructors,
arbitrary object hooks or cycles. Shared-container aliases retain identity. IEEE non-finite
floats use explicit string tags.
"""

import base64
import math

MAX_NODES = 200000
MAX_DEPTH = 80
MAX_STRING = 2000000


class UnsupportedWireValue(ValueError):
    pass


def encode(value):
    seen, active, count = {}, set(), [0]

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
        if id(value) in active:
            raise UnsupportedWireValue("cycles unsupported")
        if id(value) in seen:
            return {"t": "ref", "v": seen[id(value)]}
        seen[id(value)] = len(seen)
        active.add(id(value))
        if kind is dict:
            payload = [[visit(k, depth + 1), visit(v, depth + 1)] for k, v in value.items()]
        else:
            payload = [visit(x, depth + 1) for x in value]
        active.remove(id(value))
        return {"t": kind.__name__, "v": payload}

    return visit(value, 0)


def decode(value):
    count, memo, active = [0], [], set()

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
        if kind == "ref":
            if type(payload) is not int or payload < 0 or payload >= len(memo) or payload in active:
                raise UnsupportedWireValue("invalid or cyclic container reference")
            return memo[payload]
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
        slot = len(memo)
        memo.append(None)
        active.add(slot)
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
        else:
            items = [visit(x, depth + 1) for x in payload]
            try:
                result = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](items)
            except TypeError as exc:
                raise UnsupportedWireValue("unhashable set item") from exc
        memo[slot] = result
        active.remove(slot)
        return result

    return visit(value, 0)

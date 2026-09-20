"""Permit bounded task strategy hooks while retaining immutable evaluation/I/O."""

import ast

EDITABLE_FUNCTIONS = {"format_question", "parse_answer", "harness_config", "retry_prompt", "select_artifact_context"}
PURE_GLOBALS = {"json", "re", "str", "int", "float", "bool", "len", "list", "dict", "tuple", "set",
                "sorted", "enumerate", "range", "zip", "next", "min", "max", "sum", "any", "all",
                "ValueError", "TypeError", "AttributeError", "isinstance"}
PURE_ATTRIBUTES = {"items", "values", "keys", "get", "join", "strip", "lstrip", "rstrip", "upper", "lower",
                   "replace", "split", "rsplit", "splitlines", "startswith", "endswith", "index", "rindex",
                   "find", "rfind", "format", "append", "extend", "group", "groups", "groupdict",
                   "loads", "dumps", "JSONDecodeError", "search", "fullmatch", "findall", "sub", "escape",
                   "IGNORECASE", "DOTALL", "MULTILINE", "I", "S", "M"}


def validate_config_hook(function):
    """Validate finite runtime settings before committing an evolved Harness."""
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]  # A descriptive function docstring has no execution effect.
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise ValueError("harness_config must return a literal settings dictionary directly")
    try:
        config = ast.literal_eval(body[0].value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("harness_config must return a literal settings dictionary") from exc
    flags = {"use_artifacts", "generate_artifacts", "retry_on_parse_error", "retry_on_api_error"}
    if not isinstance(config, dict) or set(config) != flags | {"max_attempts", "artifact_char_limit"}:
        raise ValueError("harness_config must contain exactly the six declared settings")
    if any(type(config[key]) is not bool for key in flags):
        raise ValueError("harness_config flags must be boolean")
    if type(config["max_attempts"]) is not int or not 1 <= config["max_attempts"] <= 3:
        raise ValueError("harness_config max_attempts must be an integer between 1 and 3")
    if type(config["artifact_char_limit"]) is not int or not 0 <= config["artifact_char_limit"] <= 12000:
        raise ValueError("harness_config artifact_char_limit must be an integer between 0 and 12000")


def validate_harness_update(old_code: str, new_code: str) -> None:
    old, new = ast.parse(old_code), ast.parse(new_code)

    def protected(tree):
        return ast.dump(ast.Module(body=[node for node in tree.body
                                        if not (isinstance(node, ast.FunctionDef) and node.name in EDITABLE_FUNCTIONS)],
                                   type_ignores=[]), include_attributes=False)

    if protected(old) != protected(new):
        raise ValueError("Harness update changed protected model/runtime I/O; edit only the five declared strategy hooks")
    old_helpers = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef) and node.name in EDITABLE_FUNCTIONS}
    new_helpers = [node for node in new.body if isinstance(node, ast.FunctionDef) and node.name in EDITABLE_FUNCTIONS]
    if len(new_helpers) != len(old_helpers) or {node.name for node in new_helpers} != old_helpers.keys():
        raise ValueError("Harness helpers cannot be added, duplicated or removed")
    for function in new_helpers:
        previous = old_helpers[function.name]
        if (ast.dump(function.args) != ast.dump(previous.args) or function.decorator_list or function.returns
                or function.type_comment):
            raise ValueError("Harness helper signature/decorators cannot change")
        if function.name == "harness_config":
            validate_config_hook(function)
        local = {arg.arg for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]}
        local.update(node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
        local.update(node.name for node in ast.walk(function) if isinstance(node, ast.ExceptHandler) and node.name)
        for node in ast.walk(function):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.Lambda,
                                 ast.AsyncFunctionDef, ast.With, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom)):
                raise ValueError("Harness helper must be a pure strategy function")
            if isinstance(node, ast.FunctionDef) and node is not function:
                raise ValueError("Nested helper definitions are not supported")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in local | PURE_GLOBALS:
                raise ValueError(f"Harness helper references a protected name: {node.id}")
            if isinstance(node, ast.Attribute) and (node.attr not in PURE_ATTRIBUTES or not isinstance(node.ctx, ast.Load)):
                raise ValueError(f"Harness helper references a protected attribute: {node.attr}")
            if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
                raise ValueError("Harness helpers cannot assign/delete input dictionary entries")
            if isinstance(node, ast.Call) and not (
                (isinstance(node.func, ast.Name) and node.func.id in PURE_GLOBALS - {"json", "re"})
                or (isinstance(node.func, ast.Attribute) and node.func.attr in PURE_ATTRIBUTES)
            ):
                raise ValueError("Only declared pure helpers may be called from the harness")

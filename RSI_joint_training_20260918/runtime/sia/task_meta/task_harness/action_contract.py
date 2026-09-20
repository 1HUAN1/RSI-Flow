"""Normalize only explicit JSON actions; never infer tools or arguments."""
import json
import re


def normalize_action(message, tools, limit):
    names = {tool['name']: tool for tool in tools}
    native = message.get('tool_calls')
    if native:
        if not isinstance(native, list):
            raise ValueError('tool_calls must be a list')
        calls = []
        for item in native:
            if not isinstance(item, dict) or item.get('type', 'function') != 'function':
                raise ValueError('Unsupported native tool call')
            function = item.get('function')
            if not isinstance(function, dict):
                raise ValueError('Native tool call has no function object')
            calls.append({'name': function.get('name'), 'arguments': function.get('arguments')})
    else:
        content = message.get('content')
        if not isinstance(content, str):
            raise ValueError('Action requires JSON content or native tool_calls')
        content = content.strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, re.DOTALL)
        if fenced:
            content = fenced.group(1).strip()
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError('Action response must be a JSON object')
        calls = value.get('tools')
        if calls is None and isinstance(value.get('answer'), str):
            calls = [{'name': 'final_answer', 'arguments': {'answer': value['answer']}}]
    if not isinstance(calls, list) or not calls:
        raise ValueError('Action requires a nonempty tools list')
    normalized = []
    for call in calls:
        if not isinstance(call, dict) or call.get('name') not in names:
            raise ValueError('Unknown tool in action')
        arguments = call.get('arguments')
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError('Tool arguments must decode to an object')
        required = names[call['name']].get('parameters', {}).get('required', [])
        if not set(required) <= set(arguments):
            raise ValueError('Tool action is missing declared required arguments')
        normalized.append({'name': call['name'], 'arguments': arguments})
    return normalized[:limit]

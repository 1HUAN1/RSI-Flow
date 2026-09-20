"""Local Qwen-only model transport with per-response checkpoint verification."""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class TaskInfrastructureError(RuntimeError):
    pass


def qwen_message(text):
    """Decode the tokenizer's native tool markup without losing raw generation."""
    matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL))
    calls = []
    for match in matches:
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            return {'role': 'assistant', 'content': text}
        if not isinstance(value, dict) or not isinstance(value.get('name'), str) or not isinstance(value.get('arguments'), dict):
            return {'role': 'assistant', 'content': text}
        calls.append({'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': value['name'], 'arguments': json.dumps(value['arguments'], ensure_ascii=False)}})
    message = {'role': 'assistant', 'content': re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip() if calls else text}
    if calls:
        message['tool_calls'] = calls
    return message


class LocalTaskClient:
    def __init__(self, state, base_url, *, timeout=180, enable_thinking=False):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password:
            raise ValueError('Task model endpoint must be a credential-free local HTTP endpoint')
        if not state.checkpoint_path or not state.checkpoint_manifest:
            raise ValueError('Task client requires a real content-fingerprinted local checkpoint')
        self.state, self.base_url, self.timeout = state, base_url.rstrip('/'), timeout
        self.enable_thinking = enable_thinking

    def __call__(self, messages, *, tools=None, seed, max_tokens, temperature):
        payload = {'model': self.state.model_ref, 'messages': messages, 'seed': seed,
                   'max_tokens': max_tokens, 'temperature': temperature, 'stream': False,
                   'enable_thinking': self.enable_thinking}
        if tools:
            payload['tools'] = tools
        started = time.monotonic()
        try:
            request = Request(self.base_url + '/chat/completions', data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json'})
            with build_opener(ProxyHandler({})).open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except Exception as exc:
            raise TaskInfrastructureError(f'Local checkpoint request failed: {type(exc).__name__}: {exc}') from exc
        binding = result.get('local_checkpoint_binding', {})
        if (result.get('model') != self.state.model_ref
                or binding.get('checkpoint_path') != str(Path(self.state.checkpoint_path).resolve())
                or binding.get('weights') != self.state.checkpoint_manifest):
            raise TaskInfrastructureError('Every response must bind the requested checkpoint path and content hashes')
        try:
            choice = result['choices'][0]
            message = choice['message']
            if message['role'] != 'assistant':
                raise ValueError('Not an assistant response')
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise TaskInfrastructureError('Malformed Task model response') from exc
        return {'message': message, 'usage': result.get('usage'), 'binding': binding,
                'finish_reason': choice.get('finish_reason'), 'request_id': result.get('id'),
                'model_ref_requested': self.state.model_ref, 'model_ref_response': result.get('model'),
                'raw_generation': result.get('raw_generation'), 'wall_time_seconds': time.monotonic() - started,
                'serving_endpoint': self.base_url}

"""Build a pinned Codex native catalog from saved public provider metadata.

Declared native tool choices require the separate real compatibility smoke;
public model metadata is never treated as successful tool/API evidence.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sia.task_meta.storage import digest, save_json

PIN = '3d2ee51ca2d5db578f328aa75e20aa22c0197c9a'
MODEL = 'deepseek/deepseek-v4-flash-0731'


def build_catalog(metadata_path, codex_source, output):
    source, output = Path(codex_source), Path(output)
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != PIN:
        raise ValueError('Catalog adapter requires the registered Codex source commit')
    raw = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    raw = raw.get('official_metadata', raw)
    if 'data' in raw:
        raw = next(m for m in raw['data'] if m['id'] == MODEL)
    if raw.get('id') != MODEL:
        raise ValueError('Public metadata does not identify the exact requested model')
    contexts = [raw.get('context_length'), raw.get('top_provider', {}).get('context_length')]
    if any(type(v) is not int or v <= 0 for v in contexts):
        raise ValueError('Public model and provider context limits are required')
    if not {'tools', 'structured_outputs'} <= set(raw.get('supported_parameters', [])):
        raise ValueError('The public model does not declare required basic parameters')
    prompt = source / 'codex-rs/models-manager/prompt.md'
    model = {'slug': MODEL, 'display_name': raw.get('name', MODEL),
             'description': 'Pinned OpenRouter Meta; real native tool compatibility requires API smoke.',
             'default_reasoning_level': None, 'supported_reasoning_levels': [],
             'shell_type': 'unified_exec', 'visibility': 'list', 'supported_in_api': True, 'priority': 0,
             'availability_nux': None, 'upgrade': None,
             'model_messages': {'instructions_template': prompt.read_text(encoding='utf-8'),
                                'instructions_variables': None},
             'include_skills_usage_instructions': False, 'include_plugin_usage_instructions': False,
             'include_apps_usage_instructions': False, 'supports_reasoning_summary_parameter': False,
             'default_reasoning_summary': 'none', 'support_verbosity': False, 'default_verbosity': None,
             'apply_patch_tool_type': 'freeform', 'truncation_policy': {'mode': 'bytes', 'limit': 10000},
             'context_window': min(contexts), 'max_context_window': min(contexts),
             'effective_context_window_percent': 95, 'experimental_supported_tools': [],
             'input_modalities': ['text'], 'supports_search_tool': False, 'use_responses_lite': False,
             'node_repl_disabled': True, 'tool_mode': 'direct', 'multi_agent_version': None}
    save_json(output, {'models': [model]})
    evidence = {'status': 'IMPLEMENTED_NOT_API_VALIDATED', 'model': MODEL, 'codex_commit': commit,
                'catalog_sha256': digest(output), 'metadata_sha256': digest(Path(metadata_path)),
                'native_base_instructions_sha256': digest(prompt),
                'public_model_context_length': contexts[0], 'public_top_provider_context_length': contexts[1],
                'selected_context_limit': min(contexts), 'api_called': False,
                'native_tools': 'unified_exec/freeform apply_patch: declared for compatibility smoke, unverified',
                'controller_policy': {'output_truncation_bytes': 10000, 'context_headroom_percent': 5}}
    save_json(output.with_suffix('.provenance.json'), evidence)
    return evidence

"""Isolated IPC wrapper around the pinned BFCL v1.3 source (no inference)."""
import base64
import contextlib
import io
import json
import random
import sys
import types
from pathlib import Path

Path('official.zip').write_bytes(base64.b64decode(Path('official.b64').read_text()))
sys.path.insert(0, str(Path('official.zip').resolve()))
# Official AST checking only needs the custom handler's name-normalization flag.
# Register this adapter without importing unrelated API providers or credentials.
registration = types.ModuleType('bfcl_eval.constants.model_config')
registration.MODEL_CONFIG_MAPPING = {'rsi-harness': types.SimpleNamespace(underscore_to_dot=False)}
sys.modules[registration.__name__] = registration
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

entry = None
for line in sys.stdin:
    try:
        request = json.loads(line)
        with contextlib.redirect_stdout(io.StringIO()):
            if request['operation'] == 'reset':
                entry = request['entry']
                random.seed(request['seed'])
                execute_multi_turn_func_call([], entry['initial_config'], entry['involved_classes'],
                    'rsi-harness', entry['id'], long_context='long_context' in entry['id'])
                result = {'ready': True}
            elif request['operation'] == 'step':
                outputs, _ = execute_multi_turn_func_call([request['call']], entry['initial_config'],
                    entry['involved_classes'], 'rsi-harness', entry['id'],
                    long_context='long_context' in entry['id'])
                result = {'outputs': outputs}
            elif request['operation'] == 'score':
                row, prediction = request['entry'], request['prediction']
                category = row['id'].rsplit('_', 1)[0]
                random.seed(request['seed'])
                if category.startswith('multi_turn'):
                    if len(prediction) != len(request['ground_truth']):
                        result = {'valid': False, 'error_type': 'multi_turn:force_terminated'}
                    else:
                        # This release's official runner does not enable the extra
                        # irrelevance checker; preserve that exact scoring choice.
                        result = multi_turn_checker(prediction, request['ground_truth'], row, category, 'rsi-harness')
                elif 'relevance' in category:
                    result = {'valid': (not prediction) if 'irrelevance' in category else bool(prediction)}
                else:
                    language = 'Java' if category == 'java' else 'JavaScript' if category == 'javascript' else 'Python'
                    result = ast_checker(row['function'], prediction, request['ground_truth'], language, category, 'rsi-harness')
            elif request['operation'] == 'ready':
                result = {'ready': True, 'official_checkers_imported': True}
            else:
                raise ValueError('Unknown BFCL IPC operation')
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    except Exception as error:
        print(json.dumps({'infrastructure_error': type(error).__name__, 'detail': str(error)}), flush=True)

"""Pinned malformed-state checks fail tasks; arbitrary checker errors still block."""
import hashlib
import unittest

import test_contracts
from sia.task_meta.environments import _ENV_WORKER
from sia.task_meta.sandbox import LinuxSandbox, probe_isolation


AGE_CHECK = '''def check_func(final_state):
    for user in final_state.get("users", {}).values():
        if user.get("username") == "Priya Nair":
            age = user.get("profile_info", {}).get("age")
            return age == 29
    return False'''
AGE_HASH = 'b257542702c5d2fdce4f55fed63b0b2ea585240c551d26b36b1d88d2fcc0c96c'
UTILS = '''
def init_env_class(code, name): return object
def init_env_instance(cls, config): return config
def get_state_info(env): return env
def run_check_function(code, initial, final):
    scope = {"initial_state": initial}
    try:
        exec(code, scope)
        result = scope["check_func"](final)
        return isinstance(result, bool), result, None
    except Exception as exc:
        return False, None, str(exc)
'''


@unittest.skipUnless(probe_isolation()['available'], 'Linux isolation unavailable')
class StateFailureTests(unittest.TestCase):
    def evaluate(self, code, permitted):
        worker = LinuxSandbox().session(_ENV_WORKER, extra_files={'env_util.py':UTILS})
        try:
            worker.request(dict(operation='reset', seed=42, env_code='', class_name='fixture',
                init_config={'users':{'u':{'username':'Priya Nair','profile_info':'age 29'}}}, allowed_tools=[]))
            return worker.request(dict(operation='evaluate', checklist=[{'check_func':code}],
                                       task_failure_check_hashes=permitted))['checks'][0]
        finally:
            worker.close()

    def test_exact_pinned_type_failure_is_valid_false_with_original_evidence(self):
        self.assertEqual(hashlib.sha256(AGE_CHECK.encode()).hexdigest(), AGE_HASH)
        result = self.evaluate(AGE_CHECK, [AGE_HASH])
        self.assertTrue(result['valid'])
        self.assertFalse(result['result'])
        self.assertFalse(result['original_valid'])
        self.assertEqual(result['classification'], 'task_state_type_mismatch')
        self.assertEqual(result['error'], "'str' object has no attribute 'get'")

    def test_unapproved_or_changed_checker_stays_invalid(self):
        self.assertFalse(self.evaluate(AGE_CHECK, [])['valid'])
        self.assertFalse(self.evaluate(AGE_CHECK+'\n# changed checker', [AGE_HASH])['valid'])
        broken = 'def check_func(final_state):\n    raise RuntimeError("checker bug")'
        self.assertFalse(self.evaluate(broken, [AGE_HASH])['valid'])


if __name__ == '__main__': unittest.main()

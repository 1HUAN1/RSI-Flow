"""Scheduling checks: no model calls and no fabricated experiment results."""
import collections
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import test_contracts as native
from sia.task_meta import scoped_execution as scheduling


class Scheduling(unittest.TestCase):
    def test_slow_task_does_not_block_next_submission(self):
        later_started = threading.Event()
        def work(number):
            if number == 0:
                if not later_started.wait(3):
                    raise AssertionError('Fast completions did not refill the queue')
            if number == 2:
                later_started.set()
            return number
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = scheduling.bounded_results(pool, work, range(8), 2)
        self.assertEqual(result, list(range(8)))

    def test_exactly_once_global_queue_and_stable_order(self):
        seen = collections.Counter()
        guard = threading.Lock()
        def work(number):
            with guard:
                seen[number] += 1
            return number
        with ThreadPoolExecutor(max_workers=16) as pool:
            result = scheduling.bounded_results(pool, work, range(360), 32)
        self.assertEqual(result, list(range(360)))
        self.assertEqual(seen, collections.Counter(range(360)))

    def test_failed_task_is_not_retried(self):
        seen = []
        def work(number):
            seen.append(number)
            raise RuntimeError('fixture failure')
        with ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                scheduling.bounded_results(pool, work, range(4), 1)
        self.assertEqual(seen, [0])

    def test_sixteen_workers_use_four_endpoints(self):
        from multiprocessing import get_context
        ctx = get_context('fork')
        counter = ctx.Value('i', 0)
        slots = [ctx.BoundedSemaphore(2) for _ in range(4)]
        endpoints = []
        def factory(state, base_url=None):
            def client(**kwargs): return kwargs
            client.enable_thinking = False
            return client
        for _ in range(16):
            executor = SimpleNamespace(replicas=[{'base_url': str(i)} for i in range(4)], model_factory=factory)
            scheduling._initialize(executor, None, None, None, None, False, None, counter, slots)
            endpoints.append(executor._replica_endpoint)
            client = executor.model_factory(None)
            self.assertEqual(client(seed=42, max_tokens=2048), dict(seed=42, max_tokens=2048))
            self.assertFalse(client.enable_thinking)
        self.assertEqual(collections.Counter(endpoints), dict.fromkeys(['0','1','2','3'], 4))
        with patch.object(scheduling.os, 'sched_getaffinity', return_value=set(range(128))):
            self.assertEqual(scheduling.worker_count(4), 16)


if __name__ == '__main__':
    unittest.main()

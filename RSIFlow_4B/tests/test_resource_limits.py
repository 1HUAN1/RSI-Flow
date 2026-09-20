import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sia.task_meta.resource_limits import available_cpu_count, resource_snapshot
from sia.task_meta.meta_backends.contracts import MetaBudget, MetaBackendConfig
from sia.task_meta.meta_backends.local_execution import cpu_time_limit


class ResourceLimits(unittest.TestCase):
    def test_uses_quota_not_host_cpu_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('os.sched_getaffinity', return_value=set(range(128))):
                (root / 'cpu.max').write_text('4000000 100000')
                self.assertEqual(available_cpu_count(root), 40)
                (root / 'cpu.max').write_text('150000 100000')
                self.assertEqual(available_cpu_count(root), 1)
                (root / 'cpu.max').write_text('max 100000')
                self.assertEqual(available_cpu_count(root), 128)
                (root / 'memory.events').write_text('oom_kill 0\n')
                self.assertEqual(resource_snapshot(root)['memory.events'], 'oom_kill 0')

    def test_meta_cpu_seconds_allow_parallel_threads(self):
        budget = MetaBudget(wall_time_seconds=300, max_processes=256)
        self.assertEqual(cpu_time_limit(budget), 2400)
        self.assertEqual(budget.max_processes, 256)

    def test_selected_model_and_resources(self):
        root = Path(__file__).resolve().parents[1]
        config = MetaBackendConfig(**json.loads((root / 'runtime/configs/base.json').read_text())['meta'])
        self.assertEqual(config.model, 'DeepSeek-V4.1-Flash')
        self.assertEqual(config.expected_response_model, 'DeepSeek-Flash')
        self.assertEqual(config.budget.max_processes, 256)
        self.assertEqual(config.budget.memory_bytes, 8 * 1024**3)
        catalog = json.loads(Path(config.model_catalog_json).read_text())
        self.assertEqual([m['slug'] for m in catalog['models']], [config.model])


if __name__ == '__main__':
    unittest.main()

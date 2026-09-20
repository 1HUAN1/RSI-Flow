"""Launch CLI contract tests; no services, models, or network calls."""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch


class LaunchCli(unittest.TestCase):
    @staticmethod
    def config():
        return {
            'rounds': 3,
            'sft_epochs_per_update': 1,
            'sft_max_steps': -1,
            'candidate_policy': 'single_candidate_strict_positive_gain',
            'max_dataset_passes_per_stage': 2,
            'output_root': '/tmp/rsiflow_launch_cli_output',
        }

    def test_check_reaches_offline_preflight_and_never_starts(self):
        config = self.config()
        runtime = launch.ROOT / 'runtime'
        result = {'status': 'offline_preflight_passed', 'model_calls': 0,
                  'network_calls': 0, 'started': False}
        output = io.StringIO()
        with patch.object(launch, 'read', return_value=config), \
             patch('install_runtime.install', return_value=runtime), \
             patch.object(launch, 'check', return_value=({}, result)) as check, \
             patch.object(launch, 'dry_run') as dry_run, \
             patch.object(launch.subprocess, 'Popen') as popen, \
             contextlib.redirect_stdout(output):
            launch.main(['--config', 'fixture.json', '--check'])
        check.assert_called_once_with(config, runtime, external_readiness=False)
        dry_run.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), result)

    def test_execute_requests_external_readiness_before_start(self):
        class StopAfterPreflight(Exception):
            pass

        config = self.config()
        runtime = launch.ROOT / 'runtime'
        with patch.object(launch, 'read', return_value=config), \
             patch('install_runtime.install', return_value=runtime), \
             patch('sia.task_meta.file_lock.exclusive_lock',
                   return_value=contextlib.nullcontext()), \
             patch.object(launch, 'check', side_effect=StopAfterPreflight) as check, \
             patch.object(launch.subprocess, 'Popen') as popen:
            with self.assertRaises(StopAfterPreflight):
                launch.main(['--config', 'fixture.json', '--execute'])
        check.assert_called_once_with(config, runtime, external_readiness=True)
        popen.assert_not_called()

    def test_offline_check_runs_local_validation_without_worker_call(self):
        config = {'validation_config': 'validation.json', 'allocated_tasks': 1080,
                  'output_root': '/tmp/rsiflow_launch_cli_output'}
        names = ['livecodebench', 'humaneval_plus', 'mbpp_plus',
                 'hotpotqa_dev', '2wiki_dev']
        settings = {
            'official_specs': '/fixture/official.json',
            'tool_benchmarks': {
                'acebench': {
                    'user_api_key_env': 'ACE_USER_API_KEY',
                    'user_base_url_env': 'ACE_USER_BASE_URL',
                }
            },
        }
        official = {'evaluators': {name: {} for name in names}}
        pipeline = {'meta': {'api_key_env': 'AUTODL_API_KEY',
                             'remote_worker_token_env': 'RSI_REMOTE_WORKER_TOKEN'}}
        env = {
            'AUTODL_API_KEY': 'set',
            'RSI_REMOTE_WORKER_TOKEN': 'set',
            'ACE_USER_API_KEY': 'set',
            'ACE_USER_BASE_URL': 'set',
        }

        def fake_read(path):
            return settings if str(path).endswith('validation.json') else official

        with patch.object(launch.importlib.util, 'find_spec', return_value=object()), \
             patch('sia.task_meta.sandbox.probe_isolation',
                   return_value={'available': True}), \
             patch('sia.task_meta.reporting.OfficialEvaluatorSpec',
                   side_effect=lambda **kwargs: SimpleNamespace(validate=lambda: ['fixture'])), \
             patch('tool_validation.native_layout',
                   return_value=(Path('/fixture/repo'), {}, {'category': ['id']}, [])), \
             patch('tool_validation.check_native') as check_native, \
             patch('sia.task_meta.pipeline.PipelineConfig.model_validate') as validate, \
             patch('sia.task_meta.meta_backends.remote_execution.validate_worker') as worker, \
             patch.object(launch, 'make_pipeline', return_value=pipeline), \
             patch.object(launch, 'read', side_effect=fake_read), \
             patch.object(launch, 'dry_run',
                          return_value={'status': 'dry_run_no_model_calls'}) as dry_run, \
             patch.object(launch, 'write'), \
             patch.dict(os.environ, env, clear=True):
            validate.return_value.checked.return_value = SimpleNamespace(meta='fixture')
            _, result = launch.check(config, launch.ROOT / 'runtime',
                                     external_readiness=False)

        check_native.assert_called_once()
        dry_run.assert_called_once()
        worker.assert_not_called()
        self.assertEqual(result['status'], 'offline_preflight_passed')
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['network_calls'], 0)
        self.assertFalse(result['started'])
        self.assertFalse(result['external_readiness_checked'])
        self.assertFalse(result['details']['meta_worker']['checked'])

    def test_pause_option_propagates_without_shortening_generations(self):
        config = launch.read(launch.ROOT / 'configs/train.json')
        config['pause_after_round'] = 1
        pipeline = launch.make_pipeline(config, launch.ROOT / 'runtime')
        self.assertEqual(pipeline['max_generations'], 3)
        self.assertEqual(pipeline['round_protocol']['pause_after_round'], 1)
        self.assertEqual(pipeline['training']['round_protocol']['pause_after_round'], 1)


    def test_output_root_propagates_without_moving_frozen_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = launch.read(launch.ROOT / "configs/train.json")
            output_root = Path(tmp) / "Rollout_logs"
            config["output_root"] = str(output_root)
            pipeline = launch.make_pipeline(config, launch.ROOT / "runtime")

        self.assertEqual(Path(pipeline["output_root"]), output_root)
        self.assertEqual(
            Path(pipeline["data_dir"]).resolve(),
            Path(config["frozen_data_dir"]).resolve(),
        )
        self.assertFalse(Path(pipeline["data_dir"]).is_relative_to(output_root))

    def test_zero_exit_pause_then_same_name_completion_updates_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'runtime'
            name = 'fixture_r3'
            run = root / 'runs' / name
            run.mkdir(parents=True)
            marker = run / 'pause_after_round.json'
            marker.write_text('{}')
            final_path = run / 'final_state.json'
            final_path.write_text(json.dumps({
                'status': 'paused',
                'rounds_completed': 1,
                'paused_at_round': 1,
                'pause_marker': str(marker),
            }))
            config = {'rounds': 3, 'pause_after_round': 1, 'output_root': str(root)}
            log = root / 'logs' / 'fixture.log'
            with patch.object(launch, 'ROOT', root):
                paused = launch._finish_child(config, runtime, name, log, 0)
                active = json.loads((root / 'active_run.json').read_text())
                self.assertEqual(active, paused)
                self.assertEqual(active['status'], 'paused')
                self.assertEqual(active['rounds_completed'], 1)
                self.assertEqual(active['pause_marker'], str(marker.resolve()))
                self.assertEqual(active['resume_starts_at_round'], 2)

                final_path.write_text(json.dumps({
                    'status': 'completed',
                    'rounds_completed': 3,
                }))
                completed = launch._finish_child(config, runtime, name, log, 0)
                active = json.loads((root / 'active_run.json').read_text())
            self.assertEqual(active, completed)
            self.assertEqual(active['status'], 'completed')
            self.assertEqual(active['rounds_completed'], 3)
            self.assertEqual(active['run'], paused['run'])

    def test_zero_exit_incomplete_final_state_is_an_error_with_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'runtime'
            name = 'fixture_r3'
            run = root / 'runs' / name
            run.mkdir(parents=True)
            (run / 'final_state.json').write_text(json.dumps({
                'status': 'completed',
                'rounds_completed': 2,
            }))
            with patch.object(launch, 'ROOT', root), \
                 self.assertRaisesRegex(RuntimeError, 'valid paused or completed'):
                launch._finish_child({'rounds': 3, 'pause_after_round': 1, 'output_root': str(root)},
                                     runtime, name, root / 'run.log', 0)
            active = json.loads((root / 'active_run.json').read_text())
            self.assertEqual(active['status'], 'failed')
            self.assertEqual(active['exit_code'], 0)
            self.assertTrue(active['receipts_preserved'])
            self.assertEqual(active['final_status'], 'completed')
            self.assertEqual(active['rounds_completed'], 2)

    def test_check_and_execute_are_mutually_exclusive(self):
        stderr = io.StringIO()
        with patch.object(launch, 'read') as read, \
             contextlib.redirect_stderr(stderr), \
             self.assertRaises(SystemExit):
            launch.main(['--config', 'fixture.json', '--check', '--execute'])
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()

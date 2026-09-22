"""ACE-skipped validation remains explicitly partial and report-only."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))

BENCHMARKS=['bfcl_v3','livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev']


class PartialValidationTests(unittest.TestCase):
    def test_authenticated_partial_receipt_skips_only_completed_a1(self):
        from sia.task_meta.round_validation import validate_round
        from sia.task_meta.storage import digest
        from common import write
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            run=output/'runs'/'fixture';(run/'round_0').mkdir(parents=True)
            report=output/'validation'/'fixture'/'round_01'
            report.mkdir(parents=True)
            validation=output/'configs'/'validation.json';validation.parent.mkdir()
            validation.write_text('{}')
            manifest=output/'manifest.json';manifest.write_text('{}')
            task={'harness_path':str(output/'harness.json')}
            (output/'harness.json').write_text('{}')
            record={'task_after':task,'meta_after':{'bundle_hash':'fixture','version':1},
                    'status':'deployed','chosen_component':'MODEL'}
            write(run/'round_0/selected_update.json',{'child_hash':'task-hash'})
            metrics=report/'task_metrics.json'
            write(metrics,{'overall':{'n_scored':250,'complete':False}})
            partial=report/'code_search_partial_complete.json'
            write(partial,{'status':'code_search_completed_ace_pending','source_role':'independent_validation',
                'feedback_to_meta':False,'task_state_hash':'task-hash',
                'benchmarks_completed':BENCHMARKS,'officially_scored':250,'expected_total':300,
                'metrics_sha256':digest(metrics)})
            for name in BENCHMARKS:
                write(report/'scores'/name/'result.json',
                    {'status':'completed','state_hash':'task-hash','feedback_to_meta':False})
            write(run/'recovery/ace_skip_authorization.json',
                {'schema_version':'ace-skipped-round-validation-v1','run_name':'fixture',
                 'manifest_sha256':digest(manifest),'validation_config_sha256':digest(validation),
                 'expected_total':300,'scored_total':250,
                 'initial_partial_sha256':digest(partial)})
            config=SimpleNamespace(round_validation_config=str(validation),output_root=str(output),
                round_protocol={'validation_manifest':str(manifest)},model_dump=lambda:{})
            with patch('sia.task_meta.evolution_protocol.file_hash',return_value='harness-hash'), \
                 patch('sia.task_meta.evolution_protocol.fingerprint',return_value='config-hash'), \
                 patch('sia.task_meta.durable.load_task',return_value=object()), \
                 patch('sia.task_meta.durable.task_hash',return_value='task-hash'), \
                 patch('sia.task_meta.round_validation.subprocess.run',side_effect=AssertionError('must not rerun')):
                validate_round(config,run,0,record)
                self.assertTrue((report/'round_snapshot.json').is_file())
                self.assertFalse((report/'complete.json').exists())
                changed=json.loads(partial.read_text());changed['officially_scored']=249
                write(partial,changed)
                with self.assertRaises(ValueError):
                    validate_round(config,run,0,record)

    def test_final_test_cannot_skip_ace(self):
        from validate import validate
        with self.assertRaisesRegex(ValueError,'only permitted'):
            validate('unused','unused','unused',role='final_test',skip_ace=True)


if __name__=='__main__':unittest.main()

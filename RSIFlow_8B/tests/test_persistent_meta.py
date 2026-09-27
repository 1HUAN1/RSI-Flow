"""Native RPC and durable-tool integration tests, without a model or GPUs."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_tools import ControllerTools
from experiment_progress import ExperimentProgress
from persistent_meta import (CodexAppServer, ExperimentJobs, NativeExperimentTools,
                             dynamic_tools, native_thread_parameters, process_identity, write_json)


def test_real_process_survives_native_tool_error_and_multiple_turns(tmp_path):
    executable = tmp_path / 'fake_codex'
    executable.write_text('''#!/usr/bin/env python3
import json, sys
def send(obj):
    print(json.dumps(obj), flush=True)
turn=0
calls=0
for line in sys.stdin:
    msg=json.loads(line)
    method=msg.get('method')
    if method=='initialize':
        send({'id':msg['id'],'result':{}})
    elif method=='thread/start':
        send({'id':msg['id'],'result':{'thread':{'id':'thread'}}})
    elif method=='turn/start':
        assert msg['params']['environments'][0]['environmentId']=='local'
        turn+=1
        calls=0
        send({'id':msg['id'],'result':{'turn':{'id':str(turn)}}})
        send({'id':'call-0','method':'item/tool/call','params':{'tool':'probe','arguments':{},'callId':'0'}})
    elif 'result' in msg and str(msg.get('id','')).startswith('call-'):
        calls+=1
        if calls<3:
            send({'id':'call-'+str(calls),'method':'item/tool/call','params':{'tool':'probe','arguments':{},'callId':str(calls)}})
        else:
            send({'method':'turn/completed','params':{'threadId':'thread','turn':{'id':str(turn),'status':'completed'}}})
''')
    executable.chmod(0o755)
    pids = []
    def probe(*args):
        pids.append(server.process.pid)
        if len(pids) == 1:
            raise RuntimeError('recoverable tool error')
        return {'status': 'ok'}
    server = CodexAppServer(codex_home=tmp_path, workspace=tmp_path, journal=tmp_path/'rpc',
                            executable=str(executable), on_tool=probe)
    try:
        server.start()
        server.request('thread/start', {'dynamicTools': dynamic_tools()})
        environments = [{'environmentId':'local','cwd':str(tmp_path)}]
        assert server.turn('thread', 'first', environments=environments)['status'] == 'completed'
        assert server.turn('thread', 'second', environments=environments)['status'] == 'completed'
        assert len(pids) == 6 and len(set(pids)) == 1
        assert server.process.poll() is None
    finally:
        server.close()


def test_durable_job_reuses_process_and_stores_result_separately(tmp_path):
    project = tmp_path/'project'
    project.mkdir()
    (project/'controller_tools.py').write_text('''import json,sys,time
from pathlib import Path
request=json.load(sys.stdin)
while not Path(request['release']).exists(): time.sleep(.02)
print('noisy stdout, not JSON',flush=True)
Path(sys.argv[sys.argv.index('--result-file')+1]).write_text(json.dumps({'operation':'run_parent','status':'completed'}))
''')
    run=tmp_path/'run'
    jobs=ExperimentJobs(project,run)
    request={'operation':'run_parent','release':str(tmp_path/'release')}
    turn=run/'meta_session/turns/0000'
    write_json(turn/'command.json', {'action':'tool','request':request})
    first=jobs.start(request,turn)
    try:
        assert first['status']=='running'
        assert jobs.start(request,turn)['pid']==first['pid']
        assert process_identity(first['pid']) is not None
        # A new bridge attaches to the exact live worker without starting a second rollout.
        migrated=ExperimentJobs(project,run)
        adopted=migrated.adopt_legacy({'status':'tool_inflight','turn':0})
        assert adopted[0]['pid']==first['pid']
        assert migrated.status(first['job_id'])['status']=='running'
        (tmp_path/'release').touch()
        result=migrated.wait(first['job_id'],seconds=5)
        assert result['status']=='completed'
        assert result['result']['status']=='completed'
        assert json.loads((turn/'receipt.json').read_text())['status']=='completed'
        assert 'noisy stdout' in Path(result['stdout_path']).read_text()
    finally:
        child=jobs.children[first['job_id']]
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_running_receipt_does_not_complete_rollout_milestone(tmp_path):
    session=tmp_path/'meta_session'
    turn=session/'turns/0000'
    write_json(turn/'command.json',{'action':'tool','request':{'operation':'run_parent','round_number':1}})
    write_json(turn/'receipt.json',{'operation':'run_parent','status':'running','job_id':'job'})
    progress=ExperimentProgress(session)
    state=progress.rebuild()
    assert 'parent_rollout' not in state['stages'][1]['completed_milestones']
    write_json(turn/'receipt.json',{'operation':'run_parent','status':'completed'})
    assert 'parent_rollout' in progress.rebuild()['stages'][1]['completed_milestones']


def test_native_errors_return_to_same_tools_without_losing_ledger(tmp_path):
    run=tmp_path/'run'
    executor=ControllerTools(workspace=tmp_path,receipt_root=run/'receipts')
    tools=NativeExperimentTools(project=tmp_path,run=run,executor=executor,
                                progress=ExperimentProgress(run/'meta_session'))
    failed=tools('experiment',{'request':{'operation':'read_json','path':str(run/'missing.json')}})
    assert 'tool_error' in failed
    ok=tools('experiment',{'request':{'operation':'write_text','path':str(run/'candidate/helper.py'),'content':'x=1'}})
    assert 'written' in ok and (run/'candidate/helper.py').read_text()=='x=1'
    fixed=tools('experiment',{'request':{'operation':'write_text','path':str(tmp_path/'controller_tools.py'),'content':'# repair'}})
    assert 'written' in fixed and (tmp_path/'controller_tools.py').read_text()=='# repair'
    outside=tools('experiment',{'request':{'operation':'write_text','path':'/outside-rsiflow.py','content':'no'}})
    assert 'tool_error' in outside and 'workspace' in outside
    finish=tools('finish_experiment',{'summary':'premature'})
    assert 'incomplete' in finish and not tools.finished


def test_new_operations_are_forwarded_without_bridge_whitelist(tmp_path):
    class Executor:
        def execute(self, request):
            assert request['operation']=='new_debug_helper'
            return {'status':'diagnosed','detail':'new implementation'}
    run=tmp_path/'run'
    tools=NativeExperimentTools(project=tmp_path,run=run,executor=Executor(),
                                progress=ExperimentProgress(run/'meta_session'))
    result=tools('experiment',{'request':{'operation':'new_debug_helper'}})
    assert 'diagnosed' in result and 'unknown_operation' not in result


def test_native_coding_config_keeps_workspace_and_original_model(tmp_path):
    params=native_thread_parameters(tmp_path)
    assert params['model']=='DeepSeek-V4.1-Flash'
    assert params['sandbox']=='workspace-write' and params['cwd']==str(tmp_path)
    config=params['config']
    assert config['features.shell_tool'] is True
    assert config['features.use_legacy_landlock'] is True
    assert config['sandbox_workspace_write.exclude_slash_tmp'] is True
    assert config['sandbox_workspace_write.exclude_tmpdir_env_var'] is True
    assert 'environments' not in params
    assert 'supersedes earlier' in params['developerInstructions']
    assert 'add tools within the rsiH workspace' in params['developerInstructions']


def test_model_catalog_exposes_native_apply_patch():
    project=Path(__file__).resolve().parents[1]
    catalog=json.loads((project/'configs/codex-deepseek-v4.1-flash.json').read_text())
    model=next(row for row in catalog['models'] if row['slug']=='DeepSeek-V4.1-Flash')
    assert model['shell_type']=='unified_exec'
    assert model['apply_patch_tool_type']=='freeform'


def test_materialize_copies_full_current_parent(tmp_path):
    project=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(project/'runtime'))
    from sia.task_meta.harnessforge_manifest import load_manifest
    seed=project/'seed_harness/harnessforge_base_manifest.json'
    state=tmp_path/'task.json'
    write_json(state,{'harness_path':str(seed)})
    result=ControllerTools(project,receipt_root=tmp_path/'receipts').execute({
        'operation':'materialize_harness','state_path':str(state),'destination':str(tmp_path/'candidate')})
    assert result['status']=='materialized'
    expected=load_manifest(seed)
    assert set(result['files'])==set(expected.files)
    for name,content in expected.files.items():
        assert (tmp_path/'candidate'/name).read_text()==content
    assert [Path(p).name for p in result['production_templates']] == [
        '01_module_localization.yaml', '02_improvement_directions.yaml', '03_harness_generation.yaml']
    assert all(Path(p).is_file() for p in result['production_templates'])
    assert len(result['next_steps']) == 4
    assert result['max_validation_attempts'] == 3
    assert Path(result['localization_report_path']).parent.is_dir()
    assert Path(result['improvement_direction_path']).parent.is_dir()

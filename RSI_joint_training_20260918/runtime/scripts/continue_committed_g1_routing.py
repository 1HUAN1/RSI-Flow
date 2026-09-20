"""Two real Meta operations on frozen T0/T1 evidence; no Task side effects."""
import json, sys, time
from pathlib import Path
from dataclasses import asdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sia.task_meta.pipeline import backend_for, load_config
from sia.task_meta.meta import MetaAgent, experience_input, validate_meta_candidate
from sia.task_meta.meta_harness.five_stage import paired_outcome, materialize
from sia.task_meta.durable import load_task, task_hash
from sia.task_meta.types import MetaAgentState, ImprovementExperience, FiveStageMetaUpdate, EvaluationResult
from sia.task_meta.observations import build_observation
from sia.task_meta.updaters import updater_capabilities
from sia.task_meta.storage import save_json

run=Path(sys.argv[1]);source=Path(sys.argv[2]);config=load_config(ROOT/'configs/meta-reload-only.json')
def read(p):return json.loads(p.read_text())
status={'scope':'continue_real_routing_from_already_committed_G1','phase':'routing_running','started_at':time.time(),'API_cost_usd':None}
save_json(run/'status.json',status)
try:
    from sia.task_meta.meta_harness.bundle import MetaHarnessStore
    from sia.task_meta.meta_backends.codex_openrouter import CodexOpenRouterBackend
    store=MetaHarnessStore(run/'meta');current=store.active().verify()
    assert current.hash=='7f361582e0104341ed1f5ec291db1f2977232ee3891cddbc75bb8d3fb6119424'
    meta=MetaAgentState(config.meta.model,str(current.path/'instructions.md'),current.version,current.hash,str(current.path))
    task=load_task(read(source/'gen_1/task_state.json'));task_out=load_task(read(source/'gen_1/task_state_after_rollout.json'))
    experience=ImprovementExperience(**read(source/'gen_1/improvement_experience.json'))
    fresh=CodexOpenRouterBackend(config.meta,run/'meta',store)
    fresh.bind_context(run.name,1,task_hash(task_out))
    result=EvaluationResult(read(source/'gen_1/results.json'),read(source/'gen_1/agent_execution.json'),
        cost=read(source/'gen_1/cost.json'),evaluated_state=task,output_artifacts=task_out.artifacts,
        artifact_provenance=read(source/'gen_1/artifact_provenance.json'),
        rollout_artifact_diff=read(source/'gen_1/rollout_artifact_diff.json'))
    capabilities={'trainer_configured':True,'sft_profile':'multidomain'}
    capabilities.update(updater_capabilities(task_out,result,True,sft_profile='multidomain'))
    observation=build_observation(task,task_out,meta,result,
        [read(source/f'gen_{i}/results.json') for i in range(2)],[experience],
        [read(source/f'gen_{i}/cost.json') for i in range(2)],capabilities,retain_raw=True)
    save_json(run/'routing_observation.json',asdict(observation))
    decision=MetaAgent(fresh,capabilities).diagnose_and_route(meta,observation)
    save_json(run/'routing_response.json',decision.model_dump(mode='json'))
    operations=[];accepted=[]
    for op in (run/'meta/operations').iterdir():
        result=read(op/'result.json');audit=read(op/'harness_runtime/policy_runtime.json')
        assert read(op/'status.json')['state']=='G_OPERATION_COMPLETED'
        assert audit['status']=='completed' and audit['final_fixed_check']['passed']
        assert result['meta_harness_hash']==current.hash
        assert result['output']==decision.model_dump(mode='json')
        assert any(e.get('status')=='completed' and 'first_turn_activation_audit' in e.get('matched_instruction_rules',[]) for e in audit['events'])
        operations.append(op.name)
    for p in (run/'meta/calls').glob('*/collected.json'):
        c=p.parent;request=read(c/'request.json');assert request['meta_harness_hash']==current.hash
        assert read(c/'status.json')['state']=='META_COLLECTED'
        assert read(c/'bundle_load.json')['runtime_verified']
        for name,content in current.read_files().items():assert (c/'workspace/meta_input/G'/name).read_text()==content
        accepted.append(c.name)
    assert accepted and operations and 'first_turn_activation_audit' in decision.used_principle_ids
    status.update(phase='completed',G1_hash=current.hash,version=current.version,accepted_native_requests=accepted,operations=operations,used_principle_ids=decision.used_principle_ids,routing_action=decision.action.value,requested_targets=[x.target for x in decision.requested_changes],principle_memory_updated=True,operative_G_changed=True,new_G_loaded_and_used=True,performance_improvement_verified=False,ended_at=time.time())
except Exception as exc:
    status.update(phase='failed',error_type=type(exc).__name__,error=str(exc),ended_at=time.time())
    raise
finally:
    save_json(run/'status.json',status);print(json.dumps(status),flush=True)

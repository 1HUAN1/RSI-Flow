"""Trusted pipe-RPC benchmark bridge. Official tools own the outer dialogue.

Each requested assistant message is produced by the frozen, complete Task H
graph. Tool definitions/history are public task input; native official runners
execute tool calls. The user simulator uses a separately bounded relay.
"""
import hashlib
import json
import os
import threading
import queue
import time
from pathlib import Path
from common import read, write

from official_evaluation import OfficialTurn

class HarnessAPI:
    def __init__(self, config, frozen, directory, user_spec):
        from sia.task_meta.durable import load_task
        from sia.task_meta.seed import load_seed
        from sia.task_meta.task_client import LocalTaskClient
        self.config=config; self.frozen=frozen; self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.state=load_task(frozen['task_state']); self.spec=load_seed(self.state.harness_path)
        replicas=getattr(config,'task_replicas',None) or [{'base_url':config.task_base_url}]
        endpoints=[r['base_url'] if isinstance(r,dict) else r.base_url for r in replicas]
        if len(set(endpoints))!=len(endpoints):raise ValueError('Duplicate GPU replica endpoint')
        self.clients=queue.Queue()
        for endpoint in endpoints:
            self.clients.put(LocalTaskClient(self.state,endpoint,timeout=config.task_timeout,enable_thinking=config.task_enable_thinking))
        self.user_spec=user_spec; self.lock=threading.Lock(); self.request_locks={};self.user_lock=threading.Lock()
        self.user_client=None

    def complete(self, body):
        from sia.task_meta.seed import run_seed
        from sia.task_meta.durable import value_hash
        identity={'state_hash':self.frozen['state_hash'],'request':body,'protocol':'official_outer_saved_harness_inner_v1'}
        if self.frozen.get('manifest_hash'):
            identity['evaluation_scope']={k:self.frozen[k] for k in ('manifest_hash','source_role','system_snapshot_sha256','evaluation_config_sha256')}
        key=value_hash(identity); path=self.directory/(key+'.json')
        with self.lock:request_lock=self.request_locks.setdefault(key,threading.Lock())
        with request_lock:
            if path.exists():
                previous=read(path)
                if previous['status']!='completed':raise RuntimeError('Unreconciled benchmark call; no blind retry')
                return previous['response']
            is_user=body['model']==self.user_spec['user_model']
            write(path,{'status':'dispatching','identity':identity,'usage_unknown':True})
            calls=[];client=None;result=None
            try:
                if is_user:
                    from openai import OpenAI
                    with self.user_lock:
                        if len(list(self.directory.glob('user_*.receipt')))>=20000:raise RuntimeError('ACE user simulator request limit reached')
                        if self.user_client is None:
                            self.user_client=OpenAI(api_key=os.environ[self.user_spec['user_api_key_env']],
                                base_url=os.environ[self.user_spec['user_base_url_env']],max_retries=0,timeout=180)
                        write(self.directory/('user_'+key+'.receipt'),{'model':self.user_spec['user_model'],'status':'started'})
                    provider_model=self.user_spec.get('user_provider_model',self.user_spec['user_model'])
                    write(self.directory/('user_'+key+'.receipt'),{'model':self.user_spec['user_model'],'provider_model':provider_model,'status':'started'})
                    response=self.user_client.chat.completions.create(model=provider_model,
                        messages=body['messages'],temperature=0,max_tokens=min(int(body.get('max_tokens',2048)),2048))
                    answer=response.choices[0].message.content or ''
                    usage=response.usage.model_dump() if response.usage else {}
                    usage_unknown=not all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
                else:
                    client=self.clients.get()
                    def model(messages,**kwargs):
                        if len(calls)>=self.config.model_call_limit:raise RuntimeError('Task benchmark call budget exceeded')
                        if kwargs['max_tokens']>self.config.max_output_tokens:raise RuntimeError('Task benchmark token budget exceeded')
                        call_path=self.directory/(key+f'.call_{len(calls):03d}.json')
                        calls.append({'status':'started'})
                        write(call_path,{'status':'dispatching','messages':messages,'parameters':kwargs})
                        response=client(messages,**kwargs)
                        calls[-1]={'status':'completed','usage':response.get('usage')}
                        write(call_path,{'status':'completed','response':response})
                        return response
                    prompt=('Produce the NEXT assistant message for the official benchmark dialogue below. '
                        'The official benchmark owns execution of its tools and will provide real observations on the next turn. '
                        'Return the requested native-format response (including function calls when requested) as your final answer. '
                        'Do not claim that tools have run. Preserve all official response-format instructions.\n'
                        +json.dumps({'messages':body['messages'],'tools':body.get('tools')},ensure_ascii=False))
                    result=run_seed(self.spec,model,OfficialTurn(),prompt,'',self.config.seed)
                    if result.get('infrastructure_failure'):
                        raise RuntimeError('Task infrastructure failure: '+str(result.get('error') or result.get('error_type')))
                    answer=result.get('final_answer') or ''
                    if not isinstance(answer,str):answer=json.dumps(answer,ensure_ascii=False)
                    usage={k:sum(u for c in calls if type(u := (c.get('usage') or {}).get(k)) is int and u >= 0) for k in ['prompt_tokens','completion_tokens']}
                    usage['total_tokens']=sum(usage.values())
                    usage_unknown=any(not all(type((c.get('usage') or {}).get(k)) is int for k in ('prompt_tokens','completion_tokens')) for c in calls)
                response={'id':'chatcmpl-'+key,'object':'chat.completion','created':int(time.time()),'model':body['model'],
                    'choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':'stop'}], 'usage':usage}
                write(path,{'status':'completed','identity':identity,'response':response,'calls':calls,
                            'user_simulator':is_user,'usage_unknown':usage_unknown,
                            'evaluation_status':None if is_user else 'pending_official'})
                return response
            except BaseException as exc:
                write(path,{'status':'requires_audit','identity':identity,'error_type':type(exc).__name__,
                            'error':str(exc),'calls':calls,'usage_unknown':True,
                            'harness_failure':None if result is None else {
                                k:result.get(k) for k in ('error_type','error','infrastructure_failure')}})
                raise
            finally:
                if client is not None:self.clients.put(client)


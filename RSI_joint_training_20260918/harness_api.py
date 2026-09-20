"""Trusted pipe-RPC benchmark bridge. Official tools own the outer dialogue.

Each requested assistant message is produced by the frozen, complete Task H
graph. Tool definitions/history are public task input; native official runners
execute tool calls. The user simulator uses a separately bounded relay.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from common import read, write

class OfficialTurn:
    def tools(self): return []
    def step(self,name,args):raise ValueError('Tool execution belongs to the official outer evaluator')

class HarnessAPI:
    def __init__(self, config, frozen, directory, user_spec):
        from sia.task_meta.durable import load_task
        from sia.task_meta.seed import load_seed
        from sia.task_meta.task_client import LocalTaskClient
        self.config=config; self.frozen=frozen; self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.state=load_task(frozen['task_state']); self.spec=load_seed(self.state.harness_path)
        self.client=LocalTaskClient(self.state,config.task_base_url,timeout=config.task_timeout,enable_thinking=config.task_enable_thinking)
        self.user_spec=user_spec; self.lock=threading.Lock()
        self.user_client=None

    def complete(self, body):
        from sia.task_meta.seed import run_seed
        from sia.task_meta.durable import value_hash
        identity={'state_hash':self.frozen['state_hash'],'request':body,'protocol':'official_outer_saved_harness_inner_v1'}
        key=value_hash(identity); path=self.directory/(key+'.json')
        with self.lock:
            if path.exists():
                previous=read(path)
                if previous['status']!='completed':raise RuntimeError('Unreconciled benchmark call; no blind retry')
                return previous['response']
            is_user=body['model']==self.user_spec['user_model']
            if is_user and len(list(self.directory.glob('user_*.receipt')))>=20000:raise RuntimeError('ACE user simulator request limit reached')
            write(path,{'status':'dispatching','identity':identity,'usage_unknown':True})
            calls=[]
            try:
                if is_user:
                    from openai import OpenAI
                    if self.user_client is None:
                        self.user_client=OpenAI(api_key=os.environ[self.user_spec['user_api_key_env']],
                            base_url=os.environ[self.user_spec['user_base_url_env']],max_retries=0,timeout=180)
                    provider_model=self.user_spec.get('user_provider_model',self.user_spec['user_model'])
                    write(self.directory/('user_'+key+'.receipt'),{'model':self.user_spec['user_model'],'provider_model':provider_model,'status':'started'})
                    response=self.user_client.chat.completions.create(model=provider_model,
                        messages=body['messages'],temperature=0,max_tokens=min(int(body.get('max_tokens',2048)),2048))
                    answer=response.choices[0].message.content or ''
                    usage=response.usage.model_dump() if response.usage else {}
                    usage_unknown=not all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
                else:
                    def model(messages,**kwargs):
                        if len(calls)>=self.config.model_call_limit:raise RuntimeError('Task benchmark call budget exceeded')
                        if kwargs['max_tokens']>self.config.max_output_tokens:raise RuntimeError('Task benchmark token budget exceeded')
                        call_path=self.directory/(key+f'.call_{len(calls):03d}.json')
                        calls.append({'status':'started'})
                        write(call_path,{'status':'dispatching','messages':messages,'parameters':kwargs})
                        response=self.client(messages,**kwargs)
                        calls[-1]={'status':'completed','usage':response.get('usage')}
                        write(call_path,{'status':'completed','response':response})
                        return response
                    prompt=('Produce the NEXT assistant message for the official benchmark dialogue below. '
                        'The official benchmark owns execution of its tools and will provide real observations on the next turn. '
                        'Return the requested native-format response (including function calls when requested) as your final answer. '
                        'Do not claim that tools have run. Preserve all official response-format instructions.\n'
                        +json.dumps({'messages':body['messages'],'tools':body.get('tools')},ensure_ascii=False))
                    result=run_seed(self.spec,model,OfficialTurn(),prompt,'',self.config.seed)
                    if result.get('infrastructure_failure'):raise RuntimeError('Task infrastructure failure')
                    answer=result.get('final_answer') or ''
                    if not isinstance(answer,str):answer=json.dumps(answer,ensure_ascii=False)
                    usage={k:sum(u for c in calls if type(u := (c.get('usage') or {}).get(k)) is int and u >= 0) for k in ['prompt_tokens','completion_tokens']}
                    usage['total_tokens']=sum(usage.values())
                    usage_unknown=any(not all(type((c.get('usage') or {}).get(k)) is int for k in ('prompt_tokens','completion_tokens')) for c in calls)
                response={'id':'chatcmpl-'+key,'object':'chat.completion','created':int(time.time()),'model':body['model'],
                    'choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':'stop'}], 'usage':usage}
                write(path,{'status':'completed','identity':identity,'response':response,'calls':calls,
                            'user_simulator':is_user,'usage_unknown':usage_unknown})
                return response
            except BaseException as exc:
                write(path,{'status':'requires_audit','identity':identity,'error_type':type(exc).__name__,'calls':calls,'usage_unknown':True})
                raise


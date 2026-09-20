"""Official benchmark entrypoints with a registered frozen RSI model endpoint.

This file runs inside the tool benchmark sandbox. It has only inherited RPC
pipes, never a Meta/provider key. Evaluation code and answer files are unchanged.
"""
import copy
import dataclasses
import json
import os
import runpy
import sys
import threading
from concurrent.futures import Future
from itertools import count
from pathlib import Path
from types import SimpleNamespace

def sdk_bridge():
    import openai
    lock=threading.Lock();pending={};sequence=count()
    incoming=os.fdopen(int(os.environ['RSI_RPC_READ_FD']),'rb',closefd=False)
    outgoing=os.fdopen(int(os.environ['RSI_RPC_WRITE_FD']),'wb',closefd=False)
    def receive():
        try:
            while line:=incoming.readline(8_000_001):
                if len(line)>8_000_000 or not line.endswith(b'\n'):raise RuntimeError('RPC transport failed')
                result=json.loads(line)
                with lock:future=pending.pop(result['rpc_id'])
                future.set_result(result)
            raise RuntimeError('RPC transport closed')
        except BaseException as exc:
            with lock:
                for future in pending.values():future.set_exception(exc)
                pending.clear()
    threading.Thread(target=receive,daemon=True).start()
    def plain(value):
        if hasattr(value,'model_dump'):return plain(value.model_dump(exclude_none=True))
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    class Completions:
        def create(self,**kwargs):
            from openai.types.chat import ChatCompletion
            if kwargs.get('stream'):raise ValueError('Streaming is not supported')
            with lock:
                rpc_id=next(sequence);future=Future()
                raw=json.dumps({'rpc_id':rpc_id,'request':plain(kwargs)}).encode()+b'\n'
                if len(raw)>8_000_000:raise ValueError('RPC request too large')
                pending[rpc_id]=future
                outgoing.write(raw);outgoing.flush()
            result=future.result()
            if 'error' in result:raise RuntimeError('Frozen model RPC failed: '+result['error'])
            return ChatCompletion.model_validate(result['response'])
    class Responses:
        def __init__(self,client):self.client=client
        def create(self,**kwargs):
            from openai.types.responses import Response
            inputs=kwargs.pop('input')
            messages=inputs if isinstance(inputs,list) else [{'role':'user','content':inputs}]
            if kwargs.get('instructions'):messages=[{'role':'system','content':kwargs['instructions']}]+messages
            result=self.client.chat.completions.create(model=kwargs['model'],messages=messages,
                max_tokens=kwargs.get('max_output_tokens',2048),temperature=kwargs.get('temperature',0))
            return Response.model_validate({'id':'resp_'+result.id,'created_at':result.created,'object':'response',
                'model':result.model,'status':'completed','parallel_tool_calls':False,'tool_choice':'none','tools':[],
                'output':[{'id':'msg_'+result.id,'type':'message','status':'completed','role':'assistant',
                           'content':[{'type':'output_text','text':result.choices[0].message.content,'annotations':[]}]}],
                'usage':{'input_tokens':result.usage.prompt_tokens,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},
                         'output_tokens':result.usage.completion_tokens,'output_tokens_details':{'reasoning_tokens':0},
                         'total_tokens':result.usage.total_tokens}})
    class BoundClient:
        def __init__(self,*args,**kwargs):
            self.chat=SimpleNamespace(completions=Completions());self.responses=Responses(self)
        def close(self):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    openai.OpenAI=BoundClient

def register_bfcl(alias):
    import bfcl_eval.constants.model_config as config
    mappings=[v for v in vars(config).values() if isinstance(v,dict) and any(str(k).startswith('gpt-4o') and not str(k).endswith('FC') for k in v)]
    if not mappings:raise RuntimeError('Unsupported BFCL model registry; no compatible OpenAI prompt handler')
    for mapping in mappings:
        key=next(k for k in mapping if str(k).startswith('gpt-4o') and not str(k).endswith('FC'))
        item=copy.copy(mapping[key])
        if dataclasses.is_dataclass(item):
            fields={f.name for f in dataclasses.fields(item)}
            changes={k:v for k,v in {'model_name':alias,'display_name':'RSI frozen Task','org':'RSI','is_fc_model':False,
                                     'input_price':None,'output_price':None}.items() if k in fields}
            item=dataclasses.replace(item,**changes)
        elif isinstance(item,dict):
            item=copy.deepcopy(item)
            for k,v in [('model_name',alias),('display_name','RSI frozen Task'),('is_fc_model',False)]:
                if k in item:item[k]=v
        else:raise RuntimeError('Unsupported BFCL model configuration type')
        mapping[alias]=item

def main():
    benchmark,phase,alias,repo,output,spec_path=sys.argv[1:7]
    spec=json.loads(Path(spec_path).read_text())
    os.chdir(repo);sys.path.insert(0,repo)
    sdk_bridge()
    if phase=='check':
        import openai
        client=openai.OpenAI()
        first=client.responses.create(model=alias,input='test_override_cpu_only')
        second=client.responses.create(model=alias,input=first.output)
        assert first.output_text==second.output_text=='test_override_cpu_only'
        assert second.usage.input_tokens==1
    if benchmark=='bfcl_v3':
        from bfcl_eval.constants.category_mapping import VERSION_PREFIX
        if VERSION_PREFIX != spec['version_prefix']:
            raise RuntimeError('BFCL source version differs from the requested v3 data')
        register_bfcl(alias)
        from bfcl_eval.__main__ import cli
        if phase=='check':
            for command in ('generate','evaluate'):
                sys.argv=['bfcl',command,'--help']
                try:cli()
                except SystemExit as exc:
                    if exc.code not in (None,0):raise
            return
        command=['bfcl',phase,'--model',alias,'--test-category',','.join(spec['categories']),
                 '--result-dir',str(Path(output)/'result')]
        if phase=='evaluate':command+=['--score-dir',str(Path(output)/'score')]
        else:command+=['--num-threads',str(spec.get('num_threads',4))]
        sys.argv=command;cli()
    else:
        from model_inference.inference_map import inference_map
        from model_inference.apimodel_inference import APIModelInference
        inference_map[alias]=APIModelInference
        if phase=='check':
            for script in ('generate.py','eval_main.py'):
                sys.argv=[script,'--help']
                try:runpy.run_path(str(Path(repo)/script),run_name='__main__')
                except SystemExit as exc:
                    if exc.code not in (None,0):raise
            return
        script='generate.py' if phase=='generate' else 'eval_main.py'
        sys.argv=[script,'--model',alias,'--category',spec['category'],'--language',spec['language'],'--output-dir',output]
        if phase=='generate':sys.argv+=['--num-threads',str(spec.get('num_threads',4)),'--temperature','0','--user-model',spec['user_model']]
        runpy.run_path(str(Path(repo)/script),run_name='__main__')

if __name__=='__main__':main()

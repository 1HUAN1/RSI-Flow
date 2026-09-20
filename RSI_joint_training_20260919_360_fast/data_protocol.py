"""Frozen group-disjoint role allocation over real records; no model dependencies."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from evolution_protocol import DOMAINS, TRAIN_QUOTAS, assert_disjoint, file_hash, fingerprint, freeze, make_manifest, read


def text_hash(text):
    return hashlib.sha256(re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text).casefold()).strip().encode()).hexdigest()


def records(path):
    path=Path(path)
    with path.open('rb') as stream: array=stream.read(256).lstrip().startswith(b'[')
    if array:
        value=read(path)
        if not isinstance(value,list): raise ValueError('Expected a JSON array: '+str(path))
        for index,row in enumerate(value): yield index,None,row
    else:
        with path.open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                if line.strip():yield offset,len(line),json.loads(line)


def public_text(row, source):
    keys=('question_content','prompt','problem','question','task','instruction')
    for key in keys:
        value=row.get(key)
        if isinstance(value,str) and value.strip():
            return value+('\n'+row['starter_code'] if key=='question_content' and row.get('starter_code') else '')
        if key=='question' and isinstance(value,(list,dict)):
            return json.dumps(value,ensure_ascii=False,sort_keys=True)
    for key in ('messages','conversation','conversations'):
        if isinstance(row.get(key),list):return json.dumps(row[key],ensure_ascii=False,sort_keys=True)
    raise ValueError('No supported public prompt field in '+source)


def native_id(row, source, qhash):
    value=next((row[k] for k in ('task_id','question_id','id','_id') if row.get(k) is not None),qhash)
    if source=='envscaler':
        env=row.get('canonical_env_id',row.get('env_id'))
        if env is None:raise ValueError('EnvScaler lacks environment identity')
        return str(env)+':'+str(value)
    return str(value)


def catalog(config, evaluation):
    root=Path(config['dataset_root']);sources=[]
    for spec in config['datasets']:
        for part in spec['files']:
            sources.append(dict(source=spec['name'],domain=spec['domain'],kind='train',
                path=str(root/part['path']),original_split=part['original_split'],
                version=part.get('version'),stratify=spec.get('stratify',[])))
    specs=read(evaluation['official_specs'])['evaluators']
    for name in evaluation['benchmark_ids']:
        base=dict(source=name,domain='tool_use' if name in evaluation['tool_benchmarks'] else 'searchqa' if name.endswith('_dev') else 'code',
                  kind='evaluation',original_split='dev' if name.endswith('_dev') else 'test',stratify=['difficulty','type','category','question_type'])
        if name in evaluation['tool_benchmarks']:
            from tool_validation import native_layout
            repo,_,_,paths=native_layout(name,evaluation['tool_benchmarks'][name])
            for p in paths:sources.append(dict(base,path=str(p),category=p.stem,version='file_sha256'))
        else:
            s=specs[name]
            sources.append(dict(base,path=s['data_path'],version=s.get('release_version') or s['commit'],
                evaluator_commit=s['commit'],evaluator_sha256=s['entrypoint_sha256'],feedback_priority=False))
    return sources


def scan(sources, grouping):
    """Retain metadata/byte offsets only; never load the full candidate pool for rollout."""
    items=[];stats=[];parent=[];tokens={};duplicates=[];seen_ids={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for spec in sources:
        path=Path(spec['path'])
        if not path.is_file():raise FileNotFoundError('Required real dataset: '+str(path))
        digest=file_hash(path);count=0;missing=Counter();dates=[]
        for offset,length,row in records(path):
            if not isinstance(row,dict):raise ValueError('Dataset row must be an object')
            prompt=public_text(row,spec['source']);qhash=text_hash(prompt)
            ident=native_id(row,spec['source'],qhash);task_id=spec['source']+':'+ident
            count+=1
            if task_id in seen_ids:
                if seen_ids[task_id]!=qhash:raise ValueError('Conflicting duplicate native task ID: '+task_id)
                duplicates.append(dict(source=spec['source'],task_id=task_id,reason='duplicate_id'));continue
            seen_ids[task_id]=qhash
            strata={}
            metadata=row.get('metadata',{})
            if isinstance(metadata,str):
                try:metadata=json.loads(metadata)
                except ValueError:metadata={}
            if not isinstance(metadata,dict):metadata={}
            for field in spec.get('stratify',[]):
                value=row.get(field,metadata.get(field))
                if value is None:missing[field]+=1
                elif isinstance(value,(str,int,float,bool)):strata[field]=value
            if spec.get('category'):strata['category']=spec['category']
            group_tokens=['question:'+qhash]
            for field in grouping['family_fields']:
                if row.get(field) is not None:
                    group_tokens.append(spec['source']+':'+field+':'+str(row[field]))
            if spec['source']=='envscaler' and grouping['envscaler_environment_family']:
                env=str(row.get('canonical_env_id',row.get('env_id')))
                group_tokens.append('envscaler:environment:'+env);strata['environment']=env
            if spec['source'] in {'humaneval_plus','mbpp_plus'}:
                group_tokens.append(spec['source']+':problem:'+ident.split('/')[-1])
            if spec['source']=='acebench' and ident.startswith('normal_multi_turn_'):
                # Native ACE IDs end in conversation_number / turn_number.
                group_tokens.append('acebench:conversation:'+ident.rsplit('_',1)[0])
            date=row.get('contest_date',row.get('date'))
            if date:dates.append(str(date))
            index=len(items);parent.append(index)
            for token in group_tokens:
                if token in tokens:
                    left,right=find(index),find(tokens[token]);parent[max(left,right)]=min(left,right)
                else:tokens[token]=index
            items.append(dict(source=spec['source'],domain=spec['domain'],kind=spec['kind'],
                version=spec.get('version') or 'sha256:'+digest,source_sha256=digest,
                original_split=spec['original_split'],task_id=task_id,native_id=ident,content_hash=qhash,
                strata=strata,path=str(path.resolve()),offset=offset,length=length,
                feedback_priority=bool(spec.get('feedback_priority')),date=str(date) if date else None))
        stats.append(dict(source=spec['source'],path=str(path),sha256=digest,original_split=spec['original_split'],
            raw_records=count,missing_labels=dict(missing),date_range=[min(dates),max(dates)] if dates else None))
    groups=defaultdict(list)
    for i in range(len(items)):groups[find(i)].append(i)
    for indices in groups.values():
        group=fingerprint(sorted(items[i]['task_id'] for i in indices))
        for i in indices:items[i]['group_id']=group
    raw_count=len(items)
    items=complete_tool_tasks(items)
    return items,dict(files=stats,duplicate_ids=duplicates,valid_unique_native_ids=raw_count,
        valid_unique_ids=len(items),group_count=len(groups),
        overlap_method='NFKC/casefold/whitespace public-text equality + explicit family/dialogue IDs + environment families; transitive union',
        limitations=['No semantic paraphrase guarantee','Unlabeled variants may remain undetected','No claim about model pretraining contamination'])


def complete_tool_tasks(items):
    """A native ACE multi-turn conversation is one allocated task, never one turn."""
    families=defaultdict(list);result=[]
    for row in items:
        if row['source']=='acebench' and row['native_id'].startswith('normal_multi_turn_'):
            families[row['native_id'].rsplit('_',1)[0]].append(row)
        else:result.append(row)
    for family,parts in sorted(families.items()):
        parts=sorted(parts,key=lambda x:int(x['native_id'].rsplit('_',1)[1]))
        if len({p['path'] for p in parts})!=1:raise ValueError('ACE conversation split across source files')
        result.append({**parts[0],'native_id':family,'task_id':'acebench:'+family,
            'native_ids':[p['native_id'] for p in parts],
            'record_parts':[{k:p[k] for k in ('offset','length','native_id','content_hash')} for p in parts],
            'content_hash':fingerprint([p['content_hash'] for p in parts]),'task_unit':'complete_conversation'})
    return result


def balanced_sample(rows, count, seed, *, cap=None):
    strata=defaultdict(list)
    for row in rows:strata[fingerprint(row['strata'])].append(row)
    for key in strata:strata[key]=deque(sorted(strata[key],key=lambda r:fingerprint([seed,r['task_id']])))
    keys=sorted(strata,key=lambda k:fingerprint([seed,k]));selected=[];seen=set();groups=Counter()
    while len(selected)<count:
        progressed=False
        for key in keys:
            while strata[key]:
                row=strata[key].popleft()
                if row['content_hash'] in seen or (cap and groups[row['group_id']]>=cap):continue
                selected.append(row);seen.add(row['content_hash']);groups[row['group_id']]+=1;progressed=True;break
            if len(selected)==count:break
        if not progressed:raise ValueError(f'Quota shortage after dedup/group constraints: requested={count}, eligible={len(selected)}, deficit={count-len(selected)}')
    return selected


def exposure(row, ledger):
    states=[]
    for event in ledger.get('events',[]):
        if event.get('source')!=row['source']:continue
        if (event.get('all_tasks') or row['task_id'] in event.get('task_ids',[]) or
            row['native_id'] in event.get('task_ids',[]) or
            any(i in event.get('task_ids',[]) or row['source']+':'+i in event.get('task_ids',[]) for i in row.get('native_ids',[])) or
            row['group_id'] in event.get('group_ids',[]) or
            row['content_hash'] in event.get('content_hashes',[])):
            if not event.get('evidence'):raise ValueError('Exposure claim requires historical evidence')
            states.append(event.get('status','exposed'))
    if 'exposed' in states:return 'exposed'
    coverage=ledger.get('coverage',{}).get(row['source'],{})
    if coverage.get('complete') and coverage.get('evidence') and coverage.get('source_sha256')==row['source_sha256']:
        return 'unexposed' if not states else states[-1]
    return 'exposure_unknown'


def allocate(items, config, ledger):
    seed=config['split_seed'];training=[r for r in items if r['kind']=='train'];evaluation=[r for r in items if r['kind']=='evaluation']
    # A family touching any evaluation source cannot enter train_evolution.
    evaluation_groups={r['group_id'] for r in evaluation}
    training=[r for r in training if r['group_id'] not in evaluation_groups]
    group_rows=defaultdict(list)
    for row in training:group_rows[row['group_id']].append(row)
    bins=[[] for _ in range(3)];loads=[Counter() for _ in range(3)]
    # Assign whole families before sampling; no variants cross round boundaries.
    for group,rows in sorted(group_rows.items(),key=lambda pair:(-len(pair[1]),fingerprint([seed,pair[0]]))):
        weights=Counter(r['source'] for r in rows)
        r=min(range(3),key=lambda i:(sum(loads[i][s]/config['train_quotas_per_round'][s] for s in weights),i))
        bins[r].extend(rows);loads[r].update(weights)
    manifests=[]
    for r,rows in enumerate(bins,1):
        selected=[]
        for source,count in config['train_quotas_per_round'].items():
            cap=max(1,int(count*config['grouping']['max_environment_fraction_per_round'])) if source=='envscaler' else None
            selected+=balanced_sample([x for x in rows if x['source']==source],count,seed+r,cap=cap)
        tasks=[dict(x,role='train_evolution',purpose='evolution_train',round_id=r) for x in selected]
        manifests.append(make_manifest('train_evolution',r,tasks,split_seed=seed,feedback_scope='all_B_r'))
    # A group's most conservative history status applies to every variant.
    status={};priority={'unexposed':0,'exposure_unknown':1,'exposed':2}
    for row in evaluation:
        state=exposure(row,ledger);old=status.get(row['group_id'],'unexposed')
        status[row['group_id']]=max((old,state),key=lambda s:priority[s])
    validation=[];validation_groups=set()
    for source,limit in config['validation_limits'].items():
        candidates=[r for r in evaluation if r['source']==source]
        # No inherited cap means the existing full source, with its size previewed
        # by the reporting entry before --execute. Do not reuse old Meta-dev quotas.
        if type(limit) is not int or limit < 1: raise ValueError('Independent validation needs an explicit positive quota')
        selected=balanced_sample(candidates,limit,seed+9000)
        validation.extend(selected);validation_groups.update(r['group_id'] for r in selected)
    final=[];quarantine=[]
    for row in evaluation:
        if row['group_id'] in validation_groups:continue
        marked=dict(row,exposure=status[row['group_id']])
        (final if marked['exposure']=='unexposed' else quarantine).append(marked)
    manifests.append(make_manifest('independent_validation',None,[dict(x,role='independent_validation',purpose='report_only',round_id=None,exposure=status[x['group_id']]) for x in validation],split_seed=seed,fixed_across_rounds=True,
        clean_independence_audited=all(status[x['group_id']]=='unexposed' for x in validation)))
    manifests.append(make_manifest('final_test',None,[dict(x,role='final_test',round_id=None) for x in final],split_seed=seed,mode='full_remaining',clean_independence_audited=True))
    assert_disjoint(manifests)
    return manifests,quarantine


def materialize(value, destination):
    """Copy ONLY selected records into role-local packs. Do not expose full source paths to updates."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    tasks=[];by_source=defaultdict(list)
    for row in value['tasks']:by_source[row['source']].append(row)
    for source,rows in by_source.items():
        target=destination/(source+'.jsonl')
        if target.exists():raise FileExistsError('Prepared pack already exists; use frozen manifests')
        arrays={}
        with target.open('xb') as out:
            for row in sorted(rows,key=lambda x:x['task_id']):
                p=Path(row['path'])
                if row.get('record_parts'):
                    source_records=read(p) if row['record_parts'][0]['length'] is None else None
                    parts=[]
                    for part in row['record_parts']:
                        if source_records is not None:member=source_records[part['offset']]
                        else:
                            with p.open('rb') as f:f.seek(part['offset']);member=json.loads(f.read(part['length']))
                        if text_hash(public_text(member,source=row['source']))!=part['content_hash']:
                            raise ValueError('Conversation member changed')
                        parts.append(member)
                    raw={'id':row['native_id'],'native_records':parts}
                elif row['length'] is None:
                    if str(p) not in arrays:arrays[str(p)]=read(p)
                    raw=arrays[str(p)][row['offset']]
                else:
                    with p.open('rb') as f:f.seek(row['offset']);raw=json.loads(f.read(row['length']))
                if not row.get('record_parts') and text_hash(public_text(raw,source))!=row['content_hash']:raise ValueError('Source record changed during packing')
                line=(json.dumps(raw,ensure_ascii=False)+'\n').encode();offset=out.tell();out.write(line)
                tasks.append({**{k:v for k,v in row.items() if k not in ('path','offset','length','kind','feedback_priority','record_parts')},
                    'record_file':str(target.resolve()),'record_offset':offset,'record_length':len(line)})
        digest=file_hash(target)
        for task in tasks:
            if task['source']==source:task['record_sha256']=digest
    return make_manifest(value['role'],value['round_id'],tasks,
        **{k:v for k,v in value.items() if k not in ('role','round_id','tasks','manifest_hash','schema_version')})


def prepare_protocol(config, evaluation, output, final_root):
    output=Path(output).resolve();final_root=Path(final_root).resolve()
    if output==final_root or output in final_root.parents or final_root in output.parents:
        raise ValueError('Final-test vault and evolution data must have separate roots')
    if output.exists() or final_root.exists():raise FileExistsError('Never overwrite a prepared data release')
    if config['rounds']!=3 or config['train_quotas_per_round']!=TRAIN_QUOTAS:
        raise ValueError('This release fixes three rounds and the registered source quotas')
    corpus=Path(config['retrieval_index'])
    if not corpus.is_file() or not corpus.with_suffix('.sqlite.manifest.json').is_file():
        raise FileNotFoundError('Pinned retrieval index and manifest required before preparation')
    sources=catalog(config,evaluation);items,audit=scan(sources,config['grouping'])
    ledger=read(config['exposure_ledger']) if config.get('exposure_ledger') else {'coverage':{},'events':[]}
    manifests,quarantine=allocate(items,config,ledger)
    # Allocation is fully checked before writing any release files.
    output.mkdir(parents=True);final_root.mkdir(parents=True,mode=0o700)
    published=[]
    for value in manifests:
        role=value['role'];r=value['round_id']
        directory=output/f'B{r}' if role=='train_evolution' else final_root/('full' if role=='final_test' else 'validation')
        packed=materialize(value,directory);freeze(directory/'manifest.json',packed);published.append(packed)
    final=published[-1];limited=[];missing_limits=[]
    for source in evaluation['benchmark_ids']:
        rows=[r for r in final['tasks'] if r['source']==source]
        limit=config['final_test_limits'].get(source)
        if limit is None:missing_limits.append(source);continue
        limited+=sorted(rows,key=lambda r:fingerprint([config['split_seed'],r['task_id']]))[:limit]
    freeze(final_root/'budget_limited_manifest.json',make_manifest('final_test',None,limited,mode='budget_limited',
        full_manifest_hash=final['manifest_hash'],unconfigured_limits=missing_limits,clean_independence_audited=True))
    freeze(final_root/'quarantine.json',{'tasks':[{k:r[k] for k in ('source','task_id','group_id','content_hash','exposure')} for r in quarantine]})
    counts=[{'role':m['role'],'round_id':m['round_id'],'tasks':len(m['tasks']),
             'by_source':dict(Counter(r['source'] for r in m['tasks'])),'manifest_hash':m['manifest_hash']} for m in published]
    report=dict(status='prepared_not_executed',allocated_tasks=sum(len(m['tasks']) for m in published[:3]),executed_unique_tasks=0,rollout_attempts=0,
        manifests=counts,source_audit=audit,historical_exposure_counts=dict(Counter(r['exposure'] for r in quarantine)),
        clean_final_test_available=bool(final['tasks']),final_test_missing_sources=[s for s in evaluation['benchmark_ids'] if not any(r['source']==s for r in final['tasks'])],
        final_test_missing_limits=missing_limits,exposure_audit_sha256=fingerprint(ledger),
        official_complete_benchmark_claim=False,split_seed=config['split_seed'],rollout_seed=config['rollout_seed'],training_seed=config['training_seed'])
    # Only hashes/counts of the final vault are published to preparation audit; never copy its contents into training data.
    freeze(output/'preparation_audit.json',report);freeze(output/'protocol.json',config)
    native_index(output, published[:3], config)
    return report


def native_index(output, rounds, config):
    """Native ManifestStore index over frozen packs, not the raw candidate pool."""
    import sqlite3
    import shutil
    import sys
    installed=Path(__file__).resolve().parent/'runtime'
    runtime=installed if (installed/'sia/task_meta/data.py').is_file() else Path(config['runtime_source'])
    if str(runtime) not in sys.path: sys.path.insert(0,str(runtime))
    from sia.task_meta.data import content_hash, _prompt
    db=Path(output)/'tasks.sqlite'
    if db.exists(): raise FileExistsError('Native index already exists')
    conn=sqlite3.connect(db)
    try:
        conn.executescript('''CREATE TABLE sources(path TEXT PRIMARY KEY, hash TEXT, metadata TEXT);
          CREATE TABLE tasks(task_id TEXT PRIMARY KEY, domain TEXT, source TEXT, split TEXT,
            content_hash TEXT, group_key TEXT, path TEXT, offset INTEGER, length INTEGER, order_key TEXT, seq INTEGER);
          CREATE TABLE probe(task_id TEXT PRIMARY KEY); CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE omissions(source TEXT,task_id TEXT,reason TEXT);''')
        seq=Counter()
        for value in rounds:
            for row in value['tasks']:
                path=row['record_file']
                with Path(path).open('rb') as f:
                    f.seek(row['record_offset']); raw=json.loads(f.read(row['record_length']))
                conn.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?)',(path,row['record_sha256'],json.dumps({'role':'train_evolution','round_id':value['round_id']})))
                conn.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',(row['task_id'],row['domain'],row['source'],
                    'evolve_train',content_hash(_prompt(raw,row['domain'])),row['group_id'],path,row['record_offset'],row['record_length'],row['task_id'],seq[row['domain']]))
                seq[row['domain']]+=1
                conn.execute('INSERT INTO probe VALUES(?)',(row['task_id'],))
        conn.commit()
    finally: conn.close()
    # Reuse only an explicitly pinned public retrieval corpus; never index held-out answers.
    corpus=Path(config['retrieval_index'])
    if not corpus.is_file() or not corpus.with_suffix('.sqlite.manifest.json').is_file():
        raise FileNotFoundError('Pinned public retrieval index and its manifest are required')
    digest=file_hash(corpus)
    if config.get('retrieval_index_sha256') and digest!=config['retrieval_index_sha256']:
        raise ValueError('Retrieval index changed')
    shutil.copy2(corpus,Path(output)/'search.sqlite')
    details=read(corpus.with_suffix('.sqlite.manifest.json'))
    freeze(Path(output)/'search.sqlite.manifest.json',{**details,'frozen_index_sha256':digest})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--evaluation-config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--final-root')
    p.add_argument('--full-report-only',action='store_true',help='Prepare all official reporting tasks only; never run a model')
    a=p.parse_args();config=read(a.config);evaluation=read(a.evaluation_config)
    if a.full_report_only:
        out=Path(a.output)
        if out.exists():raise FileExistsError('Never overwrite a reporting release')
        sources=[s for s in catalog(config,evaluation) if s['kind']=='evaluation']
        rows,audit=scan(sources,config['grouping'])
        # Full report is not a fresh clean-test claim, including for historical exposure_unknown.
        packed=materialize(make_manifest('independent_validation',None,[dict(r,role='independent_validation',
            round_id=None,purpose='report_only',exposure='exposure_unknown') for r in rows],
            mode='explicit_full_reporting',clean_independence_audited=False),out)
        freeze(out/'manifest.json',packed);freeze(out/'source_audit.json',audit)
        result=dict(status='prepared_not_executed',tasks=len(rows),by_source=dict(Counter(r['source'] for r in rows)),manifest_hash=packed['manifest_hash'])
    else:
        if not a.final_root:p.error('--final-root is required for training preparation')
        result=prepare_protocol(config,evaluation,a.output,a.final_root)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

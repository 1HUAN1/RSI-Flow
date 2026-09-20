"""Build a full training manifest. Official validation files never become SFT rows."""
import hashlib
import json
import sqlite3
from pathlib import Path
from common import read, write, sha

def prepare(config, runtime):
    from sia.task_meta.data import DOMAINS, ManifestStore, jsonl_rows, _identity
    from sia.task_meta.retrieval import build_search_index
    out = Path(runtime)/'data/joint_full'
    out.mkdir(parents=True,exist_ok=True)
    frozen = {'datasets':config['datasets'],'dataset_root':config['dataset_root'],
              'environment_files':config.get('environment_files',[]),
              'seed':config['seed'],'probe_per_domain':config['training_probe_per_domain']}
    db=out/'tasks.sqlite'
    if db.exists():
        if read(out/'input_protocol.json') != frozen: raise ValueError('Training data configuration changed')
        store=ManifestStore(db)
        try: store.validate_sources()
        finally: store.close()
    else:
        tmp=out/'tasks.sqlite.building'
        if tmp.exists(): raise RuntimeError('Incomplete manifest exists; inspect '+str(tmp))
        conn=sqlite3.connect(tmp)
        conn.executescript('''
          CREATE TABLE sources(path TEXT PRIMARY KEY, hash TEXT, metadata TEXT);
          CREATE TABLE tasks(task_id TEXT PRIMARY KEY, domain TEXT, source TEXT, split TEXT,
            content_hash TEXT, group_key TEXT, path TEXT, offset INTEGER, length INTEGER, order_key TEXT, seq INTEGER);
          CREATE TABLE probe(task_id TEXT PRIMARY KEY);
          CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE omissions(source TEXT,task_id TEXT,reason TEXT);
        ''')
        counts={}; domains=dict.fromkeys(DOMAINS,0)
        try:
            for spec in config['datasets']:
                count=0
                for relative in spec['paths']:
                    path=(Path(config['dataset_root'])/relative).resolve()
                    conn.execute('INSERT INTO sources VALUES(?,?,?)',(str(path),sha(path),json.dumps(spec)))
                    for offset,length,row in jsonl_rows(path):
                        task_id,qhash=_identity(row,spec['domain'],spec['name'])
                        order=hashlib.sha256(f"{config['seed']}:{task_id}".encode()).hexdigest()
                        # Native IDs are retained. Duplicate IDs fail instead of reducing the denominator.
                        conn.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                            (task_id,spec['domain'],spec['name'],'evolve_train',qhash,
                             str(row.get('canonical_env_id',row.get('env_id',qhash))),str(path),offset,length,
                             order,domains[spec['domain']]))
                        domains[spec['domain']]+=1; count+=1
                if count != spec['count']:raise ValueError(f"{spec['name']}: expected {spec['count']}, found {count}")
                counts[spec['name']]=count
            if sum(counts.values()) != config['expected_total']:raise ValueError('Training total mismatch')
            for relative in config.get('environment_files',[]):
                path=(Path(config['dataset_root'])/relative).resolve()
                conn.execute('INSERT INTO sources VALUES(?,?,?)',(str(path),sha(path),json.dumps({'role':'tool_environment'})))
            # This is an explicitly IN-TRAIN monitoring cohort, not held-out validation.
            # It does not remove any of the user's 346,219 training scenarios.
            for domain in DOMAINS:
                ids=conn.execute('SELECT task_id FROM tasks WHERE domain=? ORDER BY order_key LIMIT ?',
                                 (domain,config['training_probe_per_domain'])).fetchall()
                if len(ids)!=config['training_probe_per_domain']:raise ValueError('Training monitor too small')
                conn.executemany('INSERT INTO probe VALUES(?)',ids)
            conn.executescript('CREATE INDEX task_window ON tasks(domain,split,seq); CREATE INDEX task_split ON tasks(split,domain);')
            conn.execute('INSERT INTO metadata VALUES(?,?)',('settings',json.dumps(frozen)))
            conn.commit()
        finally:conn.close()
        tmp.replace(db)
        write(out/'input_protocol.json',frozen)
        write(out/'manifest.json',{'counts':counts,'total':sum(counts.values()),'domains':domains,
                                  'split_seed':config['seed'],'search_dev_fraction':0,
                                  'probe_per_domain':dict.fromkeys(DOMAINS,config['training_probe_per_domain']),
                                  'training_monitor':'in_train_only_not_validation','manifest_sha256':sha(db)})
    corpus=out/'search.sqlite'
    if not corpus.exists():
        store=ManifestStore(db)
        try:build_search_index(store.iter_split('evolve_train','searchqa'),corpus,data_manifest_hash=sha(db))
        finally:store.close()
    return out

def audit_validation_overlap(config, validation):
    """Exact public-question audit only; gold labels are never emitted or indexed."""
    from sia.task_meta.data import jsonl_rows, _prompt, _final_public_texts, content_hash
    from collections import Counter
    from common import ROOT
    final=set()
    specs=read(validation['official_specs'])['evaluators']
    final_sources=[(specs[k]['data_path'],'searchqa' if k.endswith('_dev') else 'code')
                   for k in ['livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev']]
    final_sources += [(v['data_path'],'tool_use') for v in validation['tool_benchmarks'].values()]
    for path,domain in final_sources:
        p=Path(path)
        rows=read(p) if p.suffix=='.json' else (r for _,_,r in jsonl_rows(p))
        for row in rows:
            for text in _final_public_texts(row,domain):final.add((domain,content_hash(text)))
    overlap=Counter()
    for spec in config['datasets']:
        for relative in spec['paths']:
            for _,_,row in jsonl_rows(Path(config['dataset_root'])/relative):
                if (spec['domain'],content_hash(_prompt(row,spec['domain']))) in final:overlap[spec['name']]+=1
    result={'exact_public_prompt_overlap':dict(overlap),'gold_labels_exported':False,
            'audit_scope':'normalized exact public text; not semantic decontamination'}
    write(ROOT/'preflight/data_overlap.json',result)
    if overlap:raise ValueError('Train/validation overlap found; refusing to silently change requested counts. See preflight/data_overlap.json')
    return result

"""Pin external benchmark versions once for every round of this experiment."""
from pathlib import Path
from common import read,sha,immutable

def snapshot(settings):
    from tool_validation import native_layout
    files={}
    def add(path):
        path=Path(path).resolve();files[str(path)]=sha(path)
    add(settings['official_specs'])
    specs=read(settings['official_specs'])['evaluators']
    for name in ('livecodebench','humaneval_plus','mbpp_plus','hotpotqa_dev','2wiki_dev'):
        spec=specs[name]
        for field in ('entrypoint','data_path','task_ids_path','alias_path'):
            if spec.get(field):add(spec[field])
    for name,spec in settings['tool_benchmarks'].items():
        repo,_,_,_=native_layout(name,spec)
        add(spec['data_path'])
        for path in sorted(repo.rglob('*')):
            parts=path.relative_to(repo).parts
            if any(p=='.git' or p=='__pycache__' or p.startswith(('result','score')) for p in parts):continue
            if path.is_file() and path.suffix in {'.py','.json','.jsonl'}:add(path)
    return {'file_sha256':files,'validation_settings':settings}

def freeze(settings,path):
    immutable(path,snapshot(settings))

def verify(settings,path):
    if read(path)!=snapshot(settings):
        raise ValueError('External benchmark inputs changed since preflight; cannot compare rounds')

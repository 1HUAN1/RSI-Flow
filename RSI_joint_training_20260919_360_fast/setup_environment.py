"""Install the pinned official evaluator dependencies in this bundle's venv."""
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path
from common import ROOT,read,write,sha

def main():
    if Path(sys.prefix).resolve()!=(ROOT/'.venv').resolve():
        raise RuntimeError('Dependency installation is allowed only in the bundle venv')
    try:import tomllib
    except ImportError:
        try:import tomli as tomllib
        except ImportError:
            with (ROOT/'dependency_install.log').open('ab') as stream:
                subprocess.run([sys.executable,'-m','pip','install','--timeout','120','--retries','3','tomli==2.2.1'],stdout=stream,stderr=stream,check=True)
            import tomli as tomllib
    settings=read(ROOT/'configs/validation.json')
    repo=Path(settings['tool_benchmarks']['bfcl_v3']['repository_root'])
    matches=[p.parent for p in repo.glob('**/bfcl_eval/constants/model_config.py')]
    if len(matches)!=1:raise ValueError('BFCL source root is missing or ambiguous')
    project=matches[0].parents[1]/'pyproject.toml'
    dependencies=tomllib.loads(project.read_text())['project']['dependencies']
    if not dependencies or not all(isinstance(d,str) for d in dependencies):raise ValueError('Invalid official dependencies')
    requirements=ROOT/'requirements-eval.txt'
    own=[line for line in requirements.read_text().splitlines() if line and not line.startswith('#')]
    expected={'official_pyproject_sha256':sha(project),'requirements_sha256':sha(requirements),
              'dependencies':dependencies+own}
    marker=ROOT/'environment_setup.json'
    if marker.exists():
        previous=read(marker)
        if previous['inputs']!=expected:raise ValueError('Dependency inputs changed since setup; use a separate experiment directory')
        for name,version in previous['versions'].items():
            if importlib.metadata.version(name)!=version:raise ValueError('Installed package version changed: '+name)
        return
    log=ROOT/'dependency_install.log'
    with log.open('ab') as stream:
        result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--timeout','120','--retries','3',
                               '-r',str(requirements),*dependencies],stdout=stream,stderr=stream)
    if result.returncode:raise RuntimeError('Official dependency installation failed; inspect '+str(log))
    names={re.match(r'[A-Za-z0-9_.-]+',d)[0] for d in expected['dependencies']}
    names.update(('torch','transformers','peft','datasets'))
    write(marker,{'inputs':expected,'versions':{name:importlib.metadata.version(name) for name in sorted(names)},
                  'base_environment_modified':False})
    print('Official evaluator dependencies installed in the experiment venv.',flush=True)

if __name__=='__main__':main()

"""Start the authorized production controller with secrets received only on stdin."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--run-dir', required=True)
parser.add_argument('--delivery', required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
os.chdir(project)
config = json.loads(Path(args.config).read_text())
run = (project / args.run_dir).resolve()
delivery = (project / args.delivery).resolve()
if not run.is_relative_to(project / 'runs') or run.exists() or not delivery.is_relative_to(project / 'local_baseline'):
    raise SystemExit('New contained run and delivery paths required')
secrets = json.loads(sys.stdin.read(16384))
if set(secrets) != {'OPENROUTER_API_KEY', 'RSI_REMOTE_WORKER_TOKEN'} or not all(isinstance(v, str) and len(v) >= 24 for v in secrets.values()):
    raise SystemExit('Trusted bootstrap credential input invalid; no content logged')
env = dict(os.environ)
env.update(secrets)
del secrets
env['PYTHONUNBUFFERED'] = '1'
env['PATH'] = str(project / 'third_party/bubblewrap/usr/bin') + ':' + env.get('PATH', '/usr/bin:/bin')
delivery.mkdir(parents=True, exist_ok=True)
command = ['timeout', '--signal=TERM', '--kill-after=30s', str(config['max_wall_seconds']), sys.executable,
           'scripts/run_multidomain.py', 'run', '--config', args.config, '--run-dir', args.run_dir]
record = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': command, 'run_dir': str(run),
          'config': args.config, 'full_authorized': True, 'credentials_in_argv_or_log': False,
          'timeout_seconds': config['max_wall_seconds'], 'log': str(delivery / 'production.log')}
if (delivery / 'launch.json').exists():
    raise SystemExit('Launch receipt exists: audit before another launch')
with (delivery / 'launch.json').open('x') as output:
    json.dump({**record, 'state': 'launching'}, output, indent=2)
with (delivery / 'production.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                               start_new_session=True, cwd=project)
record.update(pid=process.pid, state='running')
(delivery / 'launch.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))

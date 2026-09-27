"""Own four local inference services; stop them before the fixed SFT job."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from .storage import save_json

ROOT = Path(__file__).resolve().parents[2]

def registry_path(config=None):
    configured = getattr(config, 'output_root', None) or os.environ.get('RSIFLOW_OUTPUT_ROOT')
    root = Path(configured).resolve() if configured else ROOT
    return root / 'services/four_gpu_services.json'


def process_identity(pid):
    path = Path('/proc') / str(pid)
    try:
        # comm may include spaces, so split after its closing parenthesis.
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z': return None
        return {'pid': pid, 'start_ticks': fields[19], 'pgid': int(fields[2])}
    except FileNotFoundError:
        return None


def stop_services(config=None):
    registry_file = registry_path(config)
    if not registry_file.exists(): raise RuntimeError('No owned four-GPU service registry')
    registry = json.loads(registry_file.read_text())
    owned = []
    for record in registry['services']:
        identity = process_identity(record['pid'])
        if identity is None: continue
        if identity != record['identity'] or identity['pgid'] != record['pid']:
            raise RuntimeError('Service process identity changed; refusing to signal')
        cmd = (Path('/proc') / str(record['pid']) / 'cmdline').read_bytes()
        if str(ROOT / 'scripts/start_multidomain_service.py').encode() not in cmd:
            raise RuntimeError('Service is outside this phase deployment')
        owned.append(record)
    # Validate every owner before the first signal.
    for record in owned: os.killpg(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(process_identity(r['pid']) for r in owned):
        time.sleep(0.5)
    for record in owned:
        # Only the original session/group; Linux cannot recycle its PGID while children remain.
        try: os.killpg(record['pid'], signal.SIGKILL)
        except ProcessLookupError: pass
    deadline = time.monotonic() + 60
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, check=True, timeout=15)
        free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in result.stdout.splitlines()}
        if all(free.get(gpu, 0) >= 40000 for gpu in range(4)): break
        if time.monotonic() >= deadline:
            raise RuntimeError('Inference exited but four training GPUs did not release their memory')
        time.sleep(0.5)
    registry.update(phase='services_stopped', stopped_at=time.time())
    save_json(registry_file, registry)


def ensure_services(config, checkpoint):
    """Restore the selected candidate's model between sequential attempts."""
    registry_file = registry_path(config)
    if registry_file.exists():
        previous = json.loads(registry_file.read_text())
        if (previous.get('phase') == 'inference_ready'
                and Path(previous['checkpoint']).resolve() == Path(checkpoint).resolve()
                and all(process_identity(r['pid']) == r['identity'] for r in previous['services'])):
            verify_services(config, checkpoint)
            return
        stop_services(config)
    start_services(config, checkpoint)


def start_services(config, checkpoint):
    from .storage import checkpoint_manifest
    registry_file = registry_path(config)
    if registry_file.exists():
        previous = json.loads(registry_file.read_text())
        if any(process_identity(r['pid']) for r in previous['services']):
            raise RuntimeError('Existing inference owner must be reconciled, not duplicated')
    directory = registry_file.parent / 'four_gpu'
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    registry = {'phase': 'starting', 'checkpoint': str(checkpoint), 'services': records,
                'started_at': time.time()}
    environment = {k: v for k, v in os.environ.items()
                   if not any(word in k.upper() for word in ('KEY', 'TOKEN', 'PASSWORD', 'SECRET', 'AUTH'))}
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="4")
    for replica in config.task_replicas:
        gpu = replica['gpu']
        config_path = directory / f'gpu_{gpu}.json'
        serving = config.model_dump()
        serving.update(inference_gpu=gpu, task_base_url=replica['base_url'], task_checkpoint=str(checkpoint))
        save_json(config_path, serving)
        command = [config.trainer_python, '-B', str(ROOT / 'scripts/start_multidomain_service.py'),
                   '--config', str(config_path), '--enable-gpu']
        with (directory / f'gpu_{gpu}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        records.append({**replica, 'pid': child.pid, 'identity': process_identity(child.pid),
                        'command': command, 'config': str(config_path)})
        save_json(registry_file, registry)
    expected = {'checkpoint_path': str(Path(checkpoint).resolve()), 'weights': checkpoint_manifest(checkpoint)}
    pending = list(records)
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 600
    while pending and time.monotonic() < deadline:
        for record in list(pending):
            if process_identity(record['pid']) != record['identity']:
                raise RuntimeError(f"Inference service exited on GPU {record['gpu']}")
            try:
                with opener.open(record['base_url'].removesuffix('/v1') + '/health', timeout=3) as response:
                    health = json.load(response)
            except (OSError, TimeoutError): continue
            if (health.get('ready') and health.get('visible_devices') == str(record['gpu'])
                    and health.get('bindings', {}).get(str(checkpoint)) == expected):
                record['verified_binding'] = expected
                pending.remove(record)
            elif health.get('ready'):
                raise RuntimeError('New service loaded an unexpected checkpoint or GPU')
        if pending: time.sleep(1)
    if pending: raise TimeoutError('Four-GPU inference startup did not complete in 600 seconds')
    registry.update(phase='inference_ready', verified_at=time.time())
    save_json(registry_file, registry)
    return records


def verify_services(config, checkpoint):
    from sia.task_meta.storage import checkpoint_manifest
    from sia.task_meta.evolution_protocol import fingerprint
    if len(config.task_replicas) != 4 or {r["gpu"] for r in config.task_replicas} != {0,1,2,3}:
        raise ValueError("Four distinct GPUs are required")
    expected = {"checkpoint_path":str(Path(checkpoint).resolve()),"weights":checkpoint_manifest(checkpoint)}
    opener=build_opener(ProxyHandler({})); records=[]
    for replica in config.task_replicas:
        with opener.open(replica["base_url"].removesuffix("/v1")+"/health",timeout=5) as response:health=json.load(response)
        actual=health.get("bindings",{}).get(str(checkpoint))
        if not health.get("ready") or actual!=expected or health.get("visible_devices")!=str(replica["gpu"]):
            raise RuntimeError("Actual serving weights/GPU do not match frozen child")
        records.append({"gpu":replica["gpu"],"base_url":replica["base_url"],"weights":actual,"model_fingerprint":fingerprint(actual)})
    return records

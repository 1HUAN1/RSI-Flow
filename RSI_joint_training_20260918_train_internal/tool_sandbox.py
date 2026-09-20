"""Native benchmark subprocess: Landlock/seccomp, no capabilities, pipe RPC.

The child cannot open a network connection. Only its two inherited RPC pipes
reach the trusted frozen-Task / bounded user-simulator controller.
"""
import ast
import ctypes
import errno
import inspect
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_RPC_BYTES = 8_000_000


def serve_rpc(incoming,outgoing,handler,errors,workers=4):
    """Bounded concurrent calls, correlated replies; the sandbox remains network-free."""
    writer=threading.Lock();slots=threading.Semaphore(workers)
    def reply(frame):
        try:
            try:
                if handler is None:raise RuntimeError('Preflight cannot issue model requests')
                answer={'response':handler(frame['request'])}
            except Exception as exc:
                answer={'error':type(exc).__name__};errors.append(type(exc).__name__)
            payload=json.dumps({'rpc_id':frame['rpc_id'],**answer}).encode()+b'\n'
            if len(payload)>MAX_RPC_BYTES:raise RuntimeError('RPC result exceeds limit')
            with writer:outgoing.write(payload);outgoing.flush()
        except Exception as exc:errors.append(type(exc).__name__)
        finally:slots.release()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for raw in iter(lambda:incoming.readline(MAX_RPC_BYTES+1),b''):
            if len(raw)>MAX_RPC_BYTES or not raw.endswith(b'\n'):raise RuntimeError('RPC frame limit exceeded')
            frame=json.loads(raw)
            if type(frame.get('rpc_id')) is not int or not isinstance(frame.get('request'),dict):
                raise RuntimeError('Invalid RPC frame')
            slots.acquire();pool.submit(reply,frame)

def run_native(command, work, output, log, handler=None, timeout=604800):
    from sia.task_meta import sandbox
    if not sandbox.probe_isolation()['available']:
        raise RuntimeError('Native benchmark isolation unavailable')
    if Path(output).is_symlink() or any(p.is_symlink() for p in Path(output).parents):
        raise ValueError('Native output must not use symlinks')
    work,output=Path(work).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if output.is_symlink():raise ValueError('Native output must not be a symlink')
    # Reclaim only this dedicated output tree before dropping capabilities.
    for directory,dirs,files in os.walk(output,followlinks=False):
        os.chown(directory,os.geteuid(),os.getegid(),follow_symlinks=False)
        for name in files:os.chown(Path(directory)/name,os.geteuid(),os.getegid(),follow_symlinks=False)
    scratch=output/'sandbox_tmp';scratch.mkdir(exist_ok=True)
    # Reuse the audited deny list, permitting atomic result renames and only
    # CLONE_THREAD (not new processes). Network/ptrace/mount remain denied.
    tree=ast.parse(inspect.getsource(sandbox._restrict_syscalls))
    allowed={'clone','clone3','rename','renameat','renameat2'}
    changed=False
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='denied' for t in node.targets):
            values=ast.literal_eval(node.value)
            if not allowed<=set(values):raise RuntimeError('Unsupported seccomp revision')
            node.value=ast.parse(repr([x for x in values if x not in allowed]),mode='eval').body;changed=True
    if not changed:raise RuntimeError('Seccomp deny list missing')
    namespace=dict(vars(sandbox));exec(compile(ast.fix_missing_locations(tree),'<benchmark-seccomp>','exec'),namespace)
    restrict_syscalls=namespace['_restrict_syscalls']
    filesystem_source=inspect.getsource(sandbox._restrict_filesystem)
    anchor='[(scratch, handled)]'
    if filesystem_source.count(anchor)!=1:raise RuntimeError('Unsupported Landlock revision')
    # Native libraries redirect diagnostics to /dev/null. This device stores no
    # data; granting it write access does not grant any other host file access.
    filesystem_source=filesystem_source.replace(anchor,"[(scratch, handled), (Path('/dev/null'), (1 << 1) | (1 << 2))]")
    exec(compile(filesystem_source,'<benchmark-landlock>','exec'),namespace)
    restrict_filesystem=namespace['_restrict_filesystem']
    reads=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),work,Path(command[2]).resolve()]
    # The official tool runners use their own dependency environment. Training
    # retains the original conda stack, including its compatible NumPy/SciPy.
    evaluator_environment=Path(__file__).resolve().parent/'.venv'
    if Path(command[0]).absolute()==evaluator_environment/'bin/python':
        reads.append(evaluator_environment.resolve())
    reads += [Path(p).resolve() for p in ('/lib','/lib64','/usr/lib','/usr/lib64','/etc/ld.so.cache','/etc/mime.types','/dev/null','/dev/urandom','/dev/random') if Path(p).exists()]
    request_read,request_write=os.pipe();response_read,response_write=os.pipe()
    env={'PATH':str(Path(sys.executable).parent)+':/usr/bin:/bin','HOME':str(scratch),
         'TMPDIR':str(scratch),'LANG':'C.UTF-8','PYTHONPATH':str(work),'PYTHONDONTWRITEBYTECODE':'1',
         'RSI_RPC_WRITE_FD':str(request_write),'RSI_RPC_READ_FD':str(response_read),
         'OPENAI_API_KEY':'pipe-only','API_KEY':'pipe-only','GPT_AGENT_API_KEY':'pipe-only',
         'GPT_BASE_URL':'http://127.0.0.1:1/v1','BASE_URL':'http://127.0.0.1:1/v1',
         'OPENAI_BASE_URL':'http://127.0.0.1:1/v1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    env['CUDA_VISIBLE_DEVICES']=''
    def isolation_body():
        import resource
        os.setsid();os.setgroups([]);os.umask(0o077)
        # Conda lives under a root-only directory on this server. Retain its UID
        # for path traversal, drop ALL capabilities, and restrict filesystem +
        # network + process syscalls before executing any native benchmark code.
        drop_capabilities()
        resource.setrlimit(resource.RLIMIT_NPROC,(128,128))
        resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_FSIZE,(2_000_000_000,2_000_000_000))
        restrict_syscalls();restrict_to_threads();restrict_filesystem(reads,output)
        resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    def isolate():
        try:isolation_body()
        except BaseException as exc:
            os.write(2,('isolation_setup: '+type(exc).__name__+': '+str(exc)+'\n').encode());raise
    process=None;errors=[]
    try:
        process=subprocess.Popen(command,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            pass_fds=(request_write,response_read),preexec_fn=isolate)
        os.close(request_write);request_write=None;os.close(response_read);response_read=None
        def serve():
            try:
                with os.fdopen(request_read,'rb') as incoming,os.fdopen(response_write,'wb') as outgoing:
                    serve_rpc(incoming,outgoing,handler,errors)
            except Exception as exc:errors.append(type(exc).__name__)
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
        thread.join(timeout=5)
        if thread.is_alive():raise RuntimeError('Native RPC did not terminate')
        if code or errors:raise RuntimeError(f'Native evaluator failed (exit={code}, RPC={errors}); inspect {log.name}')
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        for fd in (request_write,response_read):
            if fd is not None:os.close(fd)
        if process is None:
            os.close(request_read);os.close(response_write)

def drop_capabilities():
    libc=ctypes.CDLL(None,use_errno=True)
    for cap in range(64):
        if libc.prctl(24,cap,0,0,0)!=0 and ctypes.get_errno()!=errno.EINVAL:
            raise OSError(ctypes.get_errno(),'drop capability bounding set')
    class Header(ctypes.Structure):_fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
    class Data(ctypes.Structure):_fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
    header=Header(0x20080522,0);data=(Data*2)()
    if libc.capset(ctypes.byref(header),data)!=0:raise OSError(ctypes.get_errno(),'capset')
    if libc.prctl(38,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'no_new_privs')

def restrict_to_threads():
    library=ctypes.CDLL('libseccomp.so.2',use_errno=True)
    library.seccomp_init.argtypes=[ctypes.c_uint32];library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    class Compare(ctypes.Structure):
        _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_uint),('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]
    library.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(Compare)]
    library.seccomp_load.argtypes=[ctypes.c_void_p];library.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=library.seccomp_init(0x7FFF0000)
    if not ctx:raise RuntimeError('seccomp thread filter unavailable')
    try:
        clone=library.seccomp_syscall_resolve_name(b'clone');clone3=library.seccomp_syscall_resolve_name(b'clone3')
        comparison=Compare(0,7,0x10000,0) # flags & CLONE_THREAD == 0
        if clone<0 or library.seccomp_rule_add_array(ctx,0x50000|errno.EPERM,clone,1,ctypes.byref(comparison))!=0:
            raise RuntimeError('Cannot deny process clone')
        # glibc falls back to clone(CLONE_THREAD) after ENOSYS.
        if clone3>=0 and library.seccomp_rule_add_array(ctx,0x50000|errno.ENOSYS,clone3,0,None)!=0:
            raise RuntimeError('Cannot restrict clone3')
        if library.seccomp_load(ctx)!=0:raise RuntimeError('Cannot load thread filter')
    finally:library.seccomp_release(ctx)

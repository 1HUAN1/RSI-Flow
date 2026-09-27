"""Small shared I/O helpers. Training and validation have separate entrypoints."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def resolve_output_root(config):
    """Return the configured external root for mutable experiment outputs."""
    value = config.get('output_root') if isinstance(config, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('output_root must be a non-empty absolute path')
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError('output_root must be absolute')
    if raw.is_symlink():
        raise ValueError('output_root must not be a symbolic link')
    resolved = raw.resolve()
    source = ROOT.resolve()
    if resolved == Path('/') or resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError('output_root must be a dedicated directory outside the project tree')
    if resolved.exists() and not resolved.is_dir():
        raise ValueError('output_root must be a directory')
    return resolved

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def immutable(path, value):
    path=Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Frozen configuration changed: '+str(path))
    else: write(path,value)

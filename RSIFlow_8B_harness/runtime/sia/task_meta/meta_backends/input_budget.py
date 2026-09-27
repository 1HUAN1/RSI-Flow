"""Independent, centrally enforced budgets for immutable inputs and outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from pydantic import BaseModel, ConfigDict, Field


class MetaInputBudget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_bytes: int = Field(default=2 * 1024**3, ge=1024, le=16 * 1024**3)
    evidence_file_bytes: int = Field(default=128 * 1024**2, ge=1024, le=1024**3)
    skill_bytes: int = Field(default=256 * 1024**2, ge=1024, le=1024**3)
    control_bytes: int = Field(default=16 * 1024**2, ge=1024, le=128 * 1024**2)
    max_files: int = Field(default=100000, ge=16, le=1000000)


def category(relative):
    name = Path(relative).as_posix()
    if name in {'meta_input/principles.json', 'meta_input/G/principles.json'} or name.startswith('meta_input/skills/'):
        return 'skill'
    if name.startswith(('meta_input/evidence/', 'meta_input/archives/')):
        return 'evidence'
    return 'control' if name.startswith('meta_input/') or name == 'AGENTS.md' else 'output'


def inventory(root, budget, output_limit):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Workspace symlinks are forbidden')
    totals = dict.fromkeys(('evidence', 'skill', 'control', 'output'), 0)
    limits = {'evidence': budget.evidence_bytes, 'skill': budget.skill_bytes,
              'control': budget.control_bytes, 'output': output_limit}
    files = []
    for parent, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise ValueError('Workspace symlinks are forbidden')
        for name in names:
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f'Workspace symlinks are forbidden: {path}')
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Nonregular workspace file: {path}')
            relative = path.relative_to(root).as_posix()
            kind = category(relative)
            file_limit = budget.evidence_file_bytes if kind == 'evidence' else limits[kind]
            if info.st_size > file_limit:
                raise ValueError(f'Meta {kind} file budget exceeded: {relative}; bytes={info.st_size}; limit={file_limit}')
            totals[kind] += info.st_size
            if totals[kind] > limits[kind]:
                raise ValueError(f'Meta {kind} total budget exceeded at {relative}; bytes={totals[kind]}; limit={limits[kind]}')
            files.append(path)
            if len(files) > budget.max_files:
                raise ValueError(f'Meta file count exceeded: count={len(files)}; limit={budget.max_files}')
    return totals, files


def attach_archives(work, run_root, budget):
    """Copy only controller-declared training records; never mount host directories.

    Hashes bind the original JSONL bytes and each record. Files are streamed rather
    than embedded into prompts or loaded as one giant Python object.
    """
    declaration = work / 'meta_input/trajectory_archives.json'
    if not declaration.exists():
        return
    from sia.task_meta.meta_harness.bundle import reject_links
    run_root = Path(run_root).resolve()
    index, total = [], 0
    destination = work / 'meta_input/archives'
    destination.mkdir(parents=True, exist_ok=True)
    for descriptor in json.loads(declaration.read_text()):
        source = Path(descriptor['path'])
        # Controller evidence links may resolve inside this run; external sources
        # and validation paths are never admitted through a Meta-supplied path.
        source = source.resolve(strict=True)
        if not source.is_relative_to(run_root) or source.name != 'train_trajectories.jsonl':
            raise ValueError('Raw trajectory source is outside the registered training run')
        reject_links(source)
        archive_hash = hashlib.sha256()
        with source.open('rb') as stream:
            for line_number, line in enumerate(stream, 1):
                archive_hash.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('source_role') != 'train_evolution' or row.get('purpose') != 'evolution_train' or row.get('split') != 'evolve_train':
                    raise ValueError('Non-training record in Meta trajectory archive')
                if row.get('round_id') != descriptor['round_id']:
                    raise ValueError('Trajectory archive round mismatch')
                digest = hashlib.sha256(line).hexdigest()
                relative = f'meta_input/archives/{digest}.json'
                target = work / relative
                if len(line) > budget.evidence_file_bytes:
                    raise ValueError(f'Meta evidence file budget exceeded: {source}:{line_number}; bytes={len(line)}; limit={budget.evidence_file_bytes}')
                if not target.exists():
                    total += len(line)
                    if total > budget.evidence_bytes:
                        raise ValueError(f'Meta archive budget exceeded: bytes={total}; limit={budget.evidence_bytes}')
                    with target.open('xb') as output:
                        output.write(line)
                index.append({'task_id': row['task_id'], 'rollout_id': row['rollout_id'],
                    'collection_stage': row.get('collection_stage'), 'round_id': row['round_id'],
                    'source': str(source), 'source_line': line_number,
                    'source_archive_sha256': descriptor['sha256'],
                    'content_reference': {'file': relative, 'sha256': digest, 'encoding': 'canonical_json'}})
        if archive_hash.hexdigest() != descriptor['sha256']:
            raise ValueError('Original training trajectory archive hash mismatch')
    (destination / 'index.json').write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')


def archive_descriptors(run_root, feedback_root, generation, candidate_attempts=()):
    """Trusted controller provenance, no model-chosen files or validation data."""
    root = Path(run_root).resolve()
    result, seen = [], set()
    sources = [Path(feedback_root) / f'gen_{g}' / 'train_trajectories.jsonl' for g in (generation, generation + 1)]
    sources.extend(Path(attempt['trajectory_after']).parent / 'train_trajectories.jsonl'
                   for attempt in candidate_attempts
                   if attempt.get('candidate_executed') and attempt.get('trajectory_after'))
    for source in sources:
        path = source.resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        if not path.is_relative_to(root):
            raise ValueError('Training archive escaped run')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                digest.update(block)
        result.append({'path': str(path), 'sha256': digest.hexdigest(),
                       'round_id': generation + 1, 'bytes': path.stat().st_size})
    return result

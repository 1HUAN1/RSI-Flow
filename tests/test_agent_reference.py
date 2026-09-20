"""Tests for the target agent's reference model (sia.agent_reference)."""

from sia.agent_reference import (
    copy_reference_into,
    parse_agent_reference,
    resolve_agent_reference,
)
from sia.layout import TaskLayout


def _task_dir_with_reference(tmp_path, *, requirements=False):
    """Build a minimal task directory with a bundled reference/ folder."""
    task_dir = tmp_path / "task"
    ref = task_dir / "reference"
    ref.mkdir(parents=True)
    (ref / "reference_target_agent.py").write_text("print('bundled reference')")
    if requirements:
        (ref / "requirements.txt").write_text("anthropic\n")
    return task_dir


def test_default_resolves_to_task_reference(tmp_path):
    task_dir = _task_dir_with_reference(tmp_path)
    layout = TaskLayout(str(task_dir), str(tmp_path))

    ref = parse_agent_reference("default")
    assert ref.kind == "default"

    resolved = resolve_agent_reference(ref, layout)
    assert resolved.inline_seed == "print('bundled reference')"
    assert resolved.ref_dir is None
    assert resolved.entrypoint == "reference_target_agent.py"
    assert resolved.requirements is None


def test_default_picks_up_reference_requirements(tmp_path):
    task_dir = _task_dir_with_reference(tmp_path, requirements=True)
    layout = TaskLayout(str(task_dir), str(tmp_path))

    resolved = resolve_agent_reference(parse_agent_reference("default"), layout)
    assert resolved.requirements is not None
    assert resolved.requirements.name == "requirements.txt"


def test_omitted_spec_is_default():
    assert parse_agent_reference(None).kind == "default"


def test_single_file_reference(tmp_path):
    (tmp_path / "my_agent.py").write_text("print('mine')")
    layout = TaskLayout(str(tmp_path / "task"), str(tmp_path))

    ref = parse_agent_reference({"source": "./my_agent.py"}, base_dir=tmp_path)
    assert ref.kind == "file"

    resolved = resolve_agent_reference(ref, layout)
    assert resolved.inline_seed == "print('mine')"
    assert resolved.ref_dir is None
    assert resolved.entrypoint == "my_agent.py"


def test_directory_reference_reads_from_disk(tmp_path):
    src = tmp_path / "agent_dir"
    src.mkdir()
    (src / "main.py").write_text("import helper")
    (src / "helper.py").write_text("VALUE = 1")
    (src / "requirements.txt").write_text("numpy\n")
    layout = TaskLayout(str(tmp_path / "task"), str(tmp_path))

    ref = parse_agent_reference({"source": "./agent_dir/", "entrypoint": "main.py"}, base_dir=tmp_path)
    assert ref.kind == "dir"

    resolved = resolve_agent_reference(ref, layout)
    # Multi-file reference is NOT embedded — the agent reads it from disk.
    assert resolved.inline_seed is None
    assert resolved.ref_dir == src
    assert resolved.entrypoint == "main.py"
    assert resolved.requirements is not None


def test_copy_reference_into_directory(tmp_path):
    src = tmp_path / "agent_dir"
    src.mkdir()
    (src / "main.py").write_text("x")
    (src / "helper.py").write_text("y")
    layout = TaskLayout(str(tmp_path / "task"), str(tmp_path))
    resolved = resolve_agent_reference(
        parse_agent_reference({"source": "./agent_dir/", "entrypoint": "main.py"}, base_dir=tmp_path),
        layout,
    )

    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    copy_reference_into(resolved, gen_dir)
    assert (gen_dir / "main.py").read_text() == "x"
    assert (gen_dir / "helper.py").read_text() == "y"


def test_copy_reference_into_default_with_requirements(tmp_path):
    task_dir = _task_dir_with_reference(tmp_path, requirements=True)
    layout = TaskLayout(str(task_dir), str(tmp_path))
    resolved = resolve_agent_reference(parse_agent_reference("default"), layout)

    gen_dir = tmp_path / "gen_1"
    gen_dir.mkdir()
    copy_reference_into(resolved, gen_dir)
    # Only the requirements.txt is carried in for a default reference (seed is inline).
    assert (gen_dir / "requirements.txt").read_text() == "anthropic\n"
    assert not (gen_dir / "reference_target_agent.py").exists()

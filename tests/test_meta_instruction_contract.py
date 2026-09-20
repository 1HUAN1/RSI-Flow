"""Guard the actual version-label/full-content confusion without weakening checks."""
from pathlib import Path

import pytest

from sia.task_meta.meta import validate_meta_candidate
from sia.task_meta.meta_harness import MetaHarnessStore
from sia.task_meta.types import MetaAgentState, MetaHarnessUpdate


def test_instruction_content_contract_and_duplicate_guard(tmp_path):
    bundle = MetaHarnessStore(tmp_path / "meta").initialize(
        Path(__file__).resolve().parents[1] / "meta_harness/seed")
    text = (bundle.path / "instructions.md").read_text(encoding="utf-8")
    state = MetaAgentState("fixed-model", str(bundle.path / "instructions.md"),
                           bundle_hash=bundle.hash, bundle_path=str(bundle.path))
    bad = MetaHarnessUpdate(harness="meta_harness_v1", rationale="candidate", changed_rules=[],
                            bundle_files={"instructions.md": text})
    with pytest.raises(ValueError, match="not a version label"):
        validate_meta_candidate(bad, state)
    good = bad.model_copy(update={"harness": text, "bundle_files": {}, "status": "NO_CHANGE"})
    assert validate_meta_candidate(good, state)["passed"]
    schema = MetaHarnessUpdate.model_json_schema()
    assert "complete UTF-8 instructions.md CONTENT" in schema["properties"]["harness"]["description"]

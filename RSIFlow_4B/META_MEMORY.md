# Meta memory and evidence delivery

Task rollout, component routing, single-candidate acceptance, SFT selection and
independent validation are unchanged. This change affects evidence delivery only.

## Storage

- Original parent/child trajectories remain in `Rollout_logs`.
- `principles.json` remains the authoritative ADD-only skill/principle container
  inside each immutable Meta harness snapshot. The existing controller validates
  component skills followed by general principles; old records cannot be rewritten.
- New component records use `skill.<COMPONENT>.<unique_id>`, followed by general
  records using `principle.<unique_id>`. A general operation rationale cites the
  full new skill ID and explains its derivation. New records start at revision 1,
  active=true, evidence_state=tentative, including observed successes. Success and
  failure mechanisms are both appended; existing records remain unchanged.
- Native and experiment-specific validation errors are both delivered to the
  summary repair step. Fixing an uncommitted summary is not a second Task candidate.
- All paired scores and outcome bindings remain available. Large trajectory
  fields use deduplicated, content-addressed JSON references. This is lossless
  presentation, not a modification of recorded evidence or acceptance.
- Each operation exposes a full skill index and at most 64,000 UTF-8 bytes of
  selected active records. Selection is advisory, never a routing gate.
- Previous-round structured handoff remains available. Complete parent/child
  training records are readable through `meta_input/archives/index.json`, with
  task/rollout IDs, original path/line and SHA-256. Validation/test data are excluded.
- `python3 meta_input/read_evidence.py FILE --path '["pairs",0,"after"]'`
  reads a field; `--expand` resolves hashed references. `--offset`/`--max-chars`
  page output (12,000 characters default, 64,000 maximum). Archives are not inlined.

## Independent capacities

Persistent skills have a 64 MiB per-snapshot budget, separate from the 128 KB
ordinary harness-file limit. Record/revision limits are 100,000/1,000,000.
Exceeding a limit raises an explicit error before publication, never silent pruning.
Retained snapshots also consume disk; this is not an unlimited-storage promise.

Training configuration `meta_input_budget` overrides these defaults:

```json
{"meta_input_budget": {
  "evidence_bytes": 2147483648,
  "evidence_file_bytes": 134217728,
  "skill_bytes": 268435456,
  "control_bytes": 16777216,
  "max_files": 100000
}}
```

These are ceilings, not allocations. Evidence, skill inputs, control inputs and
generated outputs are counted separately. Existing `budget.max_workspace_bytes`
counts generated outputs on `local_chroot`, including cumulative stage checks.
Preflight, live checks and result collection use the same input inventory.
API transport and model context limits remain separate and unchanged.

All `meta_input/` files/directories are read-only in the jail. Inputs are streamed,
not assembled into a multi-GB bytes dictionary. Returned inputs are hash-verified.
Full archives require `local_chroot`; other backends are explicitly rejected.

## Verification and recovery

`tests/test_meta_evidence_delivery.py` covers lossless references, independent
budgets, skill reading, archive eligibility and read-only recovery. Existing
append-memory, round-context, sampling and checkpoint tests remain applicable.

`tests/replay_meta_evidence.py --run RUN --call FAILED_CALL` replays real failed
input in a temporary workspace, with no API calls or changes to the run.
`tests/probe_meta_evidence_isolation.py` tests native read-only isolation and
collection of evidence larger than 16 MB, also with no API calls.

This change does not rewrite old experiment fingerprints or restart the stopped
run. Its controller/configuration identities differ; a recorded continuation is
required before resuming unfinished Meta learning. Existing scores, decisions
and checkpoints must not be relabelled as produced by the new controller.

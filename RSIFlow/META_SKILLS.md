# Meta skills and conditional experience

Each experiment stores append-only experience in `meta/skills.jsonl`. Original
trajectories remain in `Rollout_logs`; `skills_index.json` is a rebuildable index.
Snapshots preserve the ledger. The Meta agent makes component-selection,
candidate-design, and acceptance decisions.

## Records

- `case`: a measured intervention (`skill.HARNESS.<id>`, `skill.MODEL.<id>`, or
  `skill.ARTIFACTS.<id>`), with context, diagnosis, mechanism, prospective prediction,
  measured outcome, used skill IDs, round/attempt, applicability, unknowns, and evidence.
- `rule`: conditional experience (`principle.<id>`), with `when`, `recommend`,
  `avoid_when`, `expected_signal`, support, counterevidence, status, and confidence.
- `incident`: a diagnosed interface or grading failure, with signature, owner,
  repair status, and evidence. Zero model calls alone do not establish an external
  failure; affected tasks remain in the full paired denominator.
- `rule_update`: an appended revision targeting a principle, using `weaken`,
  `strengthen`, `contradict`, or `supersede`. Preserve reasons and evidence;
  revisions do not overwrite case facts.
- `procedure`: reusable steps, applicability, preservation requirements, and versions.
- `skill_maintenance`: supplementation, revision, merging, and retirement events.

Suggested statuses: `hypothesis`, `single_paired_support`, `repeated_support`,
`contradicted`, and `superseded`. The Meta agent states confidence from evidence;
invocation counts do not increase it automatically. Legacy `skill`/`principle`
records remain readable, with undeclared principles treated as hypotheses and
`confidence=legacy_observation`. Historical interpretations are not automatically
established conclusions. Identical ID/content retries return `reused_ids`;
conflicting content preserves the old record and returns a warning. Malformed
lines and orphan revisions warn without blocking valid records. There is no
eight-record limit.

## Maintenance

Use `maintain_skills(path=skills_path, operations=[...])`. The Meta agent judges
semantic equivalence; the tools do not automatically merge by similarity.

- `add`: create a procedure with `component`, `when`, `steps`, `preserve`, and evidence.
- `supplement`: append support/counterevidence without changing steps, version,
  or confidence automatically.
- `revise`: create a linked version when conditions or steps change; earlier
  support remains explicitly prior-version evidence, not new measured support.
- `merge`: consolidate targets into a new record or `canonical_id`, deduplicating
  evidence and retaining provenance without automatically expanding applicability.
- `retire`: exclude a procedure from default retrieval, optionally naming
  `replaced_by`; preserve facts and trajectories.

Operations can include `reason` and `evidence_refs`. Missing targets warn while
other operations proceed. IDs can be generated; retries are idempotent. Later
operations in one request can reference earlier saved records. Default retrieval
returns active records, with procedures in each component's skills bucket.
`include_inactive=true` and `read_skill` provide historical access. The index tracks
`active`, `version`, `previous_version`, `merged_from`, `replaced_by`, and maintenance
provenance. Maintenance adds no new experiment-completion gate.

## Workflow

1. `prepare_meta_evidence` produces statistics, 48 stratified snippets, the full
   trajectory index, and `failure_fingerprint_path`. Fingerprints describe observed
   domains, errors, zero model calls, official successes, and scoring versions.
   Task-success counts do not equal eligible SFT-example counts.
2. `retrieve_skills` returns component-specific support, counterexamples, conditional
   rules, unknown cases, and incidents. Defaults: one record per category and
   24,000 characters. Condition-term matching is not a learned gain predictor.
3. Meta checks applicability and counterevidence, using paginated `read_skill` and
   original evidence when needed. Empty memory and no matches are valid outcomes.
4. Before candidate construction, save `decision.json` and freeze `skill_use.json`
   with `record_skill_use`: actual IDs used, rationale, target errors, expected
   improvements, successful behaviors at risk, and retrieval receipt. Missing
   predictions or unknown IDs are recorded without blocking execution.
5. `compare_scores` reports full paired results; `compare_task_differences` reports
   new successes/regressions, retained outcomes, unknowns, error/cost changes, and
   trajectory line/hash references. Missing success booleans remain unknown;
   positive partial rewards do not imply complete-task success. Store summaries
   and references, not duplicate model-call histories.
6. Compare frozen predictions with results, append cases, maintain procedures,
   and add evidence-supported rules. Repeated summaries of one intervention are
   not independent support. `prediction_assessment.met` requires a justified Meta
   judgment; missing assessments remain unknown.
7. `snapshot_task_meta` versions the cumulative ledger. `skill_usage_report`
   aggregates unique interventions, strictly positive first-candidate gains,
   declared prediction outcomes, net new successes, and results associated with
   cited experience. Citation is not understanding; association is not causation.

Retrieval receipts freeze the fingerprint and bounded results for each attempt.
Tool errors return to the same Meta session. Independent evaluation is for
reporting and does not enter current-round experience learning. Unchanged domain
success totals can hide per-task regressions.

## Initialization

New experiments start empty (`meta_skill_seed_path=null`). The optional
`meta_skills/reviewed_seed.jsonl` contains three reviewed historical cases, three
conditional rules, and two incidents, preserving combined-intervention and old
grading limitations. It contains no independent-evaluation scores. Paired
new-success/new-regression counts are B1: 1/9, B2: 6/3, and B3: 11/6. Original
trajectory paths are provenance references; those run logs are not bundled.

Set `meta_skill_seed_path` in a custom profile or use `--meta-skill-seed` to inherit
this seed explicitly. It initializes only new ledgers; resumes do not replay it.
`skills_initialization.json` records `empty` or `seeded`. Comparisons must report
whether Meta started with prior experience. `--check` checks seed paths without
API calls or experiment writes. `meta_skill_context_chars=24000` and
`meta_skill_per_category=1` bound retrieval, not total memory size or model tokens.
Read larger records with pagination.

## Validation

Offline tests cover revisions, legacy compatibility, counterevidence, retrieval
budgets, malformed lines, initialization retries, idempotence, cross-round
inheritance, snapshots, per-task regressions despite unchanged totals, partial
rewards, predictions, and usage statistics. Capability gains require controlled
comparisons on new tasks with fixed Task state, API, budget, and batches; offline
tests do not establish those gains.

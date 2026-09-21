# 2026-09-21 evaluation audit

## Observed run, not a completed benchmark result

`rsiflow_4b_ds41_180_fresh_v1_r3` committed Meta1 (7 cumulative records),
and completed the next round's 180-task early rollout. Its first independent
validation then failed after BFCL generation. The remaining six benchmarks did
not run. This patch was not hot-loaded into that process and does not repair or
overwrite its historical receipts. No new model/API evaluation was launched.

## Findings and fixes

| Path | Finding | Change / verification |
| --- | --- | --- |
| BFCL / ACE shared bridge | `OfficialTurn.evaluate` absent; generated answer turned into infrastructure/RPC failure | Separate inference-only environment returns `pending_official`; real Harness + offline-model test passes |
| HotpotQA / 2Wiki generation | Training terminal scorer requires private reference answers, absent from the deliberately public prompt | Public tools preserved; local terminal scoring deferred to official evaluator |
| LCB / HumanEval+ / MBPP+ generation | Training terminal scorer uses a different test contract; returned dataclass also cannot be directly JSON-saved | Same inference/scoring boundary; terminal receipt explicitly serialized |
| Runtime result memory | An unscored turn must not generate correctness-based experience | Skip post-result ingestion for `pending_official`; normal training path regression-tested |
| Error reporting | Original Harness reason lost behind generic RuntimeError | Local transport receipt retains exception text and underlying Harness error fields |

Training still calls a separate `validate.py` process through `round_validation.py`.
`start_evaluation.sh` exposes that same interface for standalone evaluation. No
Meta client or SFT trainer is started by this entry point. See `EVALUATION.md`.

## Checks completed

- Pinned source verification against `Rollout_logs/validation_sources.json`: passed.
- Frozen selected data / evaluator / ID checks: LCB 50, HumanEval+ 25,
  MBPP+ 25, HotpotQA 50, 2Wiki 50. BFCL and ACE native layouts validated.
- Fixed 300-task denominator and complete ACE conversation grouping tested;
  an incomplete benchmark cannot become a completed overall score.
- New boundary + native runtime + four-replica routing + fixed protocol +
  experience append + early-rollout regression suite: **53 passed, 21 subtests passed**.
- Official reporting, public input filtering, pinned EvalPlus and LiveCodeBench
  comparator/isolation tests: **41 passed**.
- Evidence delivery, memory growth, deployed recovery, storage, 180-task protocol
  and launcher regression suite: **31 passed, 11 subtests passed**.
- Bash syntax check for standalone evaluation: passed.
- The process-pool test emits two Python warnings about `fork` from a
  multithreaded process; it passes, but these warnings are not a concurrency guarantee.

Test model responses are offline fixtures; no paid API or GPU model calls were
used for these tests. Source checks and comparator tests are not proof that all
300 live benchmark tasks finish successfully.

## Remaining operational boundaries

- Old `requires_audit` requests cannot be silently retried or counted as model
  failures. Archive/reconcile the failed attempt and record the code revision
  before resuming; preserve Task1, Meta1 and the early-rollout evidence.
- Validation is fail-fast at a benchmark infrastructure error. It does not
  currently collect later benchmark results after such a failure. This audit did
  not change that scheduling policy or claim the other six have live results.
- ACE's user simulator still requires its own private API environment and live
  endpoint. Offline tests do not establish remote API availability.
- Standalone evaluation may prepare/switch the configured GPU services. Do not
  run it concurrently with training; a universal cross-entrypoint GPU lock is
  not introduced by this patch.
- The tool bridge currently passes empty reusable artifacts text, unlike the
  Code/Search generator, which loads snapshot artifacts. Thus complete parity
  for a future nonempty reusable-artifact Task is **not established** by this fix.
- Public Search evaluation uses the configured frozen retrieval corpus and
  reports official **answer EM/F1**, not full evidence/supporting-fact leaderboard
  metrics. The seven fixed subsets are not full-benchmark leaderboard runs.
- Server configs contain absolute dependency paths. A Git clone still requires
  external model weights, datasets, pinned evaluators and prepared environments.

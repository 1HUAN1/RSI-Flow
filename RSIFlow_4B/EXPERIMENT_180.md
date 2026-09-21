# Three fresh mixed-domain rounds (180 tasks each)

Launch with `bash start_180_training.sh`. This selects `configs/train_180.json`;
the older `configs/train.json` and its 360-task run artifacts are retained.

- Three group-disjoint rounds: EnvScaler 60, DeepCoder/TACO 60, NQ 15,
  HotpotQA 23, 2Wiki 22 per round; 540 distinct training tasks in total.
- Four model replicas, 16 rollout workers; one component and at most one candidate
  per round. Parent/child evaluation uses the identical 180-task manifest.
- All-task summaries and 48 stratified excerpts remain available to Meta.
- Independent validation remains 300 tasks per round and is report-only.
- MODEL SFT remains one epoch, verified-success training trajectories only.
- Fresh rollout cache and Task Memory: no import of the previous 360-task run.
- New run name: `rsiflow_4b_ds41_180_fresh_v1_r3` under the configured Rollout_logs.

Task selection is outcome-independent and deterministic. The existing frozen
task definitions supply the candidate pool, not their rollout results. Preparation:

```bash
python prepare_subset_release.py --config configs/train_180.json \
  --parent-data-root /root/data/RSI_iclr2027/rsiH/RSI_joint_training_20260919_360_fast/runtime/data/rounds_1080
python launch.py --config configs/train_180.json --dry-run
bash start_180_training.sh
```

Preparation is exclusive and must not be repeated over an existing release.

## Failure semantics

A model request rejected by the fixed Task context budget terminates that Task
as `context_budget_exhausted`, reward 0, completed verification, and remains in
the evaluation denominator and Meta evidence. The rejected request and preceding
calls remain recorded. Service/transport and unrelated evaluator errors still
block scoring. The 32,768-token service budget has not been increased.

Four pinned check functions for EnvScaler `env_154_rl-task_21` assume dictionary
profile fields although its tools accept strings. The reproduced string `.get`
error is counted as a failed task requirement only when the task identity,
original check code hash, and exception match. Original checker errors and the
classification `task_state_type_mismatch` are retained. Unknown or changed
checker failures are not silently downgraded.

This changes the failure-policy version relative to the stopped 360-task run.
New parent and child rollouts both use this policy; historical receipts are not
rewritten or compared as if they used the new policy.

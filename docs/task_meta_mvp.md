# Task–Meta Co-Evolution MVP

This is an opt-in extension of SIA in `/root/data/RSI_iclr2027/rsiH`. The original CLI
defaults to `--evolution-mode sia`; `--evolution-mode task-meta` selects this loop.

The Task model is `/root/data/zh/huggingface/Qwen2.5-3B-Instruct`. The frozen Meta
model is OpenRouter `z-ai/glm-5.2`, as requested most recently. No secret is stored
in profiles, configuration, generated prompts, or this document.

## Run

```bash
cd /root/data/RSI_iclr2027/rsiH
conda activate sia
bash scripts/start_task_meta_qwen.sh

read -r -s -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
export LOCAL_QWEN_API_KEY=local

# Five evaluated generations, T_0 through T_4; choose an unused integer run ID.
bash scripts/run_task_meta.sh 2010 5 configs/task-meta.json

# Smaller real smoke: one GPQA question, eight rollouts per generation, three generations.
bash scripts/run_task_meta.sh 2011 3 configs/task-meta-smoke.json
```

The endpoint must have finished loading before a run starts. Its readiness and
model ID can be checked at `http://127.0.0.1:8001/health`. The launcher uses the
active Python environment; it does not create another environment for every run.
The installed SIA environment must include `pydantic-ai`, `openai`, and SIA's
existing dependencies. No GPU or API is required to run the fake-component tests.

Equivalent explicit CLI:

```bash
python -m sia run --evolution-mode task-meta --task gpqa \
  --meta-agent-profile task-meta-glm --target-agent-profile task-meta-qwen \
  --task-meta-config configs/task-meta.json --max_gen 5 --run_id 2012 --no-web
```

## State and exact order

`TaskAgentState` stores generation, model serving reference, harness path, active
artifact state, and optional checkpoint path/content manifest. `MetaAgentState`
stores the frozen model reference, current harness path, and harness version.

```text
T_0 rollout -> p_0 -> M_0 routes one Task update
T_1 rollout -> p_1 -> experience e_0 -> accept Meta harness G_1
                      M_1 routes one Task update
T_2 rollout -> p_2 -> experience e_1 -> accept Meta harness G_2
...
last Task rollout -> last experience -> last Meta self-update
                 -> one final consolidation -> final Task and Meta states
```

`--max_gen 5` means five executions (`gen_0` ... `gen_4`), four Task updates,
four improvement experiences, four ordinary Meta self-updates, and one final
consolidation. Meta `harness_v0.md` ... `harness_v5.md` therefore exist for a
complete five-generation run. Final consolidation is a distinct versioning event.
Generation 0 does not learn from an improvement that has not yet been evaluated.
No `gen_5` successor is created after a five-generation run.

Negative or zero performance deltas remain valid experiences. All structurally
valid Meta Harness updates are accepted immediately. There is no Meta evaluator,
fitness, old/new comparison, candidate search, rejection or rollback.

## Evaluation and rollout data

The first `batch_size` GPQA questions are copied once to a run-owned task directory,
with matching private evaluator records and an unchanged copy of SIA's GPQA
evaluator. Their IDs and content hashes are saved in `task/batch_manifest.json`.
Every generation evaluates this exact same batch. Public data, private data and
evaluator hashes are checked before/after execution.

For batch size B and repetitions R (default 8), the seed executes B questions in
each repetition. The controller runs SIA's evaluator on each repetition and joins
its per-question correctness back onto the recorded model messages. There are
B × R terminal rewards. The metric is

```text
success_rate = number of correct responses / (B × R)
```

Missing or invalid answers have zero reward and remain in the denominator. This
avoids GPQA's legacy `accuracy` field, which divides by attempted valid answers.
SIA's full evaluator outputs are retained in `evaluator_results.json`. Aggregate
`results.json` also includes per-task rates and attempt counts. A score from one
or two reused questions is a development signal, not a held-out benchmark result.

The loop's `primary_metric` function supports a configurable numeric key or dotted
path and a `max`/`min` direction. Positive delta always means improvement:
`after - before` for max, `before - after` for min. The currently shipped GPQA
adapter supplies `success_rate`; other tasks/metrics need their own executor adapter.

The seed records input/output messages, model answer, available token usage,
errors and elapsed time. The controller supplies `question_id`, `rollout_id` and
trusted `terminal_reward`. API/GPU dollar/hour costs that are unavailable are null.
Meta API calls have separate prompt/output/token/time records in `meta/calls/`.
Seeds 42 ... 49 are sent to the endpoint in every generation. The existing CPU
endpoint may ignore the seed; real sampling is not claimed to be deterministic.

## Artifacts lifecycle

Before the first Task execution, `ArtifactState(directory=None, manifest=[])` is
empty. A valid Task model response can provide `reusable_note`, saved as an external
strategy file after task execution. No trajectory, submission, score or Meta log
is relabeled as an artifact. Tasks that produce no reusable product may leave the
artifact state empty. The 12-token smoke config usually only generates an answer;
the 128-token config allows longer responses and optional reusable notes.

Artifacts have their own workspace. The next Task sees only the current active
snapshot. All repetitions within a generation receive the same input snapshot,
so earlier attempts do not improve later attempts' context. The latest resulting
rollout artifact snapshot becomes the active output after evaluation. Historical
snapshots remain on disk; they are never all concatenated into the current prompt.
An `artifacts_input/` copy preserves a generation's pre-execution input for replay;
`task_state.json` points to it, while `task_state_after_rollout.json` points to the
latest output in `artifacts/`. Manifests record relative paths, sizes and SHA-256.

The current GPQA adapter consumes assets as text resources in the prompt. Python
solver assets can be stored and read as text but are not executed as tools in this
MVP. Binary assets and multi-file executable workspaces need a future adapter.

## Three update branches

| Decision | Actual operation | Fixed components |
| --- | --- | --- |
| HARNESS | Reuses SIA feedback prompt context; updates pure `format_question` and `parse_answer` helpers in `target_agent.py` | Runtime/model calls, Model state and artifact bytes |
| ARTIFACTS | Structured create/change/delete operations within the active artifact workspace | Harness bytes and Model state |
| MODEL | Builds reward-labeled rollout/SFT inputs and generates SIA `train.py`; optionally invokes a configured external trainer | Harness bytes and artifact bytes |

Routing returns a Pydantic `MetaDecision`, whose action enum admits exactly these
three values. A dispatcher invokes one updater only; no best-of-three search exists.
Hashes and state comparisons reject cross-component or historical snapshot changes.

Harness helpers are restricted to pure local string/dict, regex and JSON operations;
imports, global state, file/network access, function signatures and the model/runtime
I/O remain fixed. Prompt/planning and parser improvements are supported now. Broader
tool/workflow rewrites need an explicit extension of this contract. This code is
not an OS security sandbox for arbitrary user-supplied Python.

MODEL input `rollouts.jsonl` contains messages, task/repetition IDs and terminal
rewards. `sft_positive.jsonl` contains only positive-reward model trajectories. For
GRPO, group by `question_id`. The script-generation prompt reuses SIA's existing
weight mode. The system does not add a new local RL training infrastructure.

With the default `trainer_command: null`, MODEL saves its request under the current
generation's `model_update/`, finishes Meta consolidation over observed experience,
and ends with `pending_model_update` (CLI exit code 2). It never reports a relabeled
base model as trained and never evaluates a fictional successor. Providing external
training is required to continue a MODEL intervention.

An optional external trainer command is a JSON list of arguments, not a shell
string. `{request_dir}` and `{script}` are expanded per argument. The trainer must
produce `checkpoint.json` with a new `model_ref` and separate `checkpoint_path`,
and serve that model via the configured target endpoint. The controller checks
that old weight hashes remain unchanged, new weight hashes differ, nonempty
weights exist, and the new model is advertised by the endpoint. Serving the exact
checkpoint is the external trainer's contract. No real training backend has been
validated on this CPU-only machine.

## Budget, files and failure behavior

`max_generations` is the primary budget. Optional `max_wall_time` is checked at
generation and update boundaries. It allows an already-started evaluation and
its resulting Meta feedback to finish; it is not a process kill timer. API-dollar
and GPU-hour budgets require authoritative cost reporting and are not enforced yet.

```text
runs/run_ID/
  task/                         fixed public/private batch + official evaluator
  profiles.json                 resolved model/provider configuration (no keys)
  task_meta_config.json
  meta/
    harness_v0.md ... harness_vN.md
    experiences.jsonl
    calls/                      Meta input/output/token/time records
    final_consolidation.json
    final_meta_summary.md
  gen_0/ ... gen_4/
    target_agent.py
    task_state.json             state used to execute
    task_state_after_rollout.json
    artifacts_input/            immutable incoming assets, when present
    artifacts/                  latest active outgoing assets, when present
    artifact_manifest.json
    rollouts/rollout_0/ ... rollout_7/
    agent_execution.json        trajectories with trusted terminal rewards
    results.json                aggregate performance
    evaluator_results.json
    cost.json
    meta_observation.json       when another Task update is allowed
    meta_decision.json
    task_update.json
    improvement_experience.json # generation 1 onward
    meta_self_update.json       generation 1 onward
  final_state.json
  final_summary.md
```

Task/API/evaluator failures write `failure.json` and keep existing run evidence.
They do not fabricate scores or silently route another intervention. Existing run
IDs are never overwritten. Automatic mid-run resume is not implemented; completed
generation inputs, source, outputs and configuration are retained for manual replay.

## Tests

```bash
python -m pytest -q
python -m pytest -q tests/test_task_meta_loop.py
python -m ruff check sia/task_meta tests/test_task_meta*.py
```

The five-generation fake integration fixes its action sequence to HARNESS, MODEL,
ARTIFACTS, HARNESS and scores to 0.30, 0.40, 0.52, 0.58, 0.60. Those are test inputs,
not experimental results. It checks exact timing, one selected updater, empty initial
artifacts, the new Meta Harness on the next decision, saved versions, metric
directions, early budgets, pending training and no unused final successor.

## Future work explicitly outside this MVP

- TODO: Meta Harness quality evaluation
- TODO: Meta Harness accept / reject
- TODO: Meta Harness rollback
- TODO: multiple Meta candidates
- TODO: Meta-RL
- TODO: learned routing policy
- TODO: multi-step Meta fitness
- TODO: statistical validation of Meta improvement
- TODO: experience replay prioritization
- TODO: three-way oracle comparison

Further engineering work: validate a real SFT/GRPO trainer and checkpoint serving;
extend task adapters beyond GPQA; execute solver tools; broaden the constrained Task
Harness contract; implement automatic resume and authoritative cost budgets.

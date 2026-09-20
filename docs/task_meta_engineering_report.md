# Task–Meta MVP Engineering Report

Repository: `/root/data/RSI_iclr2027/rsiH`.

## 1. Core files inspected

Inspection covered the control flow and relevant interfaces in `sia/orchestrator.py`,
`sia/context_manager.py`, `sia/prompts.py`, `sia/cli.py`, `sia/config.py`,
`sia/run_setup.py`, `sia/layout.py`, `sia/results.py`, `sia/profiles.py`,
`sia/providers.py`, `sia/config_files.py`, `sia/agent_reference.py`,
`sia/agent_impls/pydantic_ai.py`, GPQA's reference agent and evaluator,
`pyproject.toml`, generation/CLI/profile tests and the previous local Qwen scripts.
The repository already contained modified `sia/prompts.py`, `sia/providers.py`,
local baseline files, user profiles/providers and a timeout test; they were preserved.

## 2. Original SIA control flow

`main()` resolves task and model profiles, creates a run/environment, constructs
an initial Meta prompt and invokes an agent implementation to generate the initial
Task program. For every generation, `run_generation()` executes the Task program,
runs the task's evaluator, updates `ContextManager`, and invokes `_run_feedback_agent()`
to write the next generation unless it is the final one. `focus=weights` selects
generated `train.py` and SIA's external training-service prompts. The original
Meta-Agent initializes the program; the recurring Feedback-Agent drives later changes.

## 3–5. Changed files, new files, classes and interfaces

Two existing core files change: `sia/cli.py` adds opt-in mode/config arguments;
`sia/orchestrator.py` dispatches the new mode before the original run path.
No existing SIA evaluator, provider logic, prompt template or generation function
is replaced. Documentation and launcher/config files are additive.

| New module | Main definitions |
| --- | --- |
| `sia/task_meta/types.py` | `TaskUpdateAction`, `TaskAgentState`, `MetaAgentState`, `ArtifactState`, `MetaObservation`, `MetaDecision`, `ImprovementExperience`, `GenerationContext`, `EvaluationResult`, `TaskUpdate`, `TaskUpdater`, `MetaHarnessUpdate`, `UpdatePending` |
| `sia/task_meta/loop.py` | `run_task_meta`, `BudgetManager`, `primary_metric`, `performance_delta` |
| `sia/task_meta/meta.py` | `StructuredClient`, `MetaAgent` |
| `sia/task_meta/prompts.py` | Separate route, learn, consolidation and initial Harness instructions |
| `sia/task_meta/execution.py` | `SIAExecutor`, `prepare_gpqa_task` |
| `sia/task_meta/updaters.py` | `HarnessUpdater`, `ModelUpdater`, `ArtifactUpdater`, structured edit outputs |
| `sia/task_meta/harness.py` | Pure-helper Harness contract validation |
| `sia/task_meta/storage.py` | JSON persistence, hashes, artifact/checkpoint manifests and generation cloning |
| `sia/task_meta/gpqa_target.py` | Seed adapted from the SIA GPQA reference output/client convention |
| `sia/task_meta/entry.py` | `TaskMetaConfig`, CLI bridge and initialization |

Also added: two bundled model profiles, one local provider definition, two JSON
run configurations, two launcher scripts, new Task–Meta tests and these documents.
The existing SIA `RunLayout`, task/profile/reference loading, PydanticAI provider
resolution, evaluator, trajectory loader and feedback/weight prompt builders are reused.
The new mode uses its own compact loop so SIA's one-based lifecycle stays unchanged.

## 6–8. Task state, Meta state and artifact lifecycle

Task state contains the active model/checkpoint reference, the generation's Harness
source path, and an independent artifact workspace/manifest. Checkpoint metadata
is recorded when an external training result is verified. Meta state contains a
fixed model ID and a versioned path to learned instructions.

Artifacts start empty. Task-produced reusable notes can populate the workspace after
execution; Meta's ARTIFACTS action can edit assets. All attempts in a generation see
the same snapshot. Pre-rollout artifacts remain under `artifacts_input/`; outgoing
artifacts are stored separately. Historical snapshots stay on disk and only the
latest active version enters Task context. Logs, trajectories and results are excluded.

## 9–12. Branches, Feedback replacement and Meta update timing

The recurring `MetaAgent` replaces the new mode's Feedback-Agent role. It returns
one validated action plus diagnosis/evidence/change proposal. The dispatcher invokes
exactly the matching updater. HARNESS rewrites prompt/parser helpers under a fixed
model/runtime contract; ARTIFACTS applies scoped file edits; MODEL prepares real
reward-derived training inputs and SIA training code, optionally invoking an external
trainer with verified changed weight files and a new served reference.

The first decision uses M_0. Once T_1 has been evaluated, before/after state,
performance, trajectory references, intervention and costs form e_0. The current
Meta generates new instructions from e_0 and historical experience. The controller
saves a new immutable Harness version and updates Meta state immediately. The next
route reads that path. There is no Meta quality test, candidate competition or rollback.

## 13–15. Delta, generations and consolidation

Metrics are configurable numeric keys; `max` uses `after-before`, `min` uses
`before-after`. The GPQA runtime uses correct responses over all B × 8 attempts.
Five generations means T_0 ... T_4, four Task interventions and four experiences.
After T_4's feedback updates the Meta Harness, one final consolidation reads all
experience summaries and all historical Meta Harness versions. It creates one
final Meta Harness and summary; it never creates T_5.

## 16–17. Validation

Before modifications: **148 passed, 1 skipped**. The first integrated version:
**172 passed, 1 skipped**, including the deterministic five-generation loop and
GPQA adapter tests. Further boundary/replay tests and final real-run observations
are recorded in the validation results accompanying this report.

The real smoke uses the actual local Qwen and OpenRouter GLM, not deterministic
fake scores. Its selected GPQA subset is a repeated development batch, so its
score must not be reported as a full paper reproduction or held-out benchmark.

## 18. Current limitations

The machine exposes 2 CPU cores and 8 GiB RAM with no visible GPU. MODEL can prepare
a request; real weight training is unverified and needs an external trainer. With
no trainer it ends explicitly as pending instead of pretending the weights changed.
Only GPQA has a concrete executor. Harness evolution currently covers prompt/planning
text and answer parsing; arbitrary workflow/tool rewrites are not enabled. Assets
are read as text, including scripts as text rather than executed solvers. The short
smoke output may not produce reusable notes. There is no automatic resume, OS sandbox,
statistical validation, Meta quality evaluation, verified server-side random seed
reproducibility, or API-dollar/GPU-hour budget enforcement.

The explicit research TODO list and run commands are in [task_meta_mvp.md](task_meta_mvp.md).

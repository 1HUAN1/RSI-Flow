# Shared Task initialization

`seed.json` is the executable generation-zero Harness specification consumed by
`sia.task_meta.seed.run_seed`. Every method compared in a future experiment must
receive an independent copy of this exact specification, the same Qwen3-4B
checkpoint/tokenizer/template manifest, and the same empty initial artifacts.
Evolution changes successor copies; it does not overwrite this seed.

The reference is HarnessForge commit
`05b3ecadb3c9a7a938f75129ea22b8f2b36cf289`, through
`harness_factory.base_harness.builder.build_agent_from_context`. The builder
selects `generic_planning`, `single_react`, and `lightweight_memory`. Byte-preserved
reference files and SHA-256 identities are included under `reference/` and in
`seed.json`; the source's BOM is retained in archived files.

The initial model calls are a planning request, a short-term memory extraction
request when the context delta reaches 50 characters, and a ReAct action request.
Subsequent actions receive serialized plan/tool observations and short-term
guidance according to the configured interval. Summary calls and exhaustion
finalization use the same registered Task model. A `final_answer` call or declared
environment terminal observation stops the rollout. Internal helper calls are
recorded and count against the model-call budget.

The native provider injects five strategic and two operational cold-start rules.
They are recorded as preloaded Harness rules; native long-term provision is
disabled, so these rules are not secretly added to model context. Short-term
memory starts empty for every rollout. Previous native runs and their memory
databases are excluded. Natural notes are returned to the trusted controller for
its current-window artifact lifecycle and remain unverified knowledge.

## Declared adaptations

- Three-domain tools use a controller-supplied `tools`/`step` interface. They
  replace native benchmark-specific loading. The Harness cannot select models,
  change scoring or grant tools/path permissions.
- Summary interval is fixed at 8 instead of the native per-item random 7–9.
  Native accounting that includes planning/summary steps and the final synthesis
  after step exhaustion is retained.
- One tool call per action and serial execution apply across the three domains;
  native code has an optional cap and can execute multiple calls concurrently.
- Explicit seed budgets are 128 model calls, 50 tool calls, 2048 output tokens
  per model call and temperature 0.7. A run that declares a different initial
  budget must record a new seed snapshot; Meta H updates cannot change budgets.
- Strict JSON parsing with a bounded repair request replaces native
  `json_repair` recovery. Canonical successful JSON behavior is replayed against
  upstream methods. Permissive malformed-output behavior is intentionally not
  claimed equivalent.
- Native `tool-response` context maps to `user`, matching its `run_infer`
  transport configuration; text blocks are represented as text strings.
- Native cross-task long-term ingestion/backfill is replaced by the existing RSI
  A-in/A-out lifecycle. Rollout notes are not silently shared within a window.
- Model/infrastructure failures become explicit incomplete trajectories.
  No model response, task score or checkpoint is manufactured on failure.

## Validation scope

`tests/test_task_meta_seed.py` executes extracted, unmodified method bodies from
the archived upstream files with fake model/tool/logging dependencies. It compares
the initial planning request, initial native short-term extraction request, first
lookup action, observed tool result, next action request, and final stop against
the adapted seed. Another replay checks native step-budget finalization. These
are offline method-contract tests, not a complete upstream runtime deployment,
real Qwen smoke, or evidence that any Harness improves benchmark performance.

The H-updater test commits an explicit planning-prompt leaf change, preserves the
previous model/assets/seed, reloads the successor, and observes its changed next
model request. Only bounded leaves exposed by `seed_capabilities` can change.

`run_seed` returns every actual request and raw assistant response in
`model_calls` and `sft_conversations`. HarnessForge reconstructs past action
context with labels such as `[PLAN]` and `Calling tools`; these are context,
not new model targets. The formal SFT path expands successful rollouts into the
actual call dialogues and supervises only each dialogue's final assistant.
This covers planning/action/memory/summary decisions without training rewritten
history or tool observations. The separate `assistant_all` mode remains available
for native histories whose assistant messages are all actual recorded outputs.

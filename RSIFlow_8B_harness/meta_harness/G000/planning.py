"""Evolvable decision preparation, not a hardcoded component router."""


def prepare(context):
    phase = context.get("phase", "route")
    if phase == 'execution':
        return {'instructions': 'Continue the recorded execution/validation milestone or attach to its running job. Do not start routing or a new candidate before the current mainline stage is ready.'}
    if phase == "meta_review":
        return {"instructions": "Choose keep or replace based on a specific Meta behavior gap. A replacement is a complete workflow.py/planning.py/memory.py folder; unchanged modules are copied. Use materialize_meta_harness, check_meta_harness and update_meta_harness. Only executable/real-hook checks are required, NOT a Meta score gain. The checked package is queued for round t+1, never hot-swapped during this round."}
    if phase == "feedback":
        return {"instructions": "Use complete same-batch scores for the Task decision. A positive total delta can coexist with a failed target hypothesis. Record both; do not infer noise size, causality or generalization from a single comparison. Negative/zero Task gain retains the parent and returns to component selection."}
    return {"instructions": "Meta owns the choice among HARNESS, MODEL and ARTIFACTS. Compare each plausible component's repairable failures, available intervention/data, preservation risks and unknowns. No component rotation or diversity is required. Prior unsuccessful Harness attempts do not prove all Harness repair options are exhausted. Record diagnosis, rationale, target changes, cited skills and at-risk successes before constructing one candidate.",
            "component_capabilities": {
                "HARNESS": "Repair execution/Action, Planning, Memory or wiring using the pinned HarnessForge production stages.",
                "MODEL": "One-epoch LoRA SFT on all configured GPUs on eligible verified-success parent trajectories; inspect their coverage, not just the success count.",
                "ARTIFACTS": "Edit submissions and re-score as configured; do not describe direct answer changes as model learning."}}

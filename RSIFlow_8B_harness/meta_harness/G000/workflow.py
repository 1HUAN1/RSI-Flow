"""Evolvable evidence investigation. No Task acceptance or tool dispatch lives here."""


def prepare(context):
    phase = context.get("phase", "route")
    instructions = {
        "route": "Read all outcome statistics, the 48 representative excerpts, relevant skills and handoff. Follow original trajectory references when the failure label does not identify a mechanism. Distinguish model, Harness and platform/scorer faults before proposing a repair.",
        "candidate": "Identify the earliest observable divergence, the responsible code path and the behavior the candidate will change. Check a few relevant failures and previously successful behaviors using this SAME candidate before the full paired rollout. Checks are observations for repair, not a new acceptance gate.",
        "feedback": "Compare the frozen prediction with new successes, new regressions, target-error transitions and cost. Separate whole-candidate improvement from evidence for its proposed mechanism; combined edits do not identify individual contributions.",
        "meta_review": "After maintaining skills, inspect your OWN decision and tool history. Was relevant experience provided but not translated into action? Did the same process error recur? Does the correct behavior require program support? Identify a concrete behavior, evidence, module and expected observable change, or explicitly retain this Meta package.",
    }
    return {"phase": phase, "instructions": instructions.get(phase, "Continue the current milestone; preserve original evidence."),
            "evidence_paths": context.get("evidence_paths", [])}


def review(context):
    """Expose questions/facts, never automatically decide to change Meta."""
    selections = context.get("selections", [])
    facts = [{"component": item.get("component"), "decision": item.get("decision"),
              "delta": item.get("delta"), "prediction_assessment": item.get("prediction_assessment"),
              "paired_facts": item.get("paired_facts")} for item in selections]
    return {"facts": facts, "questions": [
        "Which necessary behavior was absent, and which decision/tool trace demonstrates that?",
        "Was the relevant skill actually made available, or is retrieval the missing step?",
        "Is this missing knowledge, a repeatable program gap, or an external engineering incident?"],
        "decision_owner": "Meta", "allowed_modules": ["workflow.py", "planning.py", "memory.py"]}

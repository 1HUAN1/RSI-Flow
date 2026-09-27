"""Evolvable use of the external append-only skill ledger, never embedded history."""
from skill_memory import SkillMemory


def retrieve(path, fingerprint, options):
    # Reporting evidence remains on disk, but must not enter the learning view.
    ledger = SkillMemory(path)
    records, _, _ = ledger.view()
    omitted = set()
    for identifier, record in records.items():
        if (record.get("purpose") == "report_only"
                or record.get("source_role") in {"independent_validation", "final_test"}
                or "report_only_validation" in identifier):
            omitted.add(identifier)
    # Apply before ranking/budget truncation, and do not let a rule bypass the
    # separation merely by citing a report-only case.
    for identifier, record in records.items():
        supports = record.get('support', record.get('supports', []))
        supports = [supports] if isinstance(supports, str) else supports or []
        if any(str(ref) in omitted for ref in supports):
            omitted.add(identifier)
    result = ledger.retrieve(fingerprint, **{**options, 'exclude_ids': omitted})
    result['report_only_excluded_count'] = len(omitted)
    return result


def append(path, entries):
    return SkillMemory(path).append(entries)


def maintain(path, operations):
    return SkillMemory(path).maintain(operations)

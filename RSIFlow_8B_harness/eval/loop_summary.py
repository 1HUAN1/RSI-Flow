"""Summarize fixed/evolving Meta runs without feeding validation back into learning."""
import argparse
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def summarize(run):
    run = Path(run).resolve()
    rows = []
    rounds = sorted(run.glob('round_*'), key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else -1)
    for folder in rounds:
        attempts = sorted(folder.glob('attempts/attempt_*'), key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else -1)
        selections = [{'path': str(p / 'selection.json'), 'selection': read(p / 'selection.json'),
                       'decision': read(p / 'decision.json')} for p in attempts if (p / 'selection.json').is_file()]
        if not selections and (folder / 'selection.json').is_file():
            selections = [{'path': str(folder / 'selection.json'), 'selection': read(folder / 'selection.json'),
                           'decision': read(folder / 'decision.json')}]
        rows.append({'round': folder.name, 'attempt_count': len(selections), 'attempts': selections,
                     'meta_review': read(run / 'meta_harness/reviews' / (folder.name + '.json'))})
    validations = [{'path': str(p), 'metrics': read(p)} for p in sorted((run / 'snapshots').rglob('task_metrics.json'))]
    return {'run_dir': str(run), 'rounds': rows, 'validations': validations,
            'meta_program_state': read(run / 'meta_harness/state.json')}


def compare(fixed, evolving):
    return {'purpose': 'report_only', 'fixed_meta': summarize(fixed), 'evolving_meta': summarize(evolving),
            'interpretation': 'Compare matching task/evaluation sets and initial states; missing outcomes are not zero. Training gain is not proof of generalization or Meta causality.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-run', type=Path, required=True)
    parser.add_argument('--evolving-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.fixed_run, args.evolving_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

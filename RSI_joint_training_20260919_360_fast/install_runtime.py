"""Copy the server runtime locally on that server, then apply checked extensions.

Historical source, runs, checkpoints, budgets and receipts are never edited.
"""
import ast
import shutil
import re
from pathlib import Path
from common import ROOT, read, write, sha

def replace(source, old, new):
    if source.count(old)!=1:raise ValueError('Unsupported runtime revision at: '+old[:90])
    return source.replace(old,new,1)

def patched_files(source):
    out={}
    p='sia/task_meta/pipeline.py'; s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"Literal['stream', 'fixed_subset']", "Literal['stream', 'fixed_subset', 'full_cohort']")
    s=replace(s,"    def checked(self):", "    round_validation_config: str | None = None\n    training_timeout_seconds: int = Field(default=604800, ge=1)\n    allowed_task_components: list[Literal['HARNESS','MODEL','ARTIFACTS']] = Field(default_factory=lambda: ['HARNESS','MODEL','ARTIFACTS'])\n\n    def checked(self):")
    s=replace(s,"search_dev_fraction: float = Field(default=0.1, gt=0, lt=1)","search_dev_fraction: float = Field(default=0.1, ge=0, lt=1)")
    s=replace(s,"    def checked(self):\n", "    def checked(self):\n        if (self.training_schedule == 'full_cohort') != (self.search_dev_fraction == 0):\n            raise ValueError('Full cohort uses the entire train split and an in-training monitor')\n")
    s=replace(s,"            if self.experiment_scope != 'envscaler_validation':\n                raise ValueError('Parallel execution is currently scoped to EnvScaler validation')", "            # Each replica owns a process and an independent domain adapter.\n            if self.experiment_scope not in {'envscaler_validation','multidomain'}:\n                raise ValueError('Unsupported domain scope')")
    s=replace(s,"        store = ManifestStore(data_root / 'tasks.sqlite')", "        from sia.task_meta.runtime_extensions import FullTrainingStore\n        store = (FullTrainingStore if config.training_schedule == 'full_cohort' else ManifestStore)(data_root / 'tasks.sqlite')")
    s=replace(s,"command, timeout=7200,", "command, timeout=config.training_timeout_seconds + 1800,")
    s=replace(s,"    updaters = {action: DurableUpdater(updater, journal) for action, updater in updaters.items()}","    updaters = {action: DurableUpdater(updater, journal) for action, updater in updaters.items()\n                if action.value in config.allowed_task_components}")
    s=replace(s,"config.training_schedule == 'fixed_subset' else\n                   max(math.ceil", "config.training_schedule in {'fixed_subset','full_cohort'} else\n                   max(math.ceil")
    marker="            runner = run_sequential_task_meta\n"
    s=replace(s,marker,marker+"            if config.round_validation_config:\n                from sia.task_meta.round_validation import validate_round\n                runner_options['after_round'] = lambda number, record: validate_round(config, run_dir, number, record)\n")
    out[p]=s
    p='sia/task_meta/sequential_loop.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'before_evaluation=None):','before_evaluation=None, after_round=None):')
    expression="capabilities.update(updater_capabilities(parent, baseline, True, sft_profile=capabilities.get('sft_profile', 'multidomain')))"
    found=list(re.finditer(r'^([ ]*)'+re.escape(expression)+r'\n',s,re.M))
    if len(found)!=1:raise ValueError('Unsupported capability construction')
    match=found[0];indent=match.group(1)
    inserted=''.join(indent+line+'\n' for line in ("for action in TaskUpdateAction:","    if action not in updaters:","        capabilities[action.value] = {'available': False, 'reason': 'Component excluded by registered experiment scope'}"))
    s=s[:match.end()]+inserted+s[match.end():]
    s=replace(s,"            scores.append(record['score']); rounds.append(record); continue", "            scores.append(record['score']); rounds.append(record)\n            if after_round: after_round(number, record)\n            continue")
    s=replace(s,"        save_json(complete, record); scores.append(record['score']); rounds.append(record); task = after", "        save_json(complete, record); scores.append(record['score']); rounds.append(record); task = after\n        if after_round: after_round(number, record)")
    out[p]=s
    p='sia/task_meta/scoped_execution.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"        return list(pool.map(_rollout, jobs))", "        # Keep only one bounded chunk of queued payloads per worker.\n        # Output order and every task/seed stay identical.\n        from itertools import islice\n        iterator = iter(jobs); rows = []\n        while batch := list(islice(iterator, 4 * len(executor.replicas))):\n            rows.extend(pool.map(_rollout, batch))\n        return rows")
    out[p]=s
    p='sia/task_meta/pipeline_execution.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"                return cached['row']", "                from sia.task_meta.runtime_extensions import compact_rollout\n                return compact_rollout(cached['row'], path) if getattr(self.store, 'compact_rollouts', False) and not probe else cached['row']")
    s=replace(s,"        return row\n\n    def execute(self, state, directory):", "        from sia.task_meta.runtime_extensions import compact_rollout\n        return compact_rollout(row, path) if getattr(self.store, 'compact_rollouts', False) and not probe else row\n\n    def execute(self, state, directory):")
    s=replace(s,"[(t, n) for t in tasks for n in range(self.rollouts_per_task)]", "((t, n) for t in tasks for n in range(self.rollouts_per_task))")
    s=replace(s,"progress['evaluation_protocol'] = 'fixed_search_dev_readonly_T_in_v1'", "progress['evaluation_protocol'] = ('fixed_in_training_monitor_not_heldout_v1' if getattr(self.store, 'compact_rollouts', False) else 'fixed_search_dev_readonly_T_in_v1')")
    out[p]=s
    p='sia/task_meta/meta_harness/five_stage.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"    value['outcome_hash']=digest(value)", "    if protocol_file.exists() and _read(protocol_file).get('config',{}).get('training_schedule') == 'full_cohort':\n        value['feedback_scope'] = 'in_training_monitor_not_external_validation_no_gold'\n        value['train_windows'] = 'full_training_cohort_reused_each_round'\n        for pair in value['pairs']:\n            for side in ('before','after'):\n                if pair.get(side): pair[side]['split'] = 'in_training_monitor_readonly'\n    value['outcome_hash']=digest(value)")
    out[p]=s
    p='sia/task_meta/sft.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'    for row in trajectories:\n        try:', '    for row in trajectories:\n        if row.get("_full_rollout_ref"):\n            if row.get("verification",{}).get("success") is not True: continue\n            from sia.task_meta.runtime_extensions import hydrate_rollout\n            row = hydrate_rollout(row, for_sft=True)\n        try:')
    out[p]=s
    p='scripts/train_task_meta_sft.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'"max_steps": 8 if legacy else None,','"num_train_epochs": None, "max_steps": 8 if legacy else None,')
    s=replace(s,'    if result["max_steps"] is None:', '    if result["num_train_epochs"] is not None:\n        if result["num_train_epochs"] != 1 or result["max_steps"] != -1:\n            raise ValueError("One epoch requires num_train_epochs=1 and max_steps=-1")\n    if result["max_steps"] is None:')
    s=replace(s,'"max_steps": (1, 10000)', '"max_steps": (-1 if result["num_train_epochs"] == 1 else 1, 10000)')
    s=s.replace('"max_samples": (1, 100000)', '"max_samples": (1, 10000000)').replace('<= 100000:', '<= 10000000:').replace('[1, 100000]','[1, 10000000]')
    s=replace(s,'max_steps=options["max_steps"],\n        per_device', 'max_steps=options["max_steps"], num_train_epochs=options["num_train_epochs"] or 1,\n        per_device')
    s=replace(s,'    if trainer.state.global_step != options["max_steps"] or not math.isfinite(result.training_loss):\n        raise RuntimeError("Trainer did not complete the requested finite SFT steps")', '    from sia.task_meta.runtime_extensions import epoch_complete\n    epoch_complete(trainer.state.epoch, trainer.state.global_step, options["max_steps"], result.training_loss)')
    s=replace(s,'    delta_squared = 0.0','    if options["num_train_epochs"] == 1 and set(sampled_indices) != set(range(len(rows))):\n        raise RuntimeError("One-epoch SFT did not visit every eligible sample")\n    delta_squared = 0.0')
    s=replace(s,'        "world_size": world, "rank_records": rank_records,','        "num_train_epochs_completed": trainer.state.epoch,\n        "world_size": world, "rank_records": rank_records,')
    out[p]=s
    p='scripts/train_four_gpu_phase.py';s=(source/p).read_text(encoding='utf-8')
    out[p]=replace(s,'child.wait(timeout=5400)','child.wait(timeout=config.training_timeout_seconds)')
    p='sia/task_meta/observations.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'from sia.task_meta.types import MetaObservation','from sia.task_meta.types import MetaObservation\nfrom sia.task_meta.runtime_extensions import meta_sample, hydrate_rollout')
    s=replace(s,"    if row.get('execution_error_type', row.get('error_type')) != 'parse_error':", "    if '_action_contract_failure' in row: return row['_action_contract_failure']\n    if row.get('execution_error_type', row.get('error_type')) != 'parse_error':")
    old='    ids = sorted({r.get("question_id") for r in rows}, key=str)\n    stats["per_task"] = [{"question_id": qid, **aggregate([r for r in rows if r.get("question_id") == qid])}\n                         for qid in ids]'
    new='    groups = defaultdict(list)\n    for row in rows: groups[row.get("question_id")].append(row)\n    ids = sorted(groups, key=str)\n    stats["per_task"] = [{"question_id": qid, **aggregate(groups[qid])} for qid in ids[:96]]\n    stats["per_task_total"] = len(ids)\n    stats["per_task_details_omitted"] = max(0, len(ids)-96)'
    s=replace(s,old,new)
    s=replace(s,'    all_ids = [trajectory_id(row, index) for index, row in enumerate(rows)]','    full_rows = rows\n    rows = [hydrate_rollout(row) for row in meta_sample(rows)]\n    all_ids = [trajectory_id(row, index) for index, row in enumerate(rows)]')
    s=replace(s,'"total_trajectories": len(rows), "included_trajectory_ids"','"total_trajectories": len(full_rows), "sampled_trajectories": len(rows), "unsampled_trajectories": len(full_rows)-len(rows), "sampling_policy": "deterministic_domain_success_96", "included_trajectory_ids"')
    s=replace(s,'"character_budget": max_chars, "statistics": trajectory_statistics(rows)','"character_budget": max_chars, "statistics": trajectory_statistics(full_rows)')
    s=replace(s,'raw_trajectories=copy.deepcopy(result.trajectories) if retain_raw else []','raw_trajectories=copy.deepcopy(traces) if retain_raw else []')
    out[p]=s
    p='sia/task_meta/meta.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'        for index, item in enumerate(raw):','        from sia.task_meta.observations import trajectory_evidence\n        examples, coverage = trajectory_evidence(raw, max_chars=72000)\n        key = lambda row: (row.get("question_id",row.get("task_id")),row.get("rollout_id"))\n        selected = {key(row):row for row in examples}\n        sources[-1]["meta_sample_count"] = len(selected)\n        sources[-1]["meta_sample_policy"] = "deterministic_domain_success_96_bounded_72000_chars"\n        sources[-1]["meta_evidence_coverage"] = coverage\n        for index, item in enumerate(raw):\n            if key(item) not in selected: continue\n            item = selected[key(item)]')
    out[p]=s
    from protocol_patches import apply
    out=apply(out,source,replace)
    from inference_optimizations import apply as optimize_inference
    out=optimize_inference(out,source)
    for name,s in out.items():ast.parse(s,filename=name)
    return out

def install(config):
    source=Path(config['runtime_source']).resolve()
    target=ROOT/'runtime'
    receipt=ROOT/'runtime_install.json'
    if receipt.exists():
        for name,value in read(receipt)['installed_hashes'].items():
            if sha(target/name)!=value:raise ValueError('Installed source changed: '+name)
        return target
    if target.exists():raise RuntimeError('Incomplete install: inspect runtime before retrying')
    changes=patched_files(source)  # Verify all patch anchors before copying.
    # Copy on the same machine. No credentials, historical runs or logs are copied.
    staging=ROOT/'runtime.installing'
    if staging.exists():raise RuntimeError('Incomplete runtime staging exists')
    staging.mkdir()
    for name in ('sia','scripts','seed_harness','meta_harness','configs'):
        if (source/name).is_dir():
            shutil.copytree(source/name,staging/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.env'))
    shutil.copy2(source/'pyproject.toml',staging/'pyproject.toml')
    original={name:sha(source/name) for name in changes}
    for name,s in changes.items():(staging/name).write_text(s,encoding='utf-8',newline='\n')
    for name in ('runtime_extensions.py','round_validation.py','evolution_protocol.py','round_evolution.py'):
        shutil.copy2(ROOT/name,staging/'sia/task_meta'/name)
    staging.replace(target)
    names=[p.relative_to(target).as_posix() for p in (target/'sia/task_meta').rglob('*.py')]
    names += [p.relative_to(target).as_posix() for p in (target/'scripts').glob('*.py')]
    write(receipt,{'source':str(source),'source_hashes_before':original,
                  'installed_hashes':{name:sha(target/name) for name in names},'historical_run_modified':False})
    return target

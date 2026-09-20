"""Checked edits to the existing native entrypoints; installed only into a new runtime."""
def apply(out, source, replace):
    import re
    p='sia/task_meta/pipeline.py';s=out[p]
    s=replace(s,"'always_apply', 'sequential_positive_gain'", "'always_apply', 'sequential_positive_gain', 'legal_single_update_then_measure'")
    s=s.replace("config.task_update_policy == 'sequential_positive_gain'", "config.task_update_policy in {'sequential_positive_gain','legal_single_update_then_measure'}")
    s=replace(s,"final.get('task_update_policy') == 'sequential_same_parent_first_positive_v1'", "final.get('task_update_policy') in {'sequential_same_parent_first_positive_v1','legal_single_update_then_measure'}")
    s=replace(s,"'fixed_subset', 'full_cohort'", "'fixed_subset', 'full_cohort', 'round_disjoint'")
    s=replace(s,"    round_validation_config: str | None = None", "    round_protocol: dict | None = None\n    round_validation_config: str | None = None")
    s=replace(s,"(self.training_schedule == 'full_cohort') !=", "(self.training_schedule in {'full_cohort','round_disjoint'}) !=")
    s=replace(s,"    destination = project_path(config.data_dir)","    if config.training_schedule == 'round_disjoint':\n        from sia.task_meta.round_evolution import require_round_release\n        return require_round_release(config)\n    destination = project_path(config.data_dir)")
    s=replace(s,"        from sia.task_meta.runtime_extensions import FullTrainingStore", "        from sia.task_meta.round_evolution import RoundStore\n        from sia.task_meta.runtime_extensions import FullTrainingStore")
    s=replace(s,"(FullTrainingStore if config.training_schedule == 'full_cohort' else ManifestStore)","(RoundStore if config.training_schedule == 'round_disjoint' else FullTrainingStore if config.training_schedule == 'full_cohort' else ManifestStore)")
    s=replace(s,"{'fixed_subset','full_cohort'}", "{'fixed_subset','full_cohort','round_disjoint'}")
    s=replace(s,"            runner = run_sequential_task_meta\n", "            runner = run_sequential_task_meta\n            if config.training_schedule == 'round_disjoint':\n                from sia.task_meta.round_evolution import RoundProtocol\n                runner_options['round_protocol'] = RoundProtocol(config, run_dir, executor)\n")
    s=replace(s,"        final = runner(run_dir, initial_task", "        if config.training_schedule == 'round_disjoint' and config.round_validation_config and config.round_protocol.get('evaluate_initial_system',False):\n            from dataclasses import asdict\n            validate_round(config,run_dir,-1,{'task_after':asdict(initial_task),'meta_after':asdict(initial_meta),'status':'initial','chosen_component':None})\n        final = runner(run_dir, initial_task")
    s=replace(s,"    save_json(root / 'best_on_dev.json', frozen)", "    if protocol['config'].get('training_schedule') != 'round_disjoint':\n        save_json(root / 'best_on_dev.json', frozen)")
    out[p]=s
    p='sia/task_meta/gpu_phases.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'            return\n        stop_services()', '            verify_services(config, checkpoint)\n            return\n        stop_services()')
    s += '\n\ndef verify_services(config, checkpoint):\n    from sia.task_meta.storage import checkpoint_manifest\n    from sia.task_meta.evolution_protocol import fingerprint\n    if len(config.task_replicas) != 4 or {r["gpu"] for r in config.task_replicas} != {0,1,2,3}:\n        raise ValueError("Four distinct GPUs are required")\n    expected = {"checkpoint_path":str(Path(checkpoint).resolve()),"weights":checkpoint_manifest(checkpoint)}\n    opener=build_opener(ProxyHandler({})); records=[]\n    for replica in config.task_replicas:\n        with opener.open(replica["base_url"].removesuffix("/v1")+"/health",timeout=5) as response:health=json.load(response)\n        actual=health.get("bindings",{}).get(str(checkpoint))\n        if not health.get("ready") or actual!=expected or health.get("visible_devices")!=str(replica["gpu"]):\n            raise RuntimeError("Actual serving weights/GPU do not match frozen child")\n        records.append({"gpu":replica["gpu"],"base_url":replica["base_url"],"weights":actual,"model_fingerprint":fingerprint(actual)})\n    return records\n'
    out[p]=s
    p='sia/task_meta/rig.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'    records = []', '    if (root/"rollout_cache").exists():\n        physical = [json.loads(p.read_text())["row"] for p in (root/"rollout_cache").glob("*/train_rollouts/*.json")]\n        task_tokens = sum(r.get("input_tokens",0)+r.get("output_tokens",0) for r in physical)\n        task_calls = sum(r.get("model_call_count",0) for r in physical)\n        task_unknown = sum(r.get("unknown_usage_calls",0) for r in physical)\n    records = []')
    s=replace(s,"    save_json(root/'method2_progress.json', result)", "    if (root/'rollout_cache').exists():\n        result['generations'] = [json.loads(p.read_text()) for p in sorted(root.glob('round_*/adaptation_feedback_metrics.json'))]\n        result['metrics_kind'] = 'training_adaptation_feedback'\n        result['rig_scope'] = 'Compare the same full B_r parent/child passes; physical cache receipts charged once'\n    save_json(root/'method2_progress.json', result)")
    out[p]=s
    p='sia/task_meta/sequential_loop.py';s=out[p]
    s=replace(s,'        for index in range(3):','        for index in range(1 if round_protocol else 3):')
    s=replace(s,'after_round=None):','after_round=None, round_protocol=None):')
    s=replace(s,"    if primary_metric_name != 'macro_success'", "    if round_protocol and max_generations != 3: raise ValueError('Round protocol requires exactly three rounds')\n    if primary_metric_name != 'macro_success'")
    s=replace(s,"    def evaluate(state, directory):", "    def evaluate(state, directory):\n        if round_protocol:\n            round_protocol.bind_execution(state, directory, round_protocol.store.mode)")
    s=replace(s,"        task.generation = number", "        if round_protocol:\n            round_protocol.begin(number, meta, history)\n            round_protocol.store.mode = 'parent_pre_update'\n        task.generation = number")
    s=replace(s,"        baseline_dir = root / f'gen_{number}'", "        baseline_dir = (round_dir / 'before' if round_protocol else root) / f'gen_{number}'")
    s=replace(s,"        save_json(baseline_dir / 'meta_state_before.json', meta)", "        save_json(baseline_dir / 'meta_state_before.json', meta)\n        if round_protocol:\n            meta_agent.round_protocol = round_protocol\n            round_protocol.register_rows(baseline_dir/'agent_execution.json',number+1)\n            round_protocol.guard()")
    match=re.search(r'^( +)base_observation = build_observation',s,re.M)
    if not match:raise ValueError('Missing native observation construction')
    indent=match.group(1)
    inserted=''.join(indent+line+'\n' for line in ("if round_protocol and TaskUpdateAction.MODEL in updaters:",
        "    capabilities['MODEL']['available'] = True", "    capabilities['MODEL']['reason'] = 'Use only frozen pre-decision parent B_r successes; no additional SFT collection'"))
    s=s[:match.start()]+inserted+s[match.start():]
    s=s.replace('baseline, scores, history, [], capabilities','baseline, [] if round_protocol else scores, history, [], capabilities')
    s=replace(s,"immutable_task_policy=CONTRACT)", "immutable_task_policy=('Choose one legal untried component from the frozen parent. MODEL uses only existing parent B_r successful trajectories. A legal update is saved before child evaluation, without a positive-gain selection gate. Only one component decision is permitted; unavailable MODEL is skipped without reselection. Run the child once on the SAME full B_r; then update Meta harness from measured before/after experience. External validation never enters optimization.' if round_protocol else CONTRACT))")
    s=replace(s,"            _validate_decision(decision)","            _validate_decision(decision)\n            if round_protocol: round_protocol.decision(decision,round_dir/f'decision_{index}')")
    s=replace(s,"                        updater = DurableUpdater", "                        if round_protocol and decision.action == TaskUpdateAction.MODEL:\n                            round_protocol.model_context(parent,context,evaluate)\n                        updater = DurableUpdater")
    s=replace(s,"                    _assert_single_component(parent, candidate", "                    if round_protocol: round_protocol.check_task_edit(parent,candidate)\n                    _assert_single_component(parent, candidate")
    s=replace(s,"        result_dir = Path(chosen['trajectory_after']).parent if chosen else baseline_dir", "        retained_result = None\n        if round_protocol and not chosen:\n            retained_dir = round_dir/'after'/f'gen_{number+1}'\n            round_protocol.select_child(parent,after,last_decision,round_dir)\n            retained_result = evaluate(after,retained_dir)\n            write_evaluation(retained_dir,after,retained_result)\n        result_dir = Path(chosen['trajectory_after']).parent if chosen else retained_dir if retained_result else baseline_dir")
    s=replace(s,"outcome_result = EvaluationResult(chosen['performance'], [], chosen['cost']) if chosen else baseline", "outcome_result = EvaluationResult(chosen['performance'], [], chosen['cost']) if chosen else retained_result or baseline")
    s=replace(s,"        learned = meta_agent.learn_from_experience(meta, experience, history, after)",
        "        if round_protocol:\n            round_protocol.completed(None,round_dir,baseline,outcome_result)\n            round_protocol.prepare_meta(meta_agent,experience,round_dir)\n        learned = meta_agent.learn_from_experience(meta, experience, history, after)\n        if round_protocol: round_protocol.validate_meta(learned,meta)")
    s=replace(s,"                    if direct_submissions and decision.action", "                    if round_protocol: round_protocol.select_child(parent,candidate,decision,round_dir)\n                    if direct_submissions and decision.action")
    s=replace(s,"positive_gain=accepted)", "positive_gain=accepted, accepted_for_deployment=True if round_protocol else accepted)")
    s=replace(s,"if attempt.get('positive_gain'):", "if attempt.get('accepted_for_deployment',attempt.get('positive_gain')):")
    s=replace(s,"'accepted': attempt.get('positive_gain', False)", "'accepted': attempt.get('accepted_for_deployment',attempt.get('positive_gain', False))")
    s=s.replace("'rule': POLICY", "'rule': 'legal_single_update_then_measure' if round_protocol else POLICY")
    s=s.replace("policy=POLICY", "policy='legal_single_update_then_measure' if round_protocol else POLICY")
    s=replace(s,'candidate_limit=3, max_generations=max_generations',
        'candidate_limit=1 if round_protocol else 3, max_generations=max_generations')
    s=s.replace("'task_update_policy': POLICY", "'task_update_policy': 'legal_single_update_then_measure' if round_protocol else POLICY")
    s=replace(s,"'final_consolidation_status': 'last_round_memory_update_completed' if len(rounds) == max_generations else 'pending'", "'final_consolidation_status': ('third_round_meta_update_completed_no_fourth' if round_protocol else 'last_round_memory_update_completed') if len(rounds) == max_generations else 'pending'")
    s=replace(s,"        save_json(complete, record);", "        if round_protocol: round_protocol.freeze_system(record,round_dir)\n        save_json(complete, record);")
    out[p]=s
    p='sia/task_meta/meta_harness/five_stage.py';s=out[p]
    s=replace(s,"        if row.get('split') != 'search_dev':", "        training_probe = row.get('source_role') == 'train_evolution' and row.get('purpose') == 'evolution_train'\n        if row.get('split') != 'search_dev' and not (training_probe and row.get('split') == 'evolve_train'):")
    s=replace(s,"        item['split'] = 'internal_dev_readonly'", "        item['split'] = 'training_full_B' if training_probe else 'internal_dev_readonly'\n        if training_probe:\n            item.update(round_id=row['round_id'],manifest_hash=row['manifest_hash'])")
    s=replace(s,"        item['excluded_from_sft_and_task_assets'] = True", "        item['excluded_from_sft_and_task_assets'] = not (training_probe and row.get('collection_stage') == 'parent_pre_update')\n        if training_probe: item['source_id'] = f\"train_full_B:{row['round_id']}:{generation}:{row['task_id']}:{row['rollout_id']}\"")
    s=replace(s,"    value['outcome_hash']=digest(value)", "    if protocol_file.exists() and _read(protocol_file).get('config',{}).get('training_schedule') == 'round_disjoint':\n        if any(a.get('round_id') != experience.generation+1 or a.get('manifest_hash') != b.get('manifest_hash') for a,b in zip(before,after)):\n            raise ValueError('Training adaptation requires the same current-round B_r')\n        value['feedback_scope']='training_full_B_adaptation_not_held_out'\n        value['train_windows']='current_B_r_only'\n    value['outcome_hash']=digest(value)")
    out[p]=s
    p='sia/task_meta/meta.py';s=out[p]
    s=replace(s,'            envelope = operation_input(observation)',
        '            envelope = operation_input(observation)\n            if getattr(self,"round_protocol",None): self.round_protocol.enrich_decision(envelope)')
    s=replace(s,'            envelope = experience_input(experience, history, current_task_state, run_directory=self.run_directory)',
        '            envelope = experience_input(experience, history, current_task_state, run_directory=self.run_directory)\n            if getattr(self,"round_protocol",None): self.round_protocol.enrich_effect(envelope)')
    s=replace(s,'    def diagnose_and_route(self, meta_state, observation, feedback=None):', '    def diagnose_and_route(self, meta_state, observation, feedback=None):\n        protocol = getattr(self,"round_protocol",None)\n        if protocol: protocol.guard()')
    s=replace(s,'    def learn_from_experience(self, meta_state, experience, history, current_task_state):',
        '    def learn_from_experience(self, meta_state, experience, history, current_task_state):\n        protocol = getattr(self,"round_protocol",None)\n        if protocol: protocol.guard()')
    s=s.replace('validate_candidate=lambda candidate: validate_meta_candidate(candidate, meta_state),',
        'validate_candidate=lambda candidate: protocol.validate_meta(candidate,meta_state) if getattr(self,"round_protocol",None) else validate_meta_candidate(candidate, meta_state),',1)
    out[p]=s
    p='sia/task_meta/pipeline_execution.py';s=out[p]
    s=replace(s,"        if path.exists():\n            cached =", "        from sia.task_meta.round_evolution import scoped_rollout\n        directory, binding = scoped_rollout(self, state, task, rollout, directory, binding)\n        path = directory / f'{identifier}.json'\n        call_directory = directory / f'{identifier}.calls'\n        if path.exists():\n            cached =")
    s=replace(s,"        save_json(path, {'binding': binding, 'row': row})", "        from sia.task_meta.round_evolution import annotate_row\n        annotate_row(self,state,row)\n        save_json(path, {'binding': binding, 'row': row})")
    s=replace(s,"        if state.generation:\n", "        if state.generation and not getattr(self, 'round_protocol', None):\n")
    s=replace(s,"            notes = row.get('notes') or []", "            notes = [] if getattr(self, 'round_protocol', None) else row.get('notes') or []")
    s=replace(s,"directory / 'probe_rollouts', artifacts_text, probe=True", "directory / 'probe_rollouts', artifacts_text, probe=not bool(getattr(self, 'round_protocol', None))")
    s=replace(s,"        probe = self._batch(", "        probe = train if getattr(self,'round_protocol',None) else self._batch(")
    s=s.replace("for r in train + probe", "for r in (train if getattr(self,'round_protocol',None) else train + probe)")
    s=replace(s,"        return EvaluationResult(progress, train, costs, frozen, output, provenance,\n                                manifest_diff(frozen.artifacts.manifest, output.manifest))", "        result = EvaluationResult(progress, train, costs, frozen, output, provenance,\n                                manifest_diff(frozen.artifacts.manifest, output.manifest))\n        if getattr(self, 'round_protocol', None):\n            result = self.round_protocol.observed(result, directory)\n        return result")
    out[p]=s
    p='sia/task_meta/durable.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"        fingerprint = task_hash(task)\n", "        fingerprint = task_hash(task)\n        scope = Path(directory)/'execution_scope.json'\n        if scope.exists(): fingerprint = value_hash([fingerprint, digest(scope)])\n")
    out[p]=s
    p='sia/task_meta/sft.py';s=out[p]
    s=replace(s,'    for row in trajectories:\n', '    for row in trajectories:\n        if row.get("source_role") in {"independent_validation","final_test","evolution_feedback"} or row.get("purpose") == "report_only":\n            raise ValueError("External or reporting trajectories cannot enter SFT")\n')
    out[p]=s
    p='sia/task_meta/meta_harness/runtime.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,'        digest = _hash(row)', '        if row.get("source_role") in {"independent_validation","final_test","evolution_feedback"} or row.get("purpose") == "report_only":\n            raise ValueError("External evidence is forbidden for Meta")\n        digest = _hash(row)')
    s=replace(s,'            payload["principle_contract"] = {', '            feedback_protocol = envelope.get("trusted_facts",{}).get("feedback_protocol",{})\n            if feedback_protocol.get("purpose") == "evolution_train":\n                payload["protocol_phase"] = feedback_protocol["phase"]\n            payload["principle_contract"] = {')
    s=replace(s,'            outcome = envelope.get("outcome_review")\n            payload["principle_contract"]["required_review_binding"] = {',
        '            memory_contract = envelope.get("trusted_facts",{}).get("memory_update_contract")\n            if memory_contract:\n                payload["principle_contract"].update(maintenance=memory_contract["instructions"],\n                    id_contract="Only ADD new skill.<COMPONENT>.<id> followed by principle.<id>; preserve every existing ID and record.",\n                    targets="Keep g_targets and harness_bindings empty; consume appended Memory in the next round.")\n            outcome = envelope.get("outcome_review")\n            payload["principle_contract"]["required_review_binding"] = {')
    out[p]=s
    p='sia/task_meta/updaters.py';s=(source/p).read_text(encoding='utf-8')
    s=replace(s,"        positives = select_positive_rows(context.evaluation.trajectories, profile=self.sft_profile)","        from sia.task_meta.round_evolution import select_sft\n        positives = select_sft(self,task_state,context)")
    s=replace(s,"        lengths = eligibility['eligible_token_lengths']", "        minimum = self.training.get('round_protocol',{}).get('minimum_sft_samples',1)\n        if len(positives) < minimum:\n            raise DecisionConstraintError('MODEL/SFT unavailable: eligible examples below fixed minimum')\n        lengths = eligibility['eligible_token_lengths']")
    s=replace(s,'            "meta_request_plan": meta_plan, "length_eligibility": eligibility,', '            "meta_request_plan": meta_plan, "length_eligibility": eligibility,\n            "source_binding": json.loads((context.directory/"sft_source_binding.json").read_text()) if self.training.get("round_protocol") else None,')
    out[p]=s
    p='scripts/train_task_meta_sft.py';s=out[p]
    s=replace(s,'    return rows\n', '    from sia.task_meta.round_evolution import validate_trainer_rows\n    validate_trainer_rows(rows,request)\n    return rows\n')
    s=replace(s,'defaults = {"num_train_epochs": None,', 'defaults = {"finetuning_type":"lora", "train_base_weights":False, "replay_previous_rounds":False, "round_protocol":None, "lora_alpha":16, "lora_dropout":0.05, "num_train_epochs": None,')
    s=replace(s,'    model.config.use_cache = False', '    from sia.task_meta.round_evolution import check_lora_modules\n    check_lora_modules(model,options)\n    model.config.use_cache = False')
    s=replace(s,'lora_alpha=2 * options["lora_rank"], lora_dropout=0.0,','lora_alpha=options["lora_alpha"], lora_dropout=options["lora_dropout"],')
    s=replace(s,'    if not trainable_before:', '    if any("lora_" not in name for name in trainable_before):\n        raise RuntimeError("Only the new LoRA may be trainable")\n    if not trainable_before:')
    s=replace(s,'        "num_train_epochs_completed": trainer.state.epoch,','        "num_train_epochs_completed": trainer.state.epoch,\n        "source_binding": request.get("source_binding"), "resolved_training": options,\n        "optimizer_reset": True, "scheduler_reset": True, "parent_initialization": str(base),')
    out[p]=s
    return out

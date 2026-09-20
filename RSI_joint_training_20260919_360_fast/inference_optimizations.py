"""Scheduling/CPU preparation only: retain model, sampling, seeds and GPU lock."""


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected inference source at: ' + old[:80])
    return text.replace(old, new, 1)


def apply(files, source):
    path = 'sia/task_meta/scoped_execution.py'
    text = files[path]
    text = replace(text, 'import multiprocessing\n', 'import multiprocessing\nimport os\n')
    text = replace(text, 'from concurrent.futures import ProcessPoolExecutor',
                   'from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED')
    text = replace(text, 'sources, counter):', 'sources, counter, request_slots):')
    text = replace(text, "    executor._replica_endpoint = executor.replicas[index]['base_url']", '''    replica = index % len(executor.replicas)
    executor._replica_endpoint = executor.replicas[replica]['base_url']
    original_factory = executor.model_factory
    slot = request_slots[replica]
    def limited_factory(task_state, base_url=None):
        client = original_factory(task_state, base_url=base_url)
        def complete(*args, **kwargs):
            # Bound HTTP queueing; GPU generation and request seeds stay serialized.
            with slot:
                return client(*args, **kwargs)
        complete.enable_thinking = getattr(client, 'enable_thinking', False)
        return complete
    executor.model_factory = limited_factory''')
    start = text.index('def run_parallel(')
    text = text[:start] + '''def bounded_results(pool, function, jobs, capacity):
    """Refill on any completion; preserve result order without a batch barrier."""
    if capacity < 1:
        raise ValueError('Positive scheduling capacity required')
    iterator = iter(enumerate(jobs))
    pending, results = {}, {}
    exhausted = False
    while pending or not exhausted:
        while not exhausted and len(pending) < capacity:
            try:
                index, job = next(iterator)
            except StopIteration:
                exhausted = True
                break
            pending[pool.submit(function, job)] = index
        if pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                results[pending.pop(future)] = future.result()
    return [results[i] for i in range(len(results))]


def worker_count(replicas):
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    return replicas * min(4, max(1, cores // (2 * replicas)))


def run_parallel(executor, state, spec, jobs, directory, assets, probe, sources):
    context = multiprocessing.get_context('fork')
    counter = context.Value('i', 0)
    workers = worker_count(len(executor.replicas))
    request_slots = [context.BoundedSemaphore(2) for _ in executor.replicas]
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_initialize, initargs=(executor, state, spec, directory, assets, probe, sources,
                                             counter, request_slots)) as pool:
        return bounded_results(pool, _rollout, jobs, 2 * workers)
'''
    files[path] = text
    path = 'sia/task_meta/serve_gpu.py'
    text = (source / path).read_text(encoding='utf-8')
    text = replace(text, 'lock = threading.Lock()', 'lock = threading.Lock()\ntokenizer_lock = threading.Lock()')
    text = replace(text, '''    with lock, torch.inference_mode():
        if request.model not in state:
            raise HTTPException(404, "Unknown or evicted model; register the checkpoint first")
        tokenizer, model = state[request.model]
        response_binding = bindings[request.model]
        inputs = tokenizer.apply_chat_template(''', '''    selected = state.get(request.model)
    if selected is None:
        raise HTTPException(404, "Unknown or evicted model; register the checkpoint first")
    tokenizer, model = selected
    # CPU tokenization can overlap an earlier request's GPU generation.
    with tokenizer_lock:
        inputs = tokenizer.apply_chat_template(''')
    text = replace(text, '''        ).to(DEVICE)
        prompt_tokens = inputs["input_ids"].shape[1]
        limit = min(int(os.environ.get("TASK_META_CONTEXT_TOKENS", "32768")), model.config.max_position_embeddings)
        if prompt_tokens + request.max_tokens > limit:
            raise HTTPException(400, "Request exceeds the configured Task context budget")
        if request.seed is not None:''', '''        )
    prompt_tokens = inputs["input_ids"].shape[1]
    limit = min(int(os.environ.get("TASK_META_CONTEXT_TOKENS", "32768")), model.config.max_position_embeddings)
    if prompt_tokens + request.max_tokens > limit:
        raise HTTPException(400, "Request exceeds the configured Task context budget")
    with lock, torch.inference_mode():
        if state.get(request.model) is not selected:
            raise HTTPException(409, "Checkpoint changed while preparing request inputs")
        response_binding = bindings[request.model]
        inputs = inputs.to(DEVICE)
        if request.seed is not None:''')
    files[path] = text
    path = 'sia/task_meta/sandbox.py'
    text = files.get(path, (source / path).read_text(encoding='utf-8'))
    text = replace(text, '                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))\n', '')
    text = replace(text, '                resource.setrlimit(resource.RLIMIT_AS, (self.limits.memory_bytes, self.limits.memory_bytes))',
                   '                # Finish trusted guard setup before capping inherited pool descriptors.\n'
                   '                # close_fds still removes them before candidate execution.\n'
                   '                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))\n'
                   '                resource.setrlimit(resource.RLIMIT_AS, (self.limits.memory_bytes, self.limits.memory_bytes))')
    files[path] = text
    return files

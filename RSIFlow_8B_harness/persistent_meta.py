"""One live Codex app-server; native tools execute the Meta's experiment requests.

The bridge carries tool calls/results, not component or acceptance decisions.
Long tools are durable jobs so Codex can inspect their logs while they run.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from experiment_progress import ExperimentProgress
from meta_loop import MAINLINE_REMINDER, render_receipt_for_codex
from skill_memory import SKILL_GUIDANCE
from meta_chat import ChatMailbox


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def process_identity(pid: int) -> str | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


class CodexAppServer:
    """JSON-RPC connection; the same server PID handles every native tool call."""

    def __init__(self, *, codex_home: Path, workspace: Path, journal: Path,
                 executable: str = 'codex', on_tool=None, chat=None):
        self.home, self.workspace, self.journal = codex_home, workspace, journal
        self.executable = os.environ.get("RSIFLOW_CODEX_EXECUTABLE", executable)
        self.on_tool = on_tool
        self.inbox = queue.Queue()
        self.sequence = 0
        self.responses = {}
        self.completed_turns = {}
        self.process = None
        self.chat = chat
        self.active_turn = None
        self.chat_requests = {}

    def start(self):
        if self.chat:
            self.chat.recover()
        self.journal.mkdir(parents=True, exist_ok=True)
        self.stderr = (self.journal / 'stderr.log').open('a', encoding='utf-8')
        self.events = (self.journal / 'events.jsonl').open('a', encoding='utf-8', buffering=1)
        temporary = self.journal / 'tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CODEX_HOME=str(self.home), TMPDIR=str(temporary))
        self.process = subprocess.Popen(
            [self.executable, 'app-server', '--listen', 'stdio://'], cwd=self.workspace,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1, env=env,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.request('initialize', {'clientInfo': {'name': 'rsiflow_meta', 'version': '1.0'},
                                    'capabilities': {'experimentalApi': True}})
        self.send({'method': 'initialized'})
        return self

    def _read(self):
        try:
            for line in self.process.stdout:
                self.events.write(line)
                try:
                    self.inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        finally:
            self.inbox.put(None)

    def send(self, message):
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
        self.process.stdin.flush()

    def _send_chat(self):
        if not self.chat or self.active_turn is None:
            return
        thread_id, turn_id = self.active_turn
        for record in self.chat.pending(turn_id):
            identifier = 'chat:' + record['id']
            record = self.chat.save({**record, 'status': 'sending', 'thread_id': thread_id,
                                     'last_attempt_turn': turn_id, 'detail': None})
            self.chat_requests[identifier] = record
            self.send({'id': identifier, 'method': 'turn/steer', 'params': {
                'threadId': thread_id, 'expectedTurnId': turn_id,
                'clientUserMessageId': record['id'],
                'input': [{'type': 'text', 'text': record['text']}]}})

    def receive(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._send_chat()
            wait = None if deadline is None else max(0, deadline-time.monotonic())
            if self.chat:
                wait = min(wait, .5) if wait is not None else .5
            try:
                event = self.inbox.get(timeout=wait)
                break
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise
        if event is None:
            raise RuntimeError(f'Codex app-server exited; see {self.journal / "stderr.log"}')
        if 'id' in event and 'method' in event:
            if event['method'] == 'item/tool/call':
                params = event['params']
                try:
                    value = self.on_tool(params['tool'], params['arguments'], params)
                    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    result = {'contentItems': [{'type': 'inputText', 'text': text}], 'success': True}
                except Exception as exc:
                    result = {'contentItems': [{'type': 'inputText', 'text': json.dumps({
                        'status': 'tool_error', 'error_type': type(exc).__name__, 'error': str(exc)})}],
                        'success': False}
                self.send({'id': event['id'], 'result': result})
            else:
                self.send({'id': event['id'], 'error': {'code': -32601,
                           'message': 'Unsupported app-server request: ' + event['method']}})
        elif event.get('id') in self.chat_requests:
            record = self.chat_requests.pop(event['id'])
            if 'error' in event:
                detail = str(event['error'].get('message', event['error']))
                stale = any(part in detail.lower() for part in ('no active turn', 'turn id', 'turn_id', 'turn mismatch'))
                self.chat.save({**record, 'status': 'queued' if stale else 'delivery_error', 'detail': detail})
            else:
                self.chat.save({**record, 'status': 'accepted', 'accepted_at': time.time(),
                                'turn_id': event['result']['turnId']})
        elif 'id' in event:
            self.responses[event['id']] = event
        elif event.get('method') == 'turn/started':
            params = event['params']
            self.active_turn = (params['threadId'], params['turn']['id'])
        elif event.get('method') == 'turn/completed':
            params = event['params']
            key = (params['threadId'], params['turn']['id'])
            self.completed_turns[key] = params['turn']
            if self.active_turn == key:
                self.active_turn = None
        return event

    def request(self, method, params):
        self.sequence += 1
        identifier = self.sequence
        self.send({'id': identifier, 'method': method, 'params': params})
        while identifier not in self.responses:
            self.receive(timeout=120)
        reply = self.responses.pop(identifier)
        if 'error' in reply:
            raise RuntimeError(f'{method}: {reply["error"]}')
        return reply['result']

    def turn(self, thread_id, prompt, *, environments=None):
        parameters = {'threadId': thread_id, 'input': [{'type': 'text', 'text': prompt}]}
        if environments is not None:
            parameters['environments'] = environments
        reply = self.request('turn/start', parameters)
        turn_id = reply['turn']['id']
        key = (thread_id, turn_id)
        if key not in self.completed_turns:
            self.active_turn = key
        while key not in self.completed_turns:
            self.receive()
        return self.completed_turns.pop(key)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10)
            self.stderr.close()
            self.reader.join(timeout=2)
            if not self.reader.is_alive():
                self.events.close()


class ExperimentJobs:
    """Durable host-side jobs, including adoption of the already-running rollout."""

    def __init__(self, project: Path, run: Path):
        self.project, self.run = project, run
        self.root = run / 'meta_session/jobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = {}

    def all(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob('*/job.json'))]

    def start(self, request, turn_dir):
        for job in self.all():
            if job['request'] == request and self.status(job['job_id'])['status'] == 'running':
                return self.status(job['job_id'])
        identifier = uuid.uuid4().hex
        directory = self.run / 'controller_transport' / identifier
        directory.mkdir(parents=True)
        request_path = directory / 'request.json'
        write_json(request_path, request)
        result_path = directory / 'result.json'
        argv = [sys.executable, str(self.project / 'controller_tools.py'),
                '--workspace', str(self.project), '--receipts', str(self.run / 'controller_receipts'),
                '--result-file', str(result_path)]
        with request_path.open() as source, (directory / 'stdout.log').open('w') as stdout, \
                (directory / 'stderr.log').open('w') as stderr:
            process = subprocess.Popen(argv, stdin=source, stdout=stdout, stderr=stderr,
                                       cwd=self.project, start_new_session=True,
                                       env={k: v for k, v in os.environ.items() if k != 'AUTODL_API_KEY'})
        self.children[identifier] = process
        job = {'job_id': identifier, 'request': request, 'pid': process.pid,
               'process_start': process_identity(process.pid), 'created_at': time.time(),
               'turn_dir': str(turn_dir), 'result_path': str(result_path),
               'stdout_path': str(directory / 'stdout.log'), 'stderr_path': str(directory / 'stderr.log')}
        write_json(self.root / identifier / 'job.json', job)
        return self.status(identifier)

    def status(self, identifier):
        job = json.loads((self.root / identifier / 'job.json').read_text())
        process = self.children.get(identifier)
        if process is not None:
            process.poll()
        result_path = Path(job['result_path'])
        info = {key: job[key] for key in ('job_id', 'pid', 'result_path', 'stdout_path', 'stderr_path')}
        info['operation'] = job['request']['operation']
        info['elapsed_seconds'] = round(time.time() - job['created_at'], 1)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            receipt = Path(job['turn_dir']) / 'receipt.json'
            if not receipt.exists() or json.loads(receipt.read_text()).get('status') == 'running':
                write_json(receipt, result)
            return {**info, 'status': 'completed', 'result': result, 'receipt_path': str(receipt)}
        alive = job.get('process_start') is not None and process_identity(job['pid']) == job['process_start']
        if alive:
            return {**info, 'status': 'running'}
        result = {'operation': info['operation'], 'status': 'job_failed',
                  'error': 'Tool process ended without a result file.', **info}
        write_json(Path(job['turn_dir']) / 'receipt.json', result)
        return result

    def wait(self, identifier, seconds=60):
        deadline = time.monotonic() + min(max(float(seconds), 0), 60)
        while True:
            result = self.status(identifier)
            if result['status'] != 'running' or time.monotonic() >= deadline:
                return result
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    def adopt_legacy(self, state):
        """Attach the old pending call to its live worker; never restart that call."""
        if state.get('status') != 'tool_inflight':
            return []
        turn_dir = self.run / 'meta_session/turns' / f'{state["turn"]:04d}'
        request = json.loads((turn_dir / 'command.json').read_text())['request']
        candidates = []
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv = path.read_bytes().decode().strip('\0').split('\0')
                if str(self.project / 'controller_tools.py') not in argv or '--result-file' not in argv:
                    continue
                result = Path(argv[argv.index('--result-file') + 1])
                if not result.is_relative_to(self.run / 'controller_transport'):
                    continue
                pid = int(path.parent.name)
                candidates.append((pid, result))
            except (OSError, ValueError, IndexError, UnicodeError):
                continue
        if not candidates:
            return []
        # Forked rollout workers inherit argv; the original controller has the lowest PID.
        pid, result = min(candidates)
        identifier = result.parent.name
        job = {'job_id': identifier, 'request': request, 'pid': pid,
               'process_start': process_identity(pid), 'created_at': (turn_dir / 'command.json').stat().st_mtime,
               'turn_dir': str(turn_dir), 'result_path': str(result),
               'stdout_path': str(result.parent / 'stdout.log'), 'stderr_path': str(result.parent / 'stderr.log'),
               'adopted': True}
        write_json(self.root / identifier / 'job.json', job)
        return [self.status(identifier)]


RECOMMENDED_OPERATIONS = {
    'bootstrap', 'run_parent', 'run_candidate', 'rollout', 'prepare_meta_evidence',
    'sft', 'harnessforge', 'artifacts', 'evaluate', 'prepare_evaluation', 'read_json',
    'read_text', 'write_text', 'compare_scores', 'activate_task', 'append_skills',
    'snapshot', 'snapshot_task_meta', 'prepare_validation_snapshot', 'materialize_harness',
    'run_command', 'apply_patch', 'compose_task',
    'retrieve_skills', 'read_skill', 'rebuild_skills_index', 'record_skill_use', 'compare_task_differences', 'maintain_skills',
    'skill_usage_report',
    'meta_harness_status', 'materialize_meta_harness', 'check_meta_harness',
    'update_meta_harness', 'prepare_meta_review',
}
LONG_OPERATIONS = {'run_parent', 'run_candidate', 'rollout', 'sft', 'evaluate', 'run_command'}


def dynamic_tools():
    definitions = [
        ('experiment', 'Execute an RSIFlow helper. request contains operation plus its named arguments. '
         'Read controller_tools.py and task_adapter.py for tool fields. Long rollout/SFT/evaluate calls '
         'return a job_id. You may fix or add implementations and use native Codex tools directly. '
         'Common operations (not a whitelist): ' + ', '.join(sorted(RECOMMENDED_OPERATIONS)),
         {'request': {'type': 'object', 'properties': {'operation': {'type': 'string'}},
                      'required': ['operation'], 'additionalProperties': True}}, ['request']),
        ('job_status', 'Inspect a job, or list all jobs when job_id is omitted. Returns facts and log paths.',
         {'job_id': {'type': 'string'}}, []),
        ('wait_job', 'Wait up to 60 seconds for a job while the Codex process stays alive. '
         'Use seconds=60 during rollout, then inspect relevant logs or wait again.',
         {'job_id': {'type': 'string'}, 'seconds': {'type': 'number'}}, ['job_id']),
        ('finish_experiment', 'Declare completion after all configured evolution rounds have their required products. '
         'Missing products are returned to you; the process remains available to finish the work.',
         {'summary': {'type': 'string'}}, ['summary']),
    ]
    return [{'type': 'function', 'name': name, 'description': description,
             'inputSchema': {'type': 'object', 'properties': properties,
                             'required': required, 'additionalProperties': False}}
            for name, description, properties, required in definitions]


class NativeExperimentTools:
    """Optional experiment helpers alongside native Codex coding tools."""

    def __init__(self, *, project, run, executor, progress):
        self.project, self.run, self.executor, self.progress = project, run, executor, progress
        self.workspace = project.resolve().parent
        self.jobs = ExperimentJobs(project, run)
        self.finished = False
        self.summary = None
        self.anchor = None
        from meta_harness_runtime import MetaHarnessRuntime
        self.meta_program = MetaHarnessRuntime(project, run)
        turns = run / 'meta_session/turns'
        turns.mkdir(parents=True, exist_ok=True)
        self.number = max((int(p.name) for p in turns.iterdir() if p.name.isdigit()), default=-1) + 1

    def _request(self, request, turn_dir):
        operation = request.get('operation')
        # Newly implemented operations go straight to the freshly loaded tool library.
        write_fields = {'output_dir', 'destination', 'active_state'}
        if operation in {'write_text', 'append_skills'}:
            write_fields.add('path')
        if operation == 'bootstrap':
            write_fields.add('state_path')
        request = dict(request)
        stage = self.progress.rebuild().get('current_stage', 'A0')
        number = int(stage[1:]) if stage.startswith('B') and stage[1:].isdigit() else 0
        request.setdefault('_meta_round', request.get('round_number', request.get('round', number)))
        for key in write_fields:
            if key not in request:
                continue
            target = Path(request[key])
            target = (self.run / target).resolve() if not target.is_absolute() else target.resolve()
            if not target.is_relative_to(self.workspace):
                return {'status': 'tool_error', 'error': 'Write files within the experiment workspace.',
                        'workspace': str(self.workspace), 'path': str(target)}
            request[key] = str(target)
        if operation in LONG_OPERATIONS:
            return self.jobs.start(request, turn_dir)
        return self.executor.execute(request)

    def __call__(self, tool, arguments, params=None):
        turn_dir = self.run / 'meta_session/turns' / f'{self.number:04d}'
        self.number += 1
        turn_dir.mkdir(parents=True, exist_ok=True)
        request = arguments.get('request', {}) if tool == 'experiment' else {'operation': tool, **arguments}
        write_json(turn_dir / 'command.json', {'action': 'tool', 'request': request,
                   'native_call_id': (params or {}).get('callId'), 'transport': 'app_server'})
        try:
            if tool == 'experiment':
                receipt = self._request(request, turn_dir)
            elif tool == 'job_status':
                identifier = arguments.get('job_id')
                receipt = (self.jobs.status(identifier) if identifier else
                           {'status': 'read', 'jobs': [self.jobs.status(j['job_id']) for j in self.jobs.all()]})
            elif tool == 'wait_job':
                receipt = self.jobs.wait(arguments['job_id'], arguments.get('seconds', 60))
            elif tool == 'finish_experiment':
                state = self.progress.rebuild()
                self.finished = state['finish_allowed']
                self.summary = arguments.get('summary') if self.finished else None
                receipt = {'status': 'complete' if self.finished else 'incomplete',
                           'remaining': {s['id']: s['remaining_milestones'] for s in state['stages']
                                         if s['remaining_milestones']}}
            else:
                receipt = {'status': 'unknown_tool', 'tool': tool}
        except Exception as exc:
            receipt = {'status': 'tool_error', 'error_type': type(exc).__name__, 'error': str(exc)}
        # A wait/query must not masquerade as the original rollout operation in the ledger.
        receipt = {**receipt, 'operation': request.get('operation')}
        recorded = (receipt['result'] if tool == 'experiment' and request.get('operation') in LONG_OPERATIONS
                    and receipt.get('status') == 'completed' else receipt)
        write_json(turn_dir / 'receipt.json', recorded)
        state = self.progress.record_turn(turn_dir)
        anchor = (state['current_stage'], state['next_milestone'])
        failed = str(receipt.get('status', '')).endswith(('error', 'failed'))
        text = render_receipt_for_codex(receipt, turn_dir / 'receipt.json')
        if anchor != self.anchor or failed:
            text += '\n' + MAINLINE_REMINDER + '\n' + self.progress.compact_status_block()
            text += '\n' + self.phase_instructions(state, receipt)
            self.anchor = anchor
        return text

    def phase_instructions(self, state, receipt=None):
        """Actually execute the round's three-file program at major phase boundaries."""
        stage = state.get('current_stage', 'A0')
        number = int(stage[1:]) if stage.startswith('B') and stage[1:].isdigit() else 0
        milestone = state.get('next_milestone')
        phase = ('route' if milestone in {'meta_evidence', 'decision_recorded', 'component_reselection'}
                 else 'candidate' if milestone in {'candidate_built', 'candidate_rollout'}
                 else 'feedback' if milestone in {'scores_compared', 'selection_recorded', 'skills_appended'}
                 else 'meta_review' if milestone in {'context_updated', 'task_meta_snapshot'}
                 else 'execution')
        context = {'round': number, 'phase': phase, 'stage': stage, 'next_milestone': milestone,
                   'run_dir': str(self.run), 'receipt': receipt or {},
                   'evidence_paths': [str(self.run / f'round_{number}'),
                                      str(self.run / 'meta/skills.jsonl'),
                                      str(self.run / 'meta/context.json'),
                                      str(self.journal_path)]}
        try:
            policy = self.meta_program.phase(context)
            path = self.run / 'meta_harness/phase_context.json'
            write_json(path, {'context': context, 'policy': policy})
            return ('Active round-pinned Meta program (use this version, not historical guidance):\n'
                    + render_receipt_for_codex(policy, path, max_inline_bytes=24000))
        except Exception as exc:
            return f'Meta program hook failed: {type(exc).__name__}: {exc}. Diagnose it and continue the same milestone; original evidence is unchanged.'

    @property
    def journal_path(self):
        return self.run / 'meta_session/app_server/events.jsonl'


PERSISTENT_META_INSTRUCTIONS = """You are the Meta agent responsible for the complete Task/Meta experiment.
The same Codex app-server process/thread remains alive across helpers, waits and all rounds.
Native shell and apply_patch are available; helper implementations can be repaired within this
experiment workspace. This supersedes earlier read-only-tool instructions.
You may add tools within the rsiH workspace for scoped debugging, not change the research protocol.
Each invocation reloads controller_tools.py; new operations are not whitelisted.
Fixed research protocol: Qwen3-4B, configured GPU replicas and CPU-bounded rollout workers, fresh round-disjoint 180-task
batches, same-round paired retests, verified-success-only one-epoch MODEL LoRA SFT on all configured GPUs,
configured direct-submission ARTIFACTS edits. HARNESS follows three HarnessForge production
stages: source/trajectory-based localization, improvement direction, full candidate package;
then executable validation/limited repair. Calling the validator alone is NOT HarnessForge production.
Write both reports BEFORE candidate edits and pass their paths to harnessforge; check at most 3 times
with repair between checks of the SAME candidate. Full bundle replaces the parent's bundle.
One candidate per attempt. You own component selection, candidate contents and acceptance.
Complete comparable paired success rates must strictly improve for Task activation.
Rejection returns to component selection on the SAME parent and batch: append failure experience,
use a fresh attempts/attempt_K directory, do not go to validation or the next round.
There is no fixed retry limit; unchanged parent rollout can be reused.
After each attempt append skill.HARNESS.<id>, skill.MODEL.<id> or skill.ARTIFACTS.<id> cases.
General principle.<id> records cite supporting cases. Preserve measured outcome, applicable
boundaries and original rollout path, task ID and content hash. Never copy a full trajectory
into skills. The next attempt/round reads relevant entries and the structured handoff.
Independent evaluation is report-only: never use it for acceptance, SFT or Meta learning.
Save numbered Task components, skill versions and the small handoff; no repeated model weight copies.
The Meta improvement METHOD comes from the round-pinned workflow.py, planning.py, memory.py package,
not a permanent copy of procedural instructions in this message. Active phase-hook outputs identify
the loaded version and supersede historical method guidance. Memory helpers execute that version's
Python functions, not just text. After Task feedback and skill maintenance, inspect your own
decision/read/tool trace with prepare_meta_review. You decide keep/replace: no automatic trigger.
For a program gap, materialize_meta_harness copies the active three files; edit only necessary
behavior, check_meta_harness checks executable wiring, update_meta_harness queues the whole package
for the NEXT round. No Meta gain gate. Stable Codex engine, scoring, transport/loaders and research
protocol are not evolution targets. Skills and evidence remain outside the program package.
Long helpers return job_id; wait_job/job_status retain this conversation. Do not duplicate running
jobs. Tool errors are observations: read facts, debug/retry the affected step, return to the mainline.
Finish only through finish_experiment when required round products and validations exist.
Live user chat belongs to this same experiment, not a request to start another one.
"""


def native_thread_parameters(workspace: Path):
    """This server cannot mount bubblewrap; use Codex's supported Landlock compatibility mode.

    Keep repository metadata protected. Temporary writes stay inside the data-disk workspace.
    The compatibility mode is tested against the installed 0.154.0-alpha.6.2 binary.
    """
    return {'model': 'DeepSeek-V4.1-Flash', 'modelProvider': 'autodl',
            'cwd': str(workspace), 'approvalPolicy': 'never', 'sandbox': 'workspace-write',
            'developerInstructions': PERSISTENT_META_INSTRUCTIONS,
            'config': {'features.shell_tool': True, 'features.use_legacy_landlock': True,
                       'features.multi_agent': False, 'features.multi_agent_v2': False,
                       'sandbox_workspace_write.network_access': True,
                       'sandbox_workspace_write.exclude_slash_tmp': True,
                       'sandbox_workspace_write.exclude_tmpdir_env_var': True}}


class PersistentMeta:
    def __init__(self, *, project, run, codex_home, executor, rounds=5):
        self.project, self.run, self.codex_home = project, run, codex_home
        self.workspace = project.resolve().parent
        self.journal = run / 'meta_session'
        self.progress = ExperimentProgress(self.journal, rounds=rounds)
        self.tools = NativeExperimentTools(project=project, run=run, executor=executor, progress=self.progress)

    def run_experiment(self, prompt):
        state_path = self.journal / 'state.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        native = previous.get('transport') == 'app_server'
        progress_state = self.progress.rebuild()
        if native and previous.get('status') == 'complete' and progress_state['finish_allowed']:
            return previous
        if native and previous.get('status') == 'complete':
            self.tools.finished = False
        if previous and not native:
            write_json(self.journal / 'legacy_cli_state.json', previous)
            self.tools.jobs.adopt_legacy(previous)
        self.progress.rebuild()
        prompt += '\n' + self.tools.phase_instructions(self.progress.rebuild())
        state = {**(previous if native else {}), 'transport': 'app_server', 'status': 'starting',
                 'launcher_pid': os.getpid(), 'workspace_root': str(self.workspace),
                 'capability_revision': 'native_coding_v1', 'interactive_chat': True,
                 'chat_path': str(self.journal / 'chat')}
        if previous and not native:
            state['previous_thread_id'] = previous.get('thread_id')
        thread_id = state.get('thread_id')
        handoff = {'previous_thread_id': state.get('previous_thread_id'),
                   'jobs': [self.tools.jobs.status(j['job_id']) for j in self.tools.jobs.all()],
                   'prior_tool_calls': str(self.journal / 'turns'),
                   'full_progress': str(self.journal / 'workflow_state.json'),
                   'progress': self.progress.compact_status_block()}
        prompt += '\nExisting experiment handoff (reuse completed work and attach to running jobs):\n'
        prompt += json.dumps(handoff, ensure_ascii=False)
        if native and previous.get('capability_revision') != 'native_coding_v1':
            prompt += ('\nNative coding is now enabled on this existing thread. Confirm the workspace '
                       'once with native pwd, then attach to the existing jobs and continue the current '
                       'milestone. Earlier read-only-tool restrictions are superseded.\n')
        prompt_path = self.journal / 'app_server/initial_prompt.txt'
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding='utf-8')
        # Landlock must resolve these metadata directories to retain their read-only protection.
        for name in ('.git', '.agents', '.codex'):
            if not (self.workspace / name).exists():
                (self.workspace / name).mkdir()
        server = CodexAppServer(codex_home=self.codex_home, workspace=self.workspace,
                                journal=self.journal / 'app_server', on_tool=self.tools,
                                chat=ChatMailbox(self.journal / 'chat'))
        try:
            server.start()
            state['codex_pid'] = server.process.pid
            parameters = native_thread_parameters(self.workspace)
            write_json(self.journal / 'app_server/thread_parameters.json', parameters)
            if thread_id:
                reply = server.request('thread/resume', {**parameters, 'threadId': thread_id,
                                                        'excludeTurns': True})
            else:
                reply = server.request('thread/start', {**parameters, 'dynamicTools': dynamic_tools()})
            thread_id = reply['thread']['id']
            state.update(thread_id=thread_id, status='running', updated_at=time.time())
            write_json(state_path, state)
            while not self.tools.finished:
                # Overrides environments=[] stored by the earlier read-only implementation on resume.
                turn = server.turn(thread_id, prompt, environments=[{
                    'environmentId': 'local', 'cwd': str(self.workspace),
                    'runtimeWorkspaceRoots': [str(self.workspace)]}])
                state.update(last_turn_id=turn['id'], last_turn_status=turn['status'], updated_at=time.time())
                if self.tools.finished:
                    break
                if turn['status'] == 'failed':
                    state['last_error'] = turn.get('error')
                    write_json(state_path, state)
                    time.sleep(30)
                prompt = ('Continue the existing experiment in this same conversation using native tools. '
                          'Inspect active jobs before launching anything again.\n' + MAINLINE_REMINDER + '\n'
                          + self.progress.compact_status_block())
                write_json(state_path, state)
            state.update(status='complete', summary=self.tools.summary, updated_at=time.time())
            write_json(state_path, state)
            return state
        except Exception as exc:
            state.update(status='disconnected', thread_id=thread_id,
                         last_error=f'{type(exc).__name__}: {exc}', updated_at=time.time())
            write_json(state_path, state)
            raise
        finally:
            server.close()

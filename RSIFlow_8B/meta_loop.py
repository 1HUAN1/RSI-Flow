"""A resumable Codex-led Task/Meta evolution loop.

The controller is a tool, not a policy engine.  Codex chooses each operation,
including whether to train, change a harness or artifact, rerun a candidate,
deploy it, append skills, and finish.  Tool receipts contain facts and paths;
no score threshold or component router lives in this module.

This module deliberately keeps the transport small.  The same Codex thread is
resumed after every tool receipt, so the Meta agent can debug the just-run Task
without losing its working context.  The receipts are kept on disk and exposed
by path instead of being copied wholesale into the next prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

MAX_INLINE_RECEIPT_BYTES = 256 * 1024

MAINLINE_REMINDER = """[Experiment mainline reminder]
For each evolution round, keep the following ten-step mainline in order:
1. Run the configured number of parent rollouts for the current Task.
2. Read statistics, 48 representative trajectories, retrieve relevant skills/contradictions, and previous-round context.
3. Choose one component from HARNESS, MODEL, or ARTIFACTS for this attempt.
4. Generate one candidate for that attempt, in a new attempt directory.
5. Re-evaluate the candidate on the same task batch.
6. Use the complete paired scores to accept the candidate or retain the parent.
7. Use paired task differences to append cases, conditional rules and revisions for every attempt. If rejected, keep the
   parent and return to step 3 on this SAME round/batch; do not go to validation or the next round.
8. Save the Task and Meta snapshots.
9. Run independent validation only when due under the configured evaluation schedule; for final_only, only after the final round.
10. Continue to the next round, carrying evidence and artifacts through the configured final stage.
HARNESS: read the pinned three production prompts and execute fault localization from parent
source + metrics/costs + failed AND successful trajectories; then improvement directions using
historical Harnesses/skills; then generate a full independent bundle; only then validate and
check at most 3 times, repairing the same bundle between failed checks. Calling the validator alone is NOT HarnessForge production.
MODEL uses successful-parent SFT. Reuse unchanged parent results across rejected attempts.
Keep decision, reports, candidate, scores and selection under round_N/attempts/attempt_K/;
never overwrite earlier attempts. No fixed retry count is imposed; Meta chooses again after rejection.
After each attempt snapshot numbered before/after components and Meta skills without full code/weight copies.
After acceptance, activate and snapshot, independently validate only when due under the configured schedule, then advance to fresh next-round tasks.
Activate a candidate only for a strict positive gain on complete paired scores.
Only this order is fixed; Meta owns all component, candidate, debugging, and acceptance decisions."""


class ToolExecutor(Protocol):
    def execute(self, request: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CodexReply:
    thread_id: str
    message: str
    events_path: Path


class CodexCli:
    """Keep one on-disk Codex thread across all tool calls and rounds.

    ``CODEX_HOME`` should point at a directory containing a DeepSeek provider
    ``config.toml`` and authentication setup.  It must be persistent; running
    Codex with ``--ephemeral`` would make ``exec resume`` impossible.
    """

    def __init__(self, *, executable: str = "codex", codex_home: Path | None = None,
                 model: str | None = None, timeout_seconds: int | None = None):
        self.executable = executable
        self.codex_home = Path(codex_home) if codex_home is not None else None
        self.model = model
        self.timeout_seconds = timeout_seconds

    def turn(self, prompt: str, *, thread_id: str | None, workspace: Path,
             turn_dir: Path) -> CodexReply:
        turn_dir.mkdir(parents=True, exist_ok=True)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        last_message = turn_dir / "last_message.txt"
        events_path = turn_dir / "events.jsonl"
        if thread_id is None:
            argv = [self.executable, "exec", "--json", "--skip-git-repo-check",
                    "-s", "workspace-write", "-C", str(workspace),
                    "-o", str(last_message)]
        else:
            argv = [self.executable, "exec", "resume", "--json",
                    "--skip-git-repo-check", "-o", str(last_message), thread_id]
        if self.model:
            argv.extend(["-m", self.model])
        argv.append("-")
        env = os.environ.copy()
        if self.codex_home is not None:
            env["CODEX_HOME"] = str(self.codex_home)
        with events_path.open("w", encoding="utf-8") as events:
            result = subprocess.run(argv, input=prompt, text=True, stdout=events,
                                    stderr=subprocess.PIPE, cwd=workspace, env=env,
                                    timeout=self.timeout_seconds, check=False)
        (turn_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Codex exited {result.returncode}; see {events_path} and stderr.txt")
        current_thread = thread_id
        with events_path.open(encoding="utf-8") as events:
            for line in events:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "thread.started":
                    current_thread = event.get("thread_id") or current_thread
        if not current_thread:
            raise RuntimeError(f"Codex returned no thread ID; see {events_path}")
        if not last_message.is_file():
            raise RuntimeError(f"Codex returned no final message; see {events_path}")
        return CodexReply(current_thread, last_message.read_text(encoding="utf-8"), events_path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_codex_command(message: str) -> dict[str, Any] | None:
    """Recover one unambiguous JSON command without choosing its semantics."""
    source = message.strip()
    try:
        direct = json.loads(source)
        if isinstance(direct, dict):
            return direct
    except json.JSONDecodeError:
        pass
    if source.startswith("```") and source.endswith("```"):
        lines = source.splitlines()
        source = "\n".join(lines[1:-1]).strip()
        try:
            fenced = json.loads(source)
            if isinstance(fenced, dict):
                return fenced
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    position = source.find("{")
    while position >= 0:
        try:
            value, _ = decoder.raw_decode(source, position)
            if isinstance(value, dict) and value.get("action") in {"tool", "finish"}:
                if value not in candidates:
                    candidates.append(value)
        except json.JSONDecodeError:
            pass
        position = source.find("{", position + 1)
    return candidates[0] if len(candidates) == 1 else None


def render_receipt_for_codex(receipt: dict[str, Any], path: Path) -> str:
    """Deliver small facts directly; point to paged reads for large evidence."""
    encoded = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= MAX_INLINE_RECEIPT_BYTES:
        return f"Tool receipt at {path}:\n{encoded}\n"
    compact = {key: value for key, value in receipt.items()
               if key not in {"value", "content", "rows", "events", "trajectories"}
               and not isinstance(value, (dict, list))}
    compact["inline_omitted_due_to_size"] = True
    compact["receipt_path"] = str(path)
    return ("Large tool receipt; metadata follows:\n"
            + json.dumps(compact, ensure_ascii=False)[:8192]
            + "\nThe full receipt was saved. Use controller read_text with path and "
              "max_chars <= 65536 and offset_chars=next_offset_chars for later pages; do not infer facts "
              "from omitted content.\n")


class MetaDecisionLoop:
    """Continue Codex after every tool result; never decide Task outcomes itself."""

    def __init__(self, *, workspace: Path, journal: Path, codex: CodexCli,
                 tools: ToolExecutor, status_provider: Callable[[], str] | None = None,
                 finish_guard: Callable[[], tuple[bool, str]] | None = None,
                 progress_recorder: Callable[[Path], Any] | None = None):
        self.workspace = Path(workspace).resolve()
        self.journal = Path(journal).resolve()
        self.codex = codex
        self.tools = tools
        self.status_provider = status_provider
        self.finish_guard = finish_guard
        self.progress_recorder = progress_recorder

    @staticmethod
    def _progress_anchor(status: str) -> dict[str, str | None]:
        """Extract only the stage and milestone that control reminder frequency.

        Evidence paths and completed-item summaries in the external progress block
        change frequently. They deliberately do not participate in this anchor.
        """
        values: dict[str, str | None] = {
            "current_stage": None,
            "next_milestone": None,
        }
        labels = {
            "current stage": "current_stage",
            "current_stage": "current_stage",
            "next required milestone": "next_milestone",
            "next milestone": "next_milestone",
            "next_milestone": "next_milestone",
        }
        for raw_line in status.splitlines():
            label, separator, raw_value = raw_line.strip().partition(":")
            key = labels.get(label.strip().lower())
            if not separator or key is None:
                continue
            value = raw_value.split(" — ", 1)[0].strip()
            values[key] = None if value.lower() in {"", "none", "null"} else value
        return values

    @staticmethod
    def _tool_receipt_failed(receipt: dict[str, Any]) -> bool:
        """Recognize both loop exceptions and controller-returned error receipts."""
        status = str(receipt.get("status", "")).strip().lower()
        return receipt.get("ok") is False or status in {
            "error", "failed", "failure", "tool_error",
        }

    def _prompt_for_turn(self, prompt: str, state: dict[str, Any], *,
                         force_mainline: bool = False) -> str:
        """Inject the full mainline only at stage changes, errors, or recovery."""
        status = self.status_provider().strip() if self.status_provider is not None else ""
        anchor = self._progress_anchor(status)
        if state.get("last_mainline_anchor") == anchor and not force_mainline:
            return prompt
        additions = [MAINLINE_REMINDER]
        if status:
            additions.append("[Current experiment progress]\n" + status)
        state["last_mainline_anchor"] = anchor
        _write_json(self.state_path, state)
        return prompt.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"

    @property
    def state_path(self) -> Path:
        return self.journal / "state.json"

    def _state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"thread_id": None, "turn": 0, "next_prompt": None,
                "status": "ready", "summary": None,
                "last_mainline_anchor": None}

    def run(self, initial_prompt: str) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.journal.mkdir(parents=True, exist_ok=True)
        recovering = self.state_path.is_file()
        state = self._state()
        if state["status"] == "complete":
            return state
        prompt = state["next_prompt"] or initial_prompt
        force_mainline = recovering
        while True:
            number = state["turn"]
            turn_dir = self.journal / "turns" / f"{number:04d}"
            # A crash after a tool starts is ambiguous.  On resume, give Codex
            # the evidence path and ask it how to proceed instead of silently
            # rerunning a potentially mutating operation.
            if state["status"] == "tool_inflight":
                prompt = ("The previous tool call may have completed before the process stopped. "
                          f"Inspect {turn_dir} and the Task output files; choose the next action. "
                          "Do not assume the tool failed or automatically repeat it.")
                state.update({"status": "ready", "next_prompt": prompt,
                              "turn": number + 1})
                _write_json(self.state_path, state)
                continue
            codex_prompt = self._prompt_for_turn(
                prompt, state, force_mainline=force_mainline,
            )
            force_mainline = False
            reply = self.codex.turn(codex_prompt, thread_id=state["thread_id"],
                                    workspace=self.workspace, turn_dir=turn_dir)
            state["thread_id"] = reply.thread_id
            (turn_dir / "response.txt").write_text(reply.message, encoding="utf-8")
            command = parse_codex_command(reply.message)
            if command is None:
                prompt = ("Your last response was not a JSON command. Return one JSON object: "
                          '{"action":"tool","request":{"operation":"..."}} or '
                          '{"action":"finish","summary":"..."}. '
                          f"Your response is recorded at {turn_dir / 'response.txt'}.")
                state.update({"status": "ready", "next_prompt": prompt,
                              "turn": number + 1})
                _write_json(self.state_path, state)
                continue
            _write_json(turn_dir / "command.json", command)
            if command.get("action") == "finish":
                if self.finish_guard is not None:
                    allowed, reason = self.finish_guard()
                    if not allowed:
                        prompt = ("Completion is not yet allowed. "
                                  + (reason.strip() or "Required experiment stages remain incomplete.")
                                  + " Continue from the current milestone and return the next JSON command.")
                        state.update({"status": "ready", "next_prompt": prompt,
                                      "turn": number + 1})
                        _write_json(self.state_path, state)
                        continue
                state.update({"status": "complete", "summary": command.get("summary"),
                              "next_prompt": None, "turn": number + 1})
                _write_json(self.state_path, state)
                return state
            request = command.get("request")
            if command.get("action") != "tool" or not isinstance(request, dict):
                prompt = ("Return a tool command with a request object, or finish. "
                          "The controller cannot infer a missing operation.")
                state.update({"status": "ready", "next_prompt": prompt,
                              "turn": number + 1})
                _write_json(self.state_path, state)
                continue
            state.update({"status": "tool_inflight", "next_prompt": None})
            _write_json(self.state_path, state)
            try:
                receipt = self.tools.execute(request)
            except Exception as exc:
                # Tool failure is evidence for Meta to debug, not a controller
                # verdict to terminate the whole experiment.
                receipt = {"ok": False, "error_type": type(exc).__name__,
                           "error": str(exc), "request": request}
            _write_json(turn_dir / "receipt.json", receipt)
            if self.progress_recorder is not None:
                self.progress_recorder(turn_dir)
            prompt = (render_receipt_for_codex(receipt, turn_dir / "receipt.json")
                      + "Read referenced Task evidence as needed, then choose the next action. "
                      "The controller has not decided whether to keep or reject any candidate. "
                      "Use external controller JSON tools only; native shell tools fail here. "
                      "Return one JSON command only.")
            state.update({"status": "ready", "next_prompt": prompt,
                          "turn": number + 1})
            _write_json(self.state_path, state)
            force_mainline = self._tool_receipt_failed(receipt)


META_INSTRUCTIONS = """You are the Meta agent, not merely an advisor. Use the controller tools
to run the Task, inspect trajectories and scores, choose exactly one component per round,
create one candidate, run the same tasks again, and decide whether the measured gain
justifies deployment. Deploy only when complete paired scores are comparable and the
candidate's success rate is strictly higher; otherwise retain the parent. Then append
component-specific and general skills and take Task
and Meta snapshots. Repeat for the configured number of rounds, using each prior
snapshot and skill library. The controller only executes requests and returns facts;
For each attempted component, append bounded structured experience, whether the
change succeeded, failed or could not run. Give component entries unique IDs
skill.HARNESS.<id>, skill.MODEL.<id>, or skill.ARTIFACTS.<id>; give general
principles unique IDs principle.<id> and cite the supporting component skill.
Record the actual change mechanism, measured outcome, applicability boundary,
and evidence references: original rollout path, task ID and content hash.
Never copy a full trajectory into the skill ledger. The next round inherits
the ledger and reads relevant entries or original trajectory files on demand.
you own selection, modification, debugging, acceptance, and continuation decisions.
If a tool fails, inspect its receipt, repair the cause if possible, and continue.
Use only the external controller protocol for reading, writing, patching and
running commands. Native Codex shell/apply_patch tools currently fail in this
server's sandbox; request read_text, write_text, apply_patch or run_command
from the controller instead.
Return only JSON commands: {"action":"tool","request":{"operation":"..."}}
or {"action":"finish","summary":"..."}. Never declare completion before the
requested rounds and independent evaluations are actually finished. Refer to large
rollout files by path and read details on demand instead of copying them into prompts.
"""

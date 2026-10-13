"""Watch and send messages to the same running Meta. Never launches Codex or GPU jobs."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from meta_chat import ChatMailbox
from monitor_meta import EventTail, EventView, read_json, show_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--history', type=int, default=12)
    parser.add_argument('--send', help='Send one message and return its ID without starting another Meta agent')
    args = parser.parse_args(argv)
    config = read_json(Path(__file__).resolve().parent/'configs/train_180_a0_v1.json')
    run = (args.run_dir or Path(config['output_root'])/'runs'/config['run_name']).resolve()
    session = run/'meta_session'
    if not (session/'state.json').is_file():
        parser.error(f'Existing experiment session not found: {session}')
    mailbox = ChatMailbox(session/'chat')
    if args.send is not None:
        try:
            record = mailbox.enqueue(args.send)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(record, ensure_ascii=False))
        return 0

    from contextlib import nullcontext
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError:
        PromptSession = None
    prompt = PromptSession() if PromptSession and sys.stdin.isatty() else None
    output_context = patch_stdout(raw=True) if prompt else nullcontext()
    stop = threading.Event()
    with output_context:
        view = EventView()
        print(f'Meta chat | {run}\nEnter a message to send; /status shows status; /quit closes the interface.')
        print('Uses the existing Meta agent. Closing this interface leaves the experiment running. Messages are read at Codex processing points and do not cancel GPU jobs.')
        show_status(session, view)
        tail = EventTail(session/'app_server/events.jsonl')
        for event in tail.history(max(args.history, 0)):
            view.handle(event, replay=True)
        seen = {record['id']: record['status'] for record in mailbox.records()}

        def follow():
            while not stop.wait(.5):
                for event in tail.poll():
                    view.handle(event)
                for record in mailbox.records():
                    if seen.get(record['id']) != record['status']:
                        view.line(mailbox.status_event(record))
                        seen[record['id']] = record['status']

        follower = threading.Thread(target=follow, daemon=True)
        follower.start()
        try:
            while True:
                try:
                    line = (prompt.prompt('You > ') if prompt else input('You > ')).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if line in {'/quit', '/exit'}:
                    break
                if line == '/status':
                    show_status(session, view)
                    for record in mailbox.records()[-5:]:
                        view.line(mailbox.status_event(record))
                elif line:
                    mailbox.enqueue(line)
        finally:
            stop.set()
            follower.join(timeout=2)
        view.line('Chat closed; Meta and the experiment continue running.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

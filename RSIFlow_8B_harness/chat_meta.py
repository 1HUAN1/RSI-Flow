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
    parser.add_argument('--send', help='只发送一条消息并返回消息 ID，不启动另一个 Meta')
    args = parser.parse_args(argv)
    config = read_json(Path(__file__).resolve().parent/'configs/train_180_a0_v1.json')
    run = (args.run_dir or Path(config['output_root'])/'runs'/config['run_name']).resolve()
    session = run/'meta_session'
    if not (session/'state.json').is_file():
        parser.error(f'没有找到现有实验会话：{session}')
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
        print(f'Meta 双向对话 | {run}\n输入消息后回车发送；/status 查看状态；/quit 退出界面。')
        print('不会另起 Meta；退出界面不停止实验。消息在 Codex 的处理点被读取，不会取消已启动的 GPU 任务。')
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
                    line = (prompt.prompt('你 > ') if prompt else input('你 > ')).strip()
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
        view.line('已关闭聊天界面；Meta 和实验继续运行。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

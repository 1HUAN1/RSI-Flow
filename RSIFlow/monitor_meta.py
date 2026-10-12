"""Read-only terminal view of an existing Meta conversation; never starts an agent."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def process_info(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except (OSError, ValueError, TypeError, IndexError):
        return None, None


def brief(value, limit=300):
    text = str(value).replace('\x1b', '').replace('\r', '')
    return text if len(text) <= limit else text[:limit] + '…'


def tool_name(item):
    args = item.get('arguments') or {}
    request = args.get('request', args) if isinstance(args, dict) else {}
    name = item.get('tool', 'tool')
    if name == 'experiment':
        name += '.' + str(request.get('operation', '?'))
    detail = {k: request[k] for k in ('round_number', 'job_id', 'seconds', 'path', 'output_dir')
              if k in request}
    return name + (' ' + brief(json.dumps(detail, ensure_ascii=False), 500) if detail else '')


def receipt_summary(item):
    """Use the factual receipt, not transport success=true, as the displayed status."""
    for content in item.get('contentItems') or []:
        text = content.get('text', '')
        marker = text.find('{')
        if marker < 0:
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[marker:])
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        details = ['status=' + str(value.get('status', '?'))]
        for key in ('job_id', 'elapsed_seconds', 'error', 'returncode', 'appended_count'):
            if value.get(key) is not None:
                details.append(f'{key}={brief(value[key])}')
        if isinstance(value.get('result'), dict):
            details.append('任务结果=' + str(value['result'].get('status', '?')))
        if text.startswith('Tool receipt at '):
            details.append('\n    回执：' + text.split('\n', 1)[0][len('Tool receipt at '):].rstrip(':'))
        return ' | '.join(details)
    return f"transport={item.get('status')} success={item.get('success')}"


class EventView:
    def __init__(self, output=None):
        self.output = output if output is not None else sys.stdout
        self.streamed = set()
        self.stream = None

    def line(self, text, timestamp=None):
        if self.stream is not None:
            print(file=self.output)
            self.stream = None
        clock = datetime.fromtimestamp(timestamp or time.time()).strftime('%H:%M:%S')
        print(f'[{clock}] {text}', file=self.output, flush=True)

    def handle(self, event, *, replay=False):
        method = event.get('method')
        params = event.get('params') or {}
        item = params.get('item') or {}
        stamp = event.get('emittedAtMs')
        stamp = stamp / 1000 if isinstance(stamp, (float, int)) else None
        if method == 'item/agentMessage/delta' and not replay:
            identifier = params.get('itemId')
            if self.stream != identifier:
                self.line('Meta：', stamp)
                self.stream = identifier
            self.streamed.add(identifier)
            print(params.get('delta', ''), end='', file=self.output, flush=True)
        elif method == 'item/completed' and item.get('type') == 'agentMessage':
            if item.get('id') not in self.streamed:
                self.line('Meta：\n' + item.get('text', ''), stamp)
            elif self.stream is not None:
                print(file=self.output, flush=True)
                self.stream = None
            self.streamed.discard(item.get('id'))
        elif method == 'item/started' and item.get('type') == 'dynamicToolCall':
            self.line('调用 → ' + tool_name(item), stamp)
        elif method == 'item/completed' and item.get('type') == 'dynamicToolCall':
            self.line('返回 ← ' + item.get('tool', '?') + ' | ' + receipt_summary(item), stamp)
        elif method in {'item/started', 'item/completed'} and item.get('type') == 'commandExecution':
            self.line('原生命令 | ' + brief(item.get('command', ''), 600)
                      + ' | ' + str(item.get('status', '?'))
                      + ' | exit=' + str(item.get('exitCode')), stamp)
            if method == 'item/completed' and item.get('aggregatedOutput'):
                self.line(brief(item['aggregatedOutput'], 2000), stamp)
        elif method in {'item/started', 'item/completed'} and item.get('type') == 'fileChange':
            paths = [change.get('path', '?') for change in item.get('changes', [])]
            self.line('原生文件修改 | ' + str(item.get('status', '?'))
                      + ' | ' + brief(', '.join(paths), 1200), stamp)
        elif method == 'turn/started':
            self.line('会话开始工作 | turn=' + str(params.get('turn', {}).get('id')), stamp)
        elif method == 'turn/completed':
            self.line('本次 turn 结束（不等于实验结束） | ' + str(params.get('turn', {}).get('status')), stamp)
        elif method in {'error', 'warning', 'configWarning'}:
            self.line('提示/错误：' + brief(json.dumps(params, ensure_ascii=False), 800), stamp)


class EventTail:
    """Keep partial JSON lines until complete; reopen on replacement/truncation."""
    def __init__(self, path):
        self.path = Path(path)
        self.position = 0
        self.identity = None
        self.pending = b''

    def history(self, count):
        try:
            with self.path.open('rb') as stream:
                stat = self.path.stat()
                self.identity = (stat.st_dev, stat.st_ino)
                stream.seek(max(0, stat.st_size - 4 * 1024 * 1024))
                if stream.tell():
                    stream.readline()
                data = stream.read()
                self.position = stream.tell()
        except FileNotFoundError:
            return []
        lines = data.split(b'\n')
        self.pending = lines.pop()
        records = deque(maxlen=count)
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item = event.get('params', {}).get('item', {})
            if (event.get('method') == 'item/completed' and item.get('type') in {
                    'agentMessage', 'dynamicToolCall', 'commandExecution', 'fileChange'}
                    or event.get('method') in {'error', 'warning', 'turn/completed'}):
                records.append(event)
        return list(records)

    def poll(self):
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.position:
                self.position, self.pending = 0, b''
            self.identity = identity
            with self.path.open('rb') as stream:
                stream.seek(self.position)
                data = stream.read(1024 * 1024)
                self.position = stream.tell()
        except FileNotFoundError:
            return []
        lines = (self.pending + data).split(b'\n')
        self.pending = lines.pop()
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        return events


def show_status(session, view):
    state = read_json(session/'state.json')
    progress = read_json(session/'workflow_state.json')
    pid = state.get('codex_pid')
    process_state, _ = process_info(pid)
    alive = process_state is not None and process_state != 'Z'
    view.line(f"状态 | Codex PID={pid} {'存活' if alive else '不在运行'}({process_state or '-'})"
              f" | launcher={state.get('launcher_pid')} | 会话记录={state.get('status', '尚无记录')}"
              f" | 阶段={progress.get('current_stage', '?')} | 下一产物={progress.get('next_milestone', '?')}")
    print('    thread=' + str(state.get('thread_id', '?')), file=view.output, flush=True)
    for path in sorted((session/'jobs').glob('*/job.json')):
        job = read_json(path)
        result_path = Path(job.get('result_path', '/nonexistent'))
        actual_state, identity = process_info(job.get('pid'))
        running = identity is not None and identity == job.get('process_start') and actual_state != 'Z'
        if result_path.is_file():
            result = read_json(result_path)
            status = '已有回执，任务结果=' + str(result.get('status', '?'))
        else:
            status = '运行中，尚无最终回执' if running else '进程已退出，未发现最终回执'
        view.line(f"后台任务 | {job.get('request', {}).get('operation')} | pid={job.get('pid')}"
                  f" | {status} | job={job.get('job_id')}")
    log = session/'app_server/events.jsonl'
    if log.exists():
        view.line(f'事件日志距最近写入 {int(max(0, time.time()-log.stat().st_mtime))} 秒；存活不代表任务成功。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--history', type=int, default=12, help='显示最近 N 条可见消息/回执；0 为只看新增')
    parser.add_argument('--once', action='store_true', help='打印一次状态和历史后退出')
    parser.add_argument('--heartbeat', type=float, default=30, help='状态刷新间隔（秒）')
    args = parser.parse_args(argv)
    if args.history < 0 or args.heartbeat <= 0:
        parser.error('history 必须 >= 0，heartbeat 必须 > 0')
    config = read_json(Path(__file__).resolve().parent/'configs/train_180_a0_v1.json')
    run = args.run_dir or Path(config['output_root'])/'runs'/config['run_name']
    session = run.resolve()/'meta_session'
    view = EventView()
    print(f'RSIFlow Meta 对话监控（只读）\n运行目录：{run}\nCtrl+C 仅退出监控，不会停止 Meta 或 GPU 任务。', flush=True)
    print('显示可见回复与工具事件，不显示内部推理；不会调用 API、重启实验或写入状态。', flush=True)
    show_status(session, view)
    tail = EventTail(session/'app_server/events.jsonl')
    for event in tail.history(args.history):
        view.handle(event, replay=True)
    if args.once:
        return 0
    view.line('开始跟随新增对话……')
    next_status = time.monotonic() + args.heartbeat
    try:
        while True:
            for event in tail.poll():
                view.handle(event)
            if time.monotonic() >= next_status:
                show_status(session, view)
                next_status = time.monotonic() + args.heartbeat
            time.sleep(.5)
    except KeyboardInterrupt:
        view.line('已退出监控；实验未停止。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

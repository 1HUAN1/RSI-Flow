"""Durable user messages for the existing Meta process, not a second agent."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class ChatMailbox:
    def __init__(self, directory):
        self.directory = Path(directory)

    def save(self, record):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (record['id'] + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
        return record

    def enqueue(self, text):
        text = text.strip()
        if not text:
            raise ValueError('Message must not be empty')
        return self.save({'id': f'{time.time_ns()}_{uuid.uuid4().hex[:8]}', 'text': text,
                          'status': 'queued', 'created_at': time.time()})

    def records(self):
        records = []
        for path in sorted(self.directory.glob('*.json')):
            try:
                records.append(json.loads(path.read_text(encoding='utf-8')))
            except (OSError, ValueError):
                continue
        return records

    def recover(self):
        # An interrupted send may already have reached Codex. Never silently repeat a user command.
        for record in self.records():
            if record.get('status') == 'sending':
                self.save({**record, 'status': 'delivery_unknown',
                           'detail': 'Connection lost before acknowledgement; inspect the conversation before resending.'})

    def pending(self, turn_id):
        return [record for record in self.records() if record.get('status') == 'queued'
                and record.get('last_attempt_turn') != turn_id]

    def status_event(self, record):
        labels = {'queued': 'Queued', 'sending': 'Sending', 'accepted': 'Accepted by Codex (execution not confirmed)',
                  'delivery_error': 'Delivery failed', 'delivery_unknown': 'Delivery unknown'}
        return f"Message {record['id']} | {labels.get(record['status'], record['status'])}" + (
            ' | ' + str(record['detail']) if record.get('detail') else '')

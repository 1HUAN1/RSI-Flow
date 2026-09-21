"""Standalone sandbox evidence pager. Never imports or executes evidence code."""
import argparse
import hashlib
import json
from pathlib import Path


def load(reference):
    path = Path(reference['file'])
    root = Path('meta_input').resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Evidence reference must stay inside meta_input')
    data = path.read_bytes()
    if reference.get('sha256') and hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('Evidence hash mismatch')
    return json.loads(data)


def expand(value):
    if isinstance(value, dict):
        ref = value.get('content_reference')
        if isinstance(ref, dict) and ref.get('encoding') == 'canonical_json':
            return expand(load(ref))
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file')
    parser.add_argument('--path', default='[]', help='JSON array of dictionary keys/list indices')
    parser.add_argument('--expand', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--max-chars', type=int, default=12000)
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.max_chars <= 64000:
        parser.error('offset >= 0 and 1 <= max-chars <= 64000 required')
    value = load({'file': args.file})
    for key in json.loads(args.path):
        if isinstance(value, dict) and 'content_reference' in value:
            value = load(value['content_reference'])
        value = value[key]
    if args.expand:
        value = expand(value)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    end = min(len(text), args.offset + args.max_chars)
    print(json.dumps({'total_chars': len(text), 'offset': args.offset, 'end': end,
                      'complete': args.offset == 0 and end == len(text),
                      'next_offset': end if end < len(text) else None,
                      'sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'text': text[args.offset:end]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

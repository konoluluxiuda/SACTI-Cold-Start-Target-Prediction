#!/usr/bin/env python3
"""Check the published code/configuration/result/archive file hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        if path.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
            raise ValueError('Release file changed: ' + item['path'])
    print('PASS: %d release file checksums.' % len(manifest['files']))


if __name__ == '__main__':
    main()

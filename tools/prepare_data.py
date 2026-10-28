#!/usr/bin/env python3
"""Install and verify the processed paper inputs, without downloading raw pages."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('Unsafe archive path: ' + name)
    return Path(*path.parts)


def relocate(value, old_root, new_root):
    if isinstance(value, str):
        if value == old_root or value.startswith(old_root + '/'):
            return new_root + value[len(old_root):]
        return value
    if isinstance(value, list):
        return [relocate(x, old_root, new_root) for x in value]
    if isinstance(value, dict):
        return {relocate(k, old_root, new_root): relocate(v, old_root, new_root)
                for k, v in value.items()}
    return value


def install(root, manifest):
    root = root.resolve()
    archive = root / safe_relative(manifest['archive'])
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('Processed-input archive checksum mismatch.')
    records = {x['path']: x for x in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate manifest path.')
    for name in records:
        target = root / safe_relative(name)
        if target.exists() or target.is_symlink():
            raise FileExistsError('Input already exists; use --verify: ' + name)
        if not target.resolve().is_relative_to(root):
            raise ValueError('Destination escapes the repository: ' + name)
    with tempfile.TemporaryDirectory(prefix='.sacti-inputs-', dir=root) as directory:
        staging = Path(directory)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                relative = safe_relative(member.name)
                if member.name not in records or member.name in seen or not member.isfile():
                    raise ValueError('Unexpected archive member: ' + member.name)
                record = records[member.name]
                if member.size != record['bytes']:
                    raise ValueError('Input size mismatch: ' + member.name)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                with target.open('wb') as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(chunk)
                if digest(target) != record['sha256']:
                    raise ValueError('Input checksum mismatch: ' + member.name)
                seen.add(member.name)
        if seen != set(records):
            raise ValueError('Archive is missing declared inputs.')

        pending = set(manifest['relocate_json'])
        finished, visiting = set(), set()

        def update(name):
            if name not in pending or name in finished:
                return
            if name in visiting:
                raise ValueError('Cyclic provenance metadata: ' + name)
            visiting.add(name)
            path = staging / name
            data = relocate(json.loads(path.read_text()), manifest['original_root'], str(root))
            sources = data.get('source_hashes', {}) if isinstance(data, dict) else {}
            for filename, original_hash in sources.items():
                dep = Path(filename)
                if not dep.is_absolute() or not dep.is_relative_to(root):
                    raise ValueError('Unresolved provenance path: ' + filename)
                relative = dep.relative_to(root).as_posix()
                update(relative)
                actual = staging / relative if relative in records else root / relative
                if relative not in pending and digest(actual) != original_hash:
                    raise ValueError('Frozen source changed: ' + relative)
                data['source_hashes'][filename] = digest(actual)
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
            visiting.remove(name)
            finished.add(name)

        for name in sorted(pending):
            update(name)
        installed = []
        for name in sorted(records):
            path = staging / name
            installed.append(dict(path=name, original_sha256=records[name]['sha256'],
                                  installed_sha256=digest(path),
                                  metadata_relocated=name in pending))
        for row in installed:
            destination = root / row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / row['path'], destination)
        receipt = dict(archive_sha256=manifest['archive_sha256'], root=str(root),
                       note='Only provenance paths and hashes of relocated metadata changed; numerical inputs are unchanged.',
                       files=installed)
        (root / 'data/installed_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def verify(root, manifest):
    root = root.resolve()
    receipt = json.loads((root / 'data/installed_manifest.json').read_text())
    if receipt['root'] != str(root) or receipt['archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Installation identity changed; unpack in a clean clone.')
    if digest(root / manifest['archive']) != manifest['archive_sha256']:
        raise ValueError('Archive checksum mismatch.')
    if {x['path'] for x in receipt['files']} != {x['path'] for x in manifest['files']}:
        raise ValueError('Installed input list differs from the release.')
    for row in receipt['files']:
        if digest(root / row['path']) != row['installed_sha256']:
            raise ValueError('Installed input changed: ' + row['path'])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify an existing installation.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    receipt = verify(ROOT, manifest) if args.verify else install(ROOT, manifest)
    print('PASS: %d processed input and provenance files; four datasets, two fixed split partitions.'
          % len(receipt['files']))


if __name__ == '__main__':
    main()

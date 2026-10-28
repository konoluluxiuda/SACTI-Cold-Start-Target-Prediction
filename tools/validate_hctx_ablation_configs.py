#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / 'configs' / 'hctx_ablation_manifest.json'


def repository_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_config(path):
    values = {}
    for line_number, raw_line in enumerate(
            Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(('#', ';')):
            continue
        if '=' not in line:
            raise ValueError('%s:%d is not key=value.' % (path, line_number))
        key, value = line.split('=', 1)
        key = key.strip()
        if key in values:
            raise ValueError('%s has duplicate key %s.' % (path, key))
        values[key] = value.strip()
    return values


def config_differences(left, right):
    differences = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences[key] = (left.get(key), right.get(key))
    return differences


def validate_config_pair(no_context, hctx, allowed_differences):
    differences = config_differences(no_context, hctx)
    unexpected = sorted(set(differences) - set(allowed_differences))
    missing = sorted(set(allowed_differences) - set(differences))
    if unexpected:
        raise ValueError('Unexpected config differences: %s' % unexpected)
    if missing:
        raise ValueError('Required config differences are absent: %s' % missing)

    expected_no_context = {
        'context.interaction': 'False',
        'context.herb_protein': 'False',
    }
    expected_hctx = {
        'context.interaction': 'True',
        'context.herb_protein': 'True',
    }
    shared_expected = {
        'experiment.protocol': 'strict',
        'split.strategy': 'pair_stratified',
        'split.reuse': 'True',
        'evaluation.setup': '-cv 5',
        'evaluation.outer.test': 'True',
        'early.stopping': 'True',
        'pair.decoder': 'dot',
        'counterfactual.context': 'False',
        'attention.max.nodes': '0',
        'random.seed': '2026',
        'split.seed': '2026',
        'validation.seed': '102026',
    }
    for key, expected in expected_no_context.items():
        if no_context.get(key) != expected:
            raise ValueError('NoContext %s must be %s.' % (key, expected))
    for key, expected in expected_hctx.items():
        if hctx.get(key) != expected:
            raise ValueError('Hctx-P %s must be %s.' % (key, expected))
    for key, expected in shared_expected.items():
        if no_context.get(key) != expected or hctx.get(key) != expected:
            raise ValueError('Both configs require %s=%s.' % (key, expected))
    return differences


def load_reference_rows(path):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def validate_manifest(manifest_path):
    manifest_path = repository_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported Hctx ablation manifest schema.')
    allowed = manifest.get('allowed_config_differences') or []
    if set(allowed) != {
            'model.variant', 'context.interaction', 'context.herb_protein'}:
        raise ValueError('The allowed config differences are not frozen.')
    datasets = manifest.get('datasets') or []
    if len(datasets) != 4:
        raise ValueError('The Hctx ablation manifest must contain four datasets.')

    reference_path = repository_path(manifest['reference_results'])
    reference_rows = load_reference_rows(reference_path)
    validated = []
    for dataset in datasets:
        no_context_path = repository_path(dataset['no_context_config'])
        hctx_path = repository_path(dataset['hctx_config'])
        for path, expected_hash in (
                (no_context_path, dataset['no_context_sha256']),
                (hctx_path, dataset['hctx_sha256'])):
            if not path.is_file():
                raise FileNotFoundError('Missing config: %s' % path)
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    'Config hash mismatch for %s: %s != %s' % (
                        path, actual_hash, expected_hash
                    )
                )
        differences = validate_config_pair(
            parse_config(no_context_path), parse_config(hctx_path), allowed
        )
        matching = [
            row for row in reference_rows
            if row.get('dataset') == dataset['name']
            and row.get('variant') == 'Hctx-P'
        ]
        if len(matching) != 1:
            raise ValueError(
                'Expected one frozen Hctx-P reference row for %s.'
                % dataset['name']
            )
        reference = matching[0]
        if reference.get('status') != 'OK':
            raise ValueError('Frozen Hctx-P reference is not successful.')
        if reference.get('config_sha256') != dataset['hctx_sha256']:
            raise ValueError(
                'Frozen Hctx-P result hash mismatch for %s.' % dataset['name']
            )
        validated.append({
            'dataset': dataset['name'],
            'no_context_config': str(no_context_path),
            'hctx_config': str(hctx_path),
            'differences': differences,
        })
    return manifest, validated


def main():
    parser = argparse.ArgumentParser(
        description='Validate the frozen four-dataset Hctx-P ablation configs.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    manifest, validated = validate_manifest(args.manifest)
    print('Protocol: %s' % manifest['protocol'])
    for item in validated:
        print('%s: OK (%s)' % (
            item['dataset'], ', '.join(sorted(item['differences']))
        ))
    print('All four Hctx-P ablation config pairs are frozen and equivalent.')


if __name__ == '__main__':
    main()

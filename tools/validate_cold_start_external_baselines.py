#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tools.validate_hctx_ablation_configs import parse_config, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / 'configs' / 'cold_start_external_baselines_manifest.json'
)


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def validate_manifest(manifest_path=DEFAULT_MANIFEST, require_split_files=True):
    manifest_path = resolve_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported external-baseline manifest schema.')
    if manifest.get('protocol') != (
            'strict_compound_cold_start_external_baselines_seed52026_v1'):
        raise ValueError('Unexpected external-baseline protocol.')
    if manifest.get('dataset_specific_tuning') is not False:
        raise ValueError('Dataset-specific tuning must remain disabled.')

    datasets = {item['key']: item for item in manifest.get('datasets', [])}
    methods = {item['key']: item for item in manifest.get('methods', [])}
    jobs = manifest.get('jobs', [])
    if len(datasets) != 4 or len(methods) != 4 or len(jobs) != 16:
        raise ValueError('Expected four datasets, four methods, and 16 jobs.')
    expected_pairs = {
        (dataset, method) for dataset in datasets for method in methods
    }
    actual_pairs = {(job['dataset'], job['method']) for job in jobs}
    if actual_pairs != expected_pairs or len(actual_pairs) != len(jobs):
        raise ValueError('Jobs must contain each dataset-method pair exactly once.')

    required = {
        'evaluation.setup': '-cv 5',
        'evaluation.outer.test': 'True',
        'experiment.protocol': 'strict',
        'random.seed': '52026',
        'split.strategy': 'compound_cold_start',
        'split.seed': '52026',
        'split.reuse': 'True',
        'early.stopping': 'True',
        'validation.seed': '152026',
        'validation.metric': 'AUPR',
        'pair.decoder': 'dot',
    }
    validated = []
    for job in jobs:
        dataset = datasets[job['dataset']]
        method = methods[job['method']]
        config_path = resolve_path(job['config'])
        if not config_path.is_file():
            raise FileNotFoundError('Missing config: %s' % config_path)
        actual_hash = sha256_file(config_path)
        if actual_hash != job['sha256']:
            raise ValueError(
                'Config hash mismatch for %s: %s != %s'
                % (config_path, actual_hash, job['sha256'])
            )
        config = parse_config(config_path)
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(
                    '%s requires %s=%s, got %s.'
                    % (config_path, key, expected, config.get(key))
                )
        if config.get('model.name') != method['model_name']:
            raise ValueError('%s has the wrong model.name.' % config_path)
        expected_profile = method.get('encoder_profile')
        if expected_profile and config.get('encoder.profile') != expected_profile:
            raise ValueError('%s has the wrong encoder.profile.' % config_path)
        if method['key'] == 'dual_hgnn':
            disabled_hdcti_features = {
                'context.interaction': 'False',
                'counterfactual.context': 'False',
                'context.mask.training': 'False',
                'support.router': 'False',
                'inductive.context': 'False',
                'hyperedge.attention': 'False',
                'global.token.attention': 'False',
                'attention.max.nodes': '0',
            }
            for key, expected in disabled_hdcti_features.items():
                if config.get(key) != expected:
                    raise ValueError(
                        '%s requires %s=%s.' % (config_path, key, expected)
                    )
        if method['key'] == 'hgt' and config.get('hgt.sampling.seed') != '52026':
            raise ValueError('%s has the wrong HGT sampling seed.' % config_path)

        split_dir = resolve_path(dataset['split_dir'])
        if resolve_path(config['split.dir']) != split_dir:
            raise ValueError('%s uses the wrong split directory.' % config_path)
        if require_split_files:
            split_manifest = split_dir / 'manifest.json'
            if not split_manifest.is_file():
                raise FileNotFoundError(
                    'Missing split manifest: %s' % split_manifest
                )
            split = json.loads(split_manifest.read_text(encoding='utf-8'))
            if split.get('split_strategy') != 'compound_cold_start':
                raise ValueError(
                    '%s is not a compound cold-start split.' % split_manifest
                )
            if int(split.get('seed', -1)) != 52026:
                raise ValueError('%s has the wrong split seed.' % split_manifest)
            if int(split.get('folds', -1)) != 5:
                raise ValueError('%s does not contain five folds.' % split_manifest)
            guarantees = split.get('strict_guarantees', {})
            if not guarantees.get('compound_disjoint_train_test'):
                raise ValueError(
                    '%s does not guarantee compound-disjoint folds.'
                    % split_manifest
                )
            if not guarantees.get(
                    'training_graph_must_use_fold_training_positives'):
                raise ValueError(
                    '%s does not constrain the fold training graph.'
                    % split_manifest
                )
        validated.append({
            'dataset': dataset['name'],
            'method': method['name'],
            'config': job['config'],
            'sha256': actual_hash,
        })
    return manifest, validated


def main():
    parser = argparse.ArgumentParser(
        description='Validate frozen compound-cold-start external baselines.'
    )
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        '--print-jobs', action='store_true',
        help='Print validated jobs as dataset|method|slug|config rows.',
    )
    args = parser.parse_args()
    manifest, jobs = validate_manifest(args.manifest)
    if args.print_jobs:
        method_keys = {item['name']: item['key'] for item in manifest['methods']}
        dataset_keys = {item['name']: item['key'] for item in manifest['datasets']}
        for job in jobs:
            slug = '%s_%s' % (
                method_keys[job['method']], dataset_keys[job['dataset']]
            )
            print('|'.join((job['dataset'], job['method'], slug, job['config'])))
        return
    print('Protocol: %s' % manifest['protocol'])
    for job in jobs:
        print('%-20s %-15s OK  %s' % (
            job['dataset'], job['method'], job['config']
        ))
    print('All 16 frozen cold-start external-baseline jobs are valid.')


if __name__ == '__main__':
    main()

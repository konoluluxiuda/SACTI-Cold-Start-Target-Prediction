#!/usr/bin/env python3
"""Run the frozen seed-52026 SACTI component ablation."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_io import parse_config, write_resolved_config, parse_five_fold_log  # noqa: E402


DEFAULT_MANIFEST = ROOT / 'configs' / 'sacti_component_ablation_seed52026.json'
METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
SUMMARY_MARKER = 'The result of 5-fold cross validation:'
MEAN_PATTERN = re.compile(r'^([0-9.]+)')


def repository_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_config(path, manifest, role_settings):
    _, values = parse_config(path)
    required = {
        'model.name': 'SACTI',
        'experiment.protocol': 'strict',
        'split.strategy': manifest['split_strategy'],
        'split.seed': str(manifest['split_seed']),
        'random.seed': str(manifest['training_seed']),
        'validation.seed': str(manifest['validation_seed']),
        'evaluation.fold.limit': str(manifest['fold_count']),
        'evaluation.outer.test': 'True',
        'early.stopping': 'True',
        'validation.metric': 'AUPR',
    }
    required.update({
        str(key): str(value)
        for key, value in role_settings['overrides'].items()
    })
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError('%s must be %s in %s.' % (key, expected, path))
    return values


def build_jobs(manifest, run_dir):
    jobs = []
    for dataset in manifest['datasets']:
        source = repository_path(dataset['source_config'])
        if sha256_file(source) != dataset['source_config_sha256']:
            raise ValueError(
                'Frozen source config hash mismatch for %s.' % dataset['slug']
            )
        for role, settings in manifest['roles'].items():
            overrides = dict(settings['overrides'])
            overrides.update({
                'evaluation.fold.limit': str(manifest['fold_count']),
                'evaluation.outer.test': 'True',
            })
            config_path = run_dir / 'configs' / (
                '%s_%s.conf' % (dataset['slug'], role)
            )
            variant = '%s_sacti_%s_seed52026_ablation_v1' % (
                dataset['slug'], role
            )
            write_resolved_config(source, config_path, overrides, variant)
            verify_config(config_path, manifest, settings)
            jobs.append({
                'dataset': dataset['name'],
                'slug': dataset['slug'],
                'role': role,
                'display_name': settings['display_name'],
                'source_config': str(source),
                'source_config_sha256': dataset['source_config_sha256'],
                'config': str(config_path.resolve()),
                'config_sha256': sha256_file(config_path),
                'training_log': str((run_dir / 'logs' / (
                    '%s_%s.log' % (dataset['slug'], role)
                )).resolve()),
            })
    return jobs


def run_and_tee(command, log_path, environment):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            handle.write(line)
        return process.wait()


def completed(job, fold_count):
    path = Path(job['training_log'])
    if not path.is_file() or SUMMARY_MARKER not in path.read_text(
            encoding='utf-8', errors='replace'):
        return False
    try:
        parsed = parse_five_fold_log(path)
    except ValueError:
        return False
    return len(parsed['fold_metrics']['AUPR']) == fold_count


def run_jobs(manifest, jobs, python, resume):
    environment = os.environ.copy()
    environment['PATH'] = (
        str(Path(python).resolve().parent) + os.pathsep
        + environment.get('PATH', '')
    )
    for index, job in enumerate(jobs, start=1):
        if resume and completed(job, manifest['fold_count']):
            print('[%d/%d] Reusing %s %s.' % (
                index, len(jobs), job['dataset'], job['display_name']
            ))
            continue
        print('[%d/%d] Training %s %s.' % (
            index, len(jobs), job['dataset'], job['display_name']
        ))
        status = run_and_tee(
            [str(ROOT / 'run_hdcti.sh'), job['config']],
            job['training_log'],
            environment,
        )
        if status:
            raise RuntimeError(
                '%s %s failed with status %d; resume the same run directory.' % (
                    job['dataset'], job['display_name'], status
                )
            )


def stage_record(parsed, source):
    return {
        'metrics': {
            key: float(parsed['summary_metrics'][key]) for key in METRICS
        },
        'fold_aupr': [
            float(value) for value in parsed['fold_metrics']['AUPR']
        ],
        'source': str(Path(source).resolve()),
    }


def paired_effect(before, after):
    before_folds = np.asarray(before['fold_aupr'], dtype=np.float64)
    after_folds = np.asarray(after['fold_aupr'], dtype=np.float64)
    deltas = after_folds - before_folds
    return {
        'aupr_delta': float(
            after['metrics']['AUPR'] - before['metrics']['AUPR']
        ),
        'fold_aupr_deltas': deltas.tolist(),
        'positive_folds': int(np.sum(deltas > 0)),
        'negative_folds': int(np.sum(deltas < 0)),
    }


def factorial_interaction(stages):
    role_order = ('full', 'prototype_only', 'jaccard_only', 'no_transfer')
    means = [stages[key]['metrics']['AUPR'] for key in role_order]
    folds = [
        np.asarray(stages[key]['fold_aupr'], dtype=np.float64)
        for key in role_order
    ]
    fold_values = folds[0] - folds[1] - folds[2] + folds[3]
    return {
        'aupr_delta': float(means[0] - means[1] - means[2] + means[3]),
        'fold_aupr_deltas': fold_values.tolist(),
        'positive_folds': int(np.sum(fold_values > 0)),
        'negative_folds': int(np.sum(fold_values < 0)),
    }


def parse_mean(value):
    match = MEAN_PATTERN.match(str(value).strip())
    if not match:
        raise ValueError('Cannot parse metric mean: %s' % value)
    return float(match.group(1))


def load_external_baselines(path):
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row.get('status') != 'OK':
                continue
            rows.append({
                'dataset': row['dataset'],
                'method': row['method'],
                'aupr': parse_mean(row['AUPR']),
                'source': str(Path(path).resolve()),
            })
    return rows


def summarize(manifest, jobs):
    stages_by_slug = {}
    for job in jobs:
        parsed = parse_five_fold_log(job['training_log'])
        if len(parsed['fold_metrics']['AUPR']) != manifest['fold_count']:
            raise ValueError('Fold count mismatch for %s.' % job['training_log'])
        stages_by_slug.setdefault(job['slug'], {})[job['role']] = (
            stage_record(parsed, job['training_log'])
        )

    effect_specs = (
        ('support_gate', 'no_support_gate', 'full'),
        ('context_expert', 'no_context_expert', 'full'),
        ('joint_transfer', 'no_transfer', 'full'),
        ('prototype_without_jaccard', 'no_transfer', 'prototype_only'),
        ('jaccard_without_prototype', 'no_transfer', 'jaccard_only'),
        ('prototype_given_jaccard', 'jaccard_only', 'full'),
        ('jaccard_given_prototype', 'prototype_only', 'full'),
    )
    rows = []
    pooled = {name: [] for name, _, _ in effect_specs}
    pooled['prototype_jaccard_interaction'] = []
    for dataset in manifest['datasets']:
        stages = stages_by_slug[dataset['slug']]
        effects = {}
        for name, before, after in effect_specs:
            effect = paired_effect(stages[before], stages[after])
            effects[name] = effect
            pooled[name].extend(effect['fold_aupr_deltas'])
        interaction = factorial_interaction(stages)
        effects['prototype_jaccard_interaction'] = interaction
        pooled['prototype_jaccard_interaction'].extend(
            interaction['fold_aupr_deltas']
        )
        rows.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'stages': stages,
            'effects': effects,
        })

    effect_summary = {}
    for name, fold_values in pooled.items():
        dataset_values = [row['effects'][name]['aupr_delta'] for row in rows]
        fold_values = np.asarray(fold_values, dtype=np.float64)
        effect_summary[name] = {
            'macro_aupr_delta': float(np.mean(dataset_values)),
            'positive_datasets': int(np.sum(np.asarray(dataset_values) > 0)),
            'positive_folds': int(np.sum(fold_values > 0)),
            'negative_folds': int(np.sum(fold_values < 0)),
            'fold_count': int(fold_values.size),
        }

    external_path = repository_path(manifest['external_baselines_tsv'])
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': manifest['protocol'],
        'split_strategy': manifest['split_strategy'],
        'split_seed': int(manifest['split_seed']),
        'training_seed': int(manifest['training_seed']),
        'outer_test_scored': True,
        'model_selection_reopened': False,
        'rows': rows,
        'effect_summary': effect_summary,
        'external_baselines': load_external_baselines(external_path),
        'external_baselines_sha256': sha256_file(external_path),
        'reporting_rule': (
            'This frozen component ablation is descriptive. Every result is '
            'reported regardless of direction and does not reopen model selection.'
        ),
    }


def write_tsv(report, path):
    fields = ['dataset', 'role', *METRICS, 'source']
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in report['rows']:
            for role, stage in row['stages'].items():
                output = {
                    'dataset': row['dataset'],
                    'role': role,
                    'source': stage['source'],
                }
                output.update(stage['metrics'])
                writer.writerow(output)


def markdown(report):
    roles = (
        ('no_transfer', 'ID + H-C context'),
        ('prototype_only', 'Prototype only'),
        ('jaccard_only', 'Jaccard only'),
        ('full', 'Full SACTI'),
        ('no_support_gate', 'No support gate'),
        ('no_context_expert', 'No H-C context expert'),
    )
    lines = [
        '# Frozen SACTI Component Ablation',
        '',
        '- Split: compound cold-start, seed `52026`, five folds',
        '- Outer test scored: `True`',
        '- Model selection reopened: `False`',
        '',
        '| Dataset | ID + H-C context | + Prototype | + Jaccard | Full SACTI | No support gate | No H-C context expert |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['rows']:
        values = [row['stages'][key]['metrics']['AUPR'] for key, _ in roles]
        lines.append('| %s | %s |' % (
            row['dataset'],
            ' | '.join('%.6f' % value for value in values),
        ))
    lines.extend([
        '',
        '## Paired Component Effects',
        '',
        '| Effect | Macro AUPR delta | Positive datasets | Positive folds |',
        '|---|---:|---:|---:|',
    ])
    labels = {
        'support_gate': 'Support gate: Full - ungated',
        'context_expert': 'H-C context expert: Full - no H-C context',
        'joint_transfer': 'Joint transfer: Full - no transfer',
        'prototype_without_jaccard': 'Prototype, Jaccard off',
        'jaccard_without_prototype': 'Jaccard, prototype off',
        'prototype_given_jaccard': 'Prototype, Jaccard on',
        'jaccard_given_prototype': 'Jaccard, prototype on',
        'prototype_jaccard_interaction': 'Prototype x Jaccard interaction',
    }
    for key, values in report['effect_summary'].items():
        lines.append('| %s | %+.6f | %d/4 | %d/%d |' % (
            labels[key],
            values['macro_aupr_delta'],
            values['positive_datasets'],
            values['positive_folds'],
            values['fold_count'],
        ))

    full = {
        row['dataset']: row['stages']['full']['metrics']['AUPR']
        for row in report['rows']
    }
    external = {}
    for row in report['external_baselines']:
        external.setdefault(row['dataset'], {})[row['method']] = row['aupr']
    lines.extend([
        '',
        '## Existing Same-Input Baseline Interface',
        '',
        '| Dataset | Full SACTI | Best existing control | Control AUPR | Delta |',
        '|---|---:|---|---:|---:|',
    ])
    for dataset, sacti_value in full.items():
        controls = external.get(dataset, {})
        if not controls:
            continue
        method, value = max(controls.items(), key=lambda item: item[1])
        lines.append('| %s | %.6f | %s | %.6f | %+.6f |' % (
            dataset, sacti_value, method, value, sacti_value - value
        ))
    lines.extend(['', report['reporting_rule'], ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    parser.add_argument('--run-dir')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()

    manifest_path = repository_path(args.manifest)
    manifest = load_json(manifest_path)
    run_dir = (
        repository_path(args.run_dir) if args.run_dir else
        ROOT / 'results' / 'batch_runs' / (
            'sacti_component_ablation_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(manifest, run_dir)
    worklist = {
        'protocol': manifest['protocol'],
        'manifest': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'outer_test_scored': True,
        'model_selection_reopened': False,
        'jobs': jobs,
    }
    (run_dir / 'worklist.json').write_text(
        json.dumps(worklist, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    if args.dry_run:
        print(json.dumps(worklist, ensure_ascii=False, indent=2))
        return 0
    if not args.summarize_only:
        run_jobs(manifest, jobs, args.python, args.resume)
    report = summarize(manifest, jobs)
    (run_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    write_tsv(report, run_dir / 'results.tsv')
    text = markdown(report)
    (run_dir / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)
    print('Summary written to: %s' % (run_dir / 'summary.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

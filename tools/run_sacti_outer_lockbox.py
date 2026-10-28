#!/usr/bin/env python3
"""Run the sealed SACTI V1 unseen-split outer lockbox."""

import argparse
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

from tools.run_dg_schpt_pilot import parse_config, write_resolved_config  # noqa: E402
from tools.summarize_schpt_full import parse_five_fold_log  # noqa: E402
from util.config import ModelConf  # noqa: E402
from util.dataSplit import DataSplit  # noqa: E402


DEFAULT_MANIFEST = ROOT / 'configs' / 'sacti_outer_lockbox_seed901026.json'
CHECKPOINT_PATTERN = re.compile(
    r'^模型权重保存成功:\s*(.+)$', re.MULTILINE
)


def repository_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_resolved_config(path, role, dataset, manifest):
    _, values = parse_config(path)
    required = {
        'experiment.protocol': 'strict',
        'random.seed': str(manifest['training_seed']),
        'split.strategy': manifest['split_strategy'],
        'split.seed': str(manifest['split_seed']),
        'split.dir': './' + dataset['split_dir'],
        'split.reuse': 'True',
        'validation.seed': str(manifest['validation_seed']),
        'evaluation.setup': '-cv %d' % manifest['fold_count'],
        'evaluation.fold.limit': str(manifest['fold_count']),
        'evaluation.outer.test': 'True',
        'early.stopping': 'True',
        'validation.metric': 'AUPR',
    }
    for key, expected in required.items():
        if values.get(key) != expected:
            raise ValueError(
                '%s must be %s in %s.' % (key, expected, path)
            )
    expected_model = 'HDCTI' if role == 'baseline' else 'SACTI'
    if values.get('model.name') != expected_model:
        raise ValueError(
            '%s role must use model.name=%s.' % (role, expected_model)
        )
    if role == 'baseline':
        if values.get('attention.max.nodes') != '0':
            raise ValueError('HDCTI Final must keep dense attention disabled.')
        if values.get('herb.prototype.dual.granularity') != 'True':
            raise ValueError('HDCTI baseline is not the matched Final model.')
    else:
        if values.get('sacti.prototype') != 'True':
            raise ValueError('SACTI prototype branch must remain enabled.')
        if values.get('sacti.jaccard') != 'True':
            raise ValueError('SACTI Jaccard branch must remain enabled.')
        if values.get('sacti.context.layers') != '2':
            raise ValueError('SACTI V1 must retain two context layers.')
    return values


def build_jobs(manifest, run_dir):
    jobs = []
    for dataset in manifest['datasets']:
        for role in ('baseline', 'candidate'):
            source_key = role + '_config'
            hash_key = role + '_config_sha256'
            source_path = repository_path(dataset[source_key])
            if sha256_file(source_path) != dataset[hash_key]:
                raise ValueError(
                    'Frozen source config hash mismatch for %s %s.' % (
                        dataset['slug'], role
                    )
                )
            overrides = {
                'random.seed': str(manifest['training_seed']),
                'split.seed': str(manifest['split_seed']),
                'split.dir': './' + dataset['split_dir'],
                'split.reuse': 'True',
                'evaluation.fold.limit': str(manifest['fold_count']),
                'evaluation.outer.test': 'True',
            }
            config_path = run_dir / 'configs' / (
                '%s_%s.conf' % (dataset['slug'], role)
            )
            variant = '%s_%s_sacti_lockbox_split_%d_v1' % (
                dataset['slug'], role, manifest['split_seed']
            )
            write_resolved_config(
                source_path, config_path, overrides, variant
            )
            verify_resolved_config(
                config_path, role, dataset, manifest
            )
            jobs.append({
                'dataset': dataset['name'],
                'slug': dataset['slug'],
                'role': role,
                'model_name': 'HDCTI' if role == 'baseline' else 'SACTI',
                'config': str(config_path.resolve()),
                'config_sha256': sha256_file(config_path),
                'source_config': str(source_path),
                'source_config_sha256': dataset[hash_key],
                'split_dir': str(repository_path(dataset['split_dir'])),
                'training_log': str(
                    (run_dir / 'logs' / (
                        '%s_%s.log' % (dataset['slug'], role)
                    )).resolve()
                ),
                'ranking_root': str(
                    (run_dir / 'ranking' / dataset['slug'] / role).resolve()
                ),
            })
    return jobs


def preexisting_split_dirs(manifest):
    return [
        str(repository_path(dataset['split_dir']))
        for dataset in manifest['datasets']
        if repository_path(dataset['split_dir']).exists()
    ]


def validate_seal(seal, jobs):
    expected = {
        (job['slug'], job['role']): job['config_sha256'] for job in jobs
    }
    for row in seal['jobs']:
        key = (row['slug'], row['role'])
        if expected.get(key) != row['config_sha256']:
            raise ValueError('Sealed config hash changed: %s.' % (key,))
    for row in seal['splits']:
        if sha256_file(row['manifest']) != row['manifest_sha256']:
            raise ValueError('Sealed split manifest hash changed.')
        if sha256_file(row['assignments']) != row['assignments_sha256']:
            raise ValueError('Sealed split assignments hash changed.')


def seal_splits(manifest, jobs, run_dir, manifest_path):
    seal_path = run_dir / 'lockbox_seal.json'
    if seal_path.exists():
        seal = load_json(seal_path)
        validate_seal(seal, jobs)
        return seal

    baseline_jobs = {
        job['slug']: job for job in jobs if job['role'] == 'baseline'
    }
    split_rows = []
    for dataset in manifest['datasets']:
        job = baseline_jobs[dataset['slug']]
        conf = ModelConf(job['config'])
        _, split = DataSplit.prepareStrictFolds(
            conf, conf['datapath'], int(manifest['fold_count'])
        )
        if split['seed'] != int(manifest['split_seed']):
            raise ValueError('Generated lockbox split has the wrong seed.')
        if split['split_strategy'] != manifest['split_strategy']:
            raise ValueError('Generated lockbox split has the wrong strategy.')
        assignments = Path(split['assignments_path']).resolve()
        split_manifest = Path(job['split_dir']) / 'manifest.json'
        split_rows.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'manifest': str(split_manifest.resolve()),
            'manifest_sha256': sha256_file(split_manifest),
            'assignments': str(assignments),
            'assignments_sha256': sha256_file(assignments),
            'fold_stats': split['fold_stats'],
            'strict_guarantees': split['strict_guarantees'],
        })
    seal = {
        'sealed_at': datetime.now().astimezone().isoformat(),
        'sealed_before_training': True,
        'outer_results_seen_before_seal': False,
        'protocol': manifest['protocol'],
        'protocol_manifest': str(manifest_path),
        'protocol_manifest_sha256': sha256_file(manifest_path),
        'split_seed': int(manifest['split_seed']),
        'training_seed': int(manifest['training_seed']),
        'validation_seed': int(manifest['validation_seed']),
        'jobs': [
            {
                'dataset': job['dataset'],
                'slug': job['slug'],
                'role': job['role'],
                'model_name': job['model_name'],
                'config': job['config'],
                'config_sha256': job['config_sha256'],
            }
            for job in jobs
        ],
        'splits': split_rows,
    }
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return seal


def checkpoint_prefixes(log_path, expected_count):
    text = Path(log_path).read_text(encoding='utf-8', errors='replace')
    values = [repository_path(value.strip()) for value in CHECKPOINT_PATTERN.findall(text)]
    if len(values) != expected_count:
        raise ValueError(
            'Expected %d checkpoints in %s, found %d.' % (
                expected_count, log_path, len(values)
            )
        )
    if len({str(value) for value in values}) != expected_count:
        raise ValueError('Fold checkpoints are not unique in %s.' % log_path)
    for prefix in values:
        if not Path(str(prefix) + '.index').is_file():
            raise FileNotFoundError('%s.index is missing.' % prefix)
    return values


def completed_training(job, fold_count):
    path = Path(job['training_log'])
    if not path.is_file():
        return False
    try:
        parse_five_fold_log(path)
        checkpoint_prefixes(path, fold_count)
    except (ValueError, FileNotFoundError):
        return False
    return True


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


def run_training(manifest, jobs, python, resume):
    environment = os.environ.copy()
    environment['PATH'] = (
        str(Path(python).resolve().parent) + os.pathsep
        + environment.get('PATH', '')
    )
    for index, job in enumerate(jobs, start=1):
        if resume and completed_training(job, manifest['fold_count']):
            print('[%d/%d] Reusing %s %s training.' % (
                index, len(jobs), job['dataset'], job['role']
            ))
            continue
        print('[%d/%d] Training lockbox %s %s.' % (
            index, len(jobs), job['dataset'], job['role']
        ))
        status = run_and_tee(
            [str(ROOT / 'run_hdcti.sh'), job['config']],
            job['training_log'],
            environment,
        )
        if status:
            raise RuntimeError(
                '%s %s failed with status %d; resume the same run directory.' % (
                    job['dataset'], job['role'], status
                )
            )


def reusable_ranking_report(path, expected_model):
    try:
        report = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        report.get('evaluation_type') ==
        'checkpoint_only_pure_inference_outer'
        and report.get('training_optimizer_steps') == 0
        and report.get('split', {}).get('outer_test_scored') is True
        and report.get('config', {}).get('model_name') == expected_model
        and report.get('fixed_candidate_metrics') is not None
    )


def run_ranking(manifest, jobs, run_dir, python, resume):
    environment = os.environ.copy()
    environment['PATH'] = (
        str(Path(python).resolve().parent) + os.pathsep
        + environment.get('PATH', '')
    )
    total = len(jobs) * int(manifest['fold_count'])
    position = 0
    for job in jobs:
        checkpoints = checkpoint_prefixes(
            job['training_log'], manifest['fold_count']
        )
        job['checkpoints'] = [str(value) for value in checkpoints]
        for fold, checkpoint in enumerate(checkpoints, start=1):
            position += 1
            output_dir = Path(job['ranking_root']) / ('fold_%d' % fold)
            report_path = output_dir / 'report.json'
            if resume and reusable_ranking_report(
                    report_path, job['model_name']):
                print('[%d/%d] Reusing %s %s fold %d ranking.' % (
                    position, total, job['dataset'], job['role'], fold
                ))
                continue
            print('[%d/%d] Ranking %s %s fold %d.' % (
                position, total, job['dataset'], job['role'], fold
            ))
            command = [
                python,
                str(ROOT / 'tools' / 'evaluate_checkpoint_ranking.py'),
                '--config', job['config'],
                '--checkpoint', str(checkpoint),
                '--fold', str(fold),
                '--evaluation-split', 'outer',
                '--ks', *[
                    str(value) for value in manifest['ranking']['ks']
                ],
                '--export-top', str(manifest['ranking']['export_top']),
                '--output-dir', str(output_dir),
            ]
            log_path = run_dir / 'logs' / (
                '%s_%s_ranking_fold_%d.log' % (
                    job['slug'], job['role'], fold
                )
            )
            status = run_and_tee(command, log_path, environment)
            if status:
                raise RuntimeError(
                    '%s %s fold %d ranking failed with status %d.' % (
                        job['dataset'], job['role'], fold, status
                    )
                )


def ranking_signature(report):
    split = report['split']
    metrics = report['fixed_candidate_metrics']
    return {
        'outer_train_sha256': split['outer_train_sha256'],
        'outer_test_sha256': split['outer_test_sha256'],
        'strict_assignments_sha256': split['strict_assignments_sha256'],
        'entity_universe': report['entity_universe'],
        'fixed_candidate_protocol': report['fixed_candidate_protocol'],
        'evaluated_compounds': metrics['evaluated_compounds'],
        'held_out_positive_pairs': metrics['held_out_positive_pairs'],
        'candidate_pairs': metrics['candidate_pairs'],
    }


def load_ranking_report(job, fold):
    path = Path(job['ranking_root']) / ('fold_%d' % fold) / 'report.json'
    report = load_json(path)
    if report.get('evaluation_type') != 'checkpoint_only_pure_inference_outer':
        raise ValueError('Ranking report is not an outer pure-inference result.')
    if report.get('training_optimizer_steps') != 0:
        raise ValueError('Ranking report contains optimizer steps.')
    if report.get('split', {}).get('outer_test_scored') is not True:
        raise ValueError('Ranking report did not score the sealed outer fold.')
    if report.get('config', {}).get('model_name') != job['model_name']:
        raise ValueError('Ranking report model mismatch: %s.' % path)
    return report


def bootstrap_stratified(dataset_fold_deltas, spec):
    rng = np.random.default_rng(int(spec['seed']))
    arrays = [np.asarray(values, dtype=np.float64) for values in dataset_fold_deltas]
    draws = np.empty(int(spec['iterations']), dtype=np.float64)
    for index in range(draws.size):
        draws[index] = np.mean([
            np.mean(rng.choice(values, size=values.size, replace=True))
            for values in arrays
        ])
    alpha = (1.0 - float(spec['confidence'])) / 2.0
    return [
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    ]


def summarize(manifest, jobs, seal, run_dir):
    lookup = {(job['slug'], job['role']): job for job in jobs}
    rows = []
    fold_deltas_by_metric = {
        'AUPR': [], 'MRR': [], 'Recall@20': []
    }
    matched_ranking = []
    for dataset in manifest['datasets']:
        baseline_job = lookup[(dataset['slug'], 'baseline')]
        candidate_job = lookup[(dataset['slug'], 'candidate')]
        baseline_training = parse_five_fold_log(
            baseline_job['training_log']
        )
        candidate_training = parse_five_fold_log(
            candidate_job['training_log']
        )
        baseline_aupr = np.asarray(
            baseline_training['fold_metrics']['AUPR'], dtype=np.float64
        )
        candidate_aupr = np.asarray(
            candidate_training['fold_metrics']['AUPR'], dtype=np.float64
        )
        baseline_rank = {'MRR': [], 'Recall@20': []}
        candidate_rank = {'MRR': [], 'Recall@20': []}
        for fold in range(1, int(manifest['fold_count']) + 1):
            baseline_report = load_ranking_report(baseline_job, fold)
            candidate_report = load_ranking_report(candidate_job, fold)
            matched = (
                ranking_signature(baseline_report) ==
                ranking_signature(candidate_report)
            )
            matched_ranking.append(matched)
            if not matched:
                raise ValueError(
                    '%s fold %d ranking protocol mismatch.' % (
                        dataset['name'], fold
                    )
                )
            for metric in ('MRR', 'Recall@20'):
                baseline_rank[metric].append(float(
                    baseline_report['fixed_candidate_metrics'][metric]
                ))
                candidate_rank[metric].append(float(
                    candidate_report['fixed_candidate_metrics'][metric]
                ))
        metric_arrays = {
            'AUPR': (baseline_aupr, candidate_aupr),
            'MRR': (
                np.asarray(baseline_rank['MRR']),
                np.asarray(candidate_rank['MRR']),
            ),
            'Recall@20': (
                np.asarray(baseline_rank['Recall@20']),
                np.asarray(candidate_rank['Recall@20']),
            ),
        }
        row = {
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'metrics': {},
            'logs': {
                'baseline': baseline_job['training_log'],
                'candidate': candidate_job['training_log'],
            },
        }
        for metric, (baseline_values, candidate_values) in metric_arrays.items():
            deltas = candidate_values - baseline_values
            fold_deltas_by_metric[metric].append(deltas)
            row['metrics'][metric] = {
                'baseline_mean': float(np.mean(baseline_values)),
                'candidate_mean': float(np.mean(candidate_values)),
                'delta': float(np.mean(deltas)),
                'fold_deltas': deltas.tolist(),
                'positive_fold_count': int(np.sum(deltas > 0)),
            }
        rows.append(row)

    macro = {}
    for metric in ('AUPR', 'MRR', 'Recall@20'):
        dataset_deltas = np.asarray([
            row['metrics'][metric]['delta'] for row in rows
        ])
        macro[metric] = {
            'baseline': float(np.mean([
                row['metrics'][metric]['baseline_mean'] for row in rows
            ])),
            'candidate': float(np.mean([
                row['metrics'][metric]['candidate_mean'] for row in rows
            ])),
            'delta': float(np.mean(dataset_deltas)),
            'positive_dataset_count': int(np.sum(dataset_deltas > 0)),
            'positive_fold_count': int(np.sum([
                row['metrics'][metric]['positive_fold_count'] for row in rows
            ])),
            'stratified_fold_bootstrap_CI': bootstrap_stratified(
                fold_deltas_by_metric[metric], manifest['bootstrap']
            ),
        }

    threshold = manifest['interpretation_thresholds']
    checks = {
        'expected_dataset_count': len(rows) == 4,
        'expected_fold_count': all(
            len(values) == 5
            for metric in fold_deltas_by_metric.values()
            for values in metric
        ),
        'ranking_protocol_matched': all(matched_ranking),
        'seal_precedes_training': bool(seal['sealed_before_training']),
        'outer_results_unseen_before_seal': not bool(
            seal['outer_results_seen_before_seal']
        ),
        'minimum_macro_aupr_delta': (
            macro['AUPR']['delta'] >= threshold['minimum_macro_aupr_delta']
        ),
        'minimum_macro_mrr_delta': (
            macro['MRR']['delta'] >= threshold['minimum_macro_mrr_delta']
        ),
        'minimum_macro_recall_at_20_delta': (
            macro['Recall@20']['delta'] >=
            threshold['minimum_macro_recall_at_20_delta']
        ),
        'minimum_positive_aupr_dataset_count': (
            macro['AUPR']['positive_dataset_count'] >=
            threshold['minimum_positive_aupr_dataset_count']
        ),
        'minimum_positive_mrr_dataset_count': (
            macro['MRR']['positive_dataset_count'] >=
            threshold['minimum_positive_mrr_dataset_count']
        ),
        'minimum_positive_recall_at_20_dataset_count': (
            macro['Recall@20']['positive_dataset_count'] >=
            threshold['minimum_positive_recall_at_20_dataset_count']
        ),
        'minimum_dataset_aupr_delta': all(
            row['metrics']['AUPR']['delta'] >=
            threshold['minimum_dataset_aupr_delta'] for row in rows
        ),
        'minimum_dataset_mrr_delta': all(
            row['metrics']['MRR']['delta'] >=
            threshold['minimum_dataset_mrr_delta'] for row in rows
        ),
        'minimum_dataset_recall_at_20_delta': all(
            row['metrics']['Recall@20']['delta'] >=
            threshold['minimum_dataset_recall_at_20_delta'] for row in rows
        ),
        'minimum_positive_aupr_fold_count': (
            macro['AUPR']['positive_fold_count'] >=
            threshold['minimum_positive_aupr_fold_count']
        ),
    }
    if all(checks.values()):
        interpretation = 'SUPPORTS_SACTI_REPLACEMENT'
    elif macro['AUPR']['delta'] < 0 and macro['MRR']['delta'] < 0:
        interpretation = 'CHALLENGES_SACTI_REPLACEMENT'
    else:
        interpretation = 'MIXED'
    return {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': manifest['protocol'],
        'model_selection_reopened': False,
        'lockbox_consumed': True,
        'split_seed': int(manifest['split_seed']),
        'training_seed': int(manifest['training_seed']),
        'validation_seed': int(manifest['validation_seed']),
        'seal_sha256': sha256_file(run_dir / 'lockbox_seal.json'),
        'rows': rows,
        'macro': macro,
        'checks': checks,
        'interpretation': interpretation,
    }


def render_markdown(report):
    lines = [
        '# SACTI V1 Unseen Outer Lockbox',
        '',
        '- Split seed: `%d`' % report['split_seed'],
        '- Training seed: `%d`' % report['training_seed'],
        '- Model selection reopened: `false`',
        '- Lockbox consumed: `true`',
        '- Interpretation: **%s**' % report['interpretation'],
        '',
        '| Dataset | Final AUPR | SACTI AUPR | Delta | Final MRR | SACTI MRR | Delta | Final R@20 | SACTI R@20 | Delta |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['rows']:
        a = row['metrics']['AUPR']
        m = row['metrics']['MRR']
        r = row['metrics']['Recall@20']
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %+.6f | %.6f | %.6f | %+.6f |' % (
                row['dataset'], a['baseline_mean'], a['candidate_mean'],
                a['delta'], m['baseline_mean'], m['candidate_mean'],
                m['delta'], r['baseline_mean'], r['candidate_mean'],
                r['delta'],
            )
        )
    lines.extend(['', '## Macro Effects', ''])
    for metric in ('AUPR', 'MRR', 'Recall@20'):
        values = report['macro'][metric]
        ci = values['stratified_fold_bootstrap_CI']
        lines.append(
            '- %s: `%+.6f` (descriptive fold-bootstrap CI `[%+.6f, %+.6f]`); '
            'positive datasets `%d/4`, folds `%d/20`.' % (
                metric, values['delta'], ci[0], ci[1],
                values['positive_dataset_count'],
                values['positive_fold_count'],
            )
        )
    lines.extend(['', '## Integrity And Interpretation Checks', ''])
    for key, value in report['checks'].items():
        lines.append('- `%s`: `%s`' % (key, value))
    lines.extend([
        '',
        'The result is reported regardless of direction. Overlapping '
        'cross-validation training sets make the bootstrap intervals '
        'descriptive rather than population-level significance claims. The '
        'lockbox does not reopen model selection.',
        '',
    ])
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
    if manifest['protocol'] != 'sacti_v1_unseen_outer_lockbox_seed901026':
        raise ValueError('Unexpected SACTI outer lockbox protocol.')
    run_dir = (
        repository_path(args.run_dir) if args.run_dir else
        ROOT / 'results' / 'batch_runs' / 'sacti_outer_lockbox_seed901026'
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(manifest, run_dir)
    existing = preexisting_split_dirs(manifest)
    worklist = {
        'manifest': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'split_dirs_preexisting': existing,
        'jobs': jobs,
    }
    (run_dir / 'worklist.json').write_text(
        json.dumps(worklist, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    if args.dry_run:
        print(json.dumps({
            'protocol': manifest['protocol'],
            'run_dir': str(run_dir),
            'job_count': len(jobs),
            'ranking_job_count_after_training': (
                len(jobs) * int(manifest['fold_count'])
            ),
            'split_seed_unused': not bool(existing),
            'preexisting_split_dirs': existing,
            'jobs': jobs,
        }, ensure_ascii=False, indent=2))
        return 0 if not existing else 2
    if args.summarize_only and not (run_dir / 'lockbox_seal.json').exists():
        raise ValueError(
            '--summarize-only requires the original sealed run directory.'
        )
    if existing and not (args.resume or args.summarize_only):
        raise ValueError(
            'Lockbox split already exists. Use the original run directory '
            'with --resume; a fresh run is forbidden.'
        )
    seal = seal_splits(manifest, jobs, run_dir, manifest_path)
    if not args.summarize_only:
        run_training(manifest, jobs, args.python, args.resume)
        run_ranking(manifest, jobs, run_dir, args.python, args.resume)
    report = summarize(manifest, jobs, seal, run_dir)
    (run_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    text = render_markdown(report)
    (run_dir / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    print('Summary written to: %s' % (run_dir / 'summary.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

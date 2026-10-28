#!/usr/bin/env python3
"""Run the frozen paired DG-SCHPT inner-validation Pilot."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r'^模型权重保存成功:\s*(.+)$', re.MULTILINE)


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_config(path):
    rows = []
    values = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', ';')) or '=' not in line:
            rows.append((None, line))
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        values[key] = value.strip()
        rows.append((key, None))
    return rows, values


def write_resolved_config(base_path, output_path, overrides, variant):
    rows, values = parse_config(base_path)
    values.update({str(key): str(value) for key, value in overrides.items()})
    values['model.variant'] = variant
    written = set()
    lines = []
    for key, raw in rows:
        if key is None:
            lines.append(raw)
            continue
        if key in written:
            continue
        lines.append('%s=%s' % (key, values[key]))
        written.add(key)
    for key, value in values.items():
        if key not in written:
            lines.append('%s=%s' % (key, value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_and_tee(command, log_path, environment=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPOSITORY_ROOT),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def checkpoint_from_log(path):
    matches = CHECKPOINT_PATTERN.findall(
        path.read_text(encoding='utf-8', errors='replace')
    )
    if len(matches) != 1:
        raise ValueError(
            'Expected one checkpoint path in %s, found %d.' % (
                path, len(matches)
            )
        )
    return str(repository_path(matches[0].strip()))


def build_jobs(manifest, run_dir):
    jobs = []
    common = manifest['common_overrides']
    for dataset in manifest['datasets']:
        for role in ('baseline', 'candidate'):
            overrides = dict(common)
            if role == 'candidate':
                overrides.update(manifest['candidate_overrides'])
            variant = '%s_%s_%s_fold%d_v1' % (
                dataset['slug'],
                manifest.get('variant_prefix', 'dg_schpt'),
                role,
                int(manifest['fold']),
            )
            config_path = run_dir / 'configs' / ('%s_%s.conf' % (
                dataset['slug'], role
            ))
            write_resolved_config(
                repository_path(dataset['base_config']),
                config_path,
                overrides,
                variant,
            )
            jobs.append({
                'dataset': dataset['name'],
                'slug': dataset['slug'],
                'role': role,
                'config': str(config_path),
                'training_log': str(run_dir / 'logs' / ('%s_%s.log' % (
                    dataset['slug'], role
                ))),
                'ranking_dir': str(run_dir / 'ranking' / (
                    '%s_%s' % (dataset['slug'], role)
                )),
            })
    return jobs


def run_jobs(manifest, jobs, run_dir, python, resume):
    environment = os.environ.copy()
    environment['PATH'] = str(Path(python).resolve().parent) + os.pathsep + environment['PATH']
    for position, job in enumerate(jobs, start=1):
        log_path = Path(job['training_log'])
        report_path = Path(job['ranking_dir']) / 'report.json'
        print('\n[%d/%d] %s %s' % (
            position, len(jobs), job['dataset'], job['role']
        ))
        if not (resume and log_path.exists() and CHECKPOINT_PATTERN.search(
                log_path.read_text(encoding='utf-8', errors='replace'))):
            status = run_and_tee(
                [str(REPOSITORY_ROOT / 'run_hdcti.sh'), job['config']],
                log_path,
                environment=environment,
            )
            if status:
                raise RuntimeError(
                    '%s %s training failed with status %d.' % (
                        job['dataset'], job['role'], status
                    )
                )
        checkpoint = checkpoint_from_log(log_path)
        job['checkpoint'] = checkpoint
        if resume and report_path.exists():
            continue
        ranking_command = [
            python,
            str(REPOSITORY_ROOT / 'tools' / 'evaluate_checkpoint_ranking.py'),
            '--config', job['config'],
            '--checkpoint', checkpoint,
            '--fold', str(manifest['fold']),
            '--evaluation-split', 'validation',
            '--ks', *[str(value) for value in manifest['ranking']['ks']],
            '--export-top', str(manifest['ranking']['export_top']),
            '--output-dir', job['ranking_dir'],
        ]
        status = run_and_tee(
            ranking_command,
            run_dir / 'logs' / ('%s_%s_ranking.log' % (
                job['slug'], job['role']
            )),
            environment=environment,
        )
        if status:
            raise RuntimeError(
                '%s %s ranking failed with status %d.' % (
                    job['dataset'], job['role'], status
                )
            )


def summarize(manifest, jobs):
    grouped = {}
    for job in jobs:
        report = json.loads(
            (Path(job['ranking_dir']) / 'report.json').read_text(encoding='utf-8')
        )
        if report['split']['outer_test_scored']:
            raise ValueError('Outer test was scored for %s.' % job['dataset'])
        grouped.setdefault(job['slug'], {})[job['role']] = {
            'job': job,
            'sampled': report['sampled_pair_metrics_for_comparison_only'],
            'ranking': report['fixed_candidate_metrics'],
        }

    rows = []
    for dataset in manifest['datasets']:
        pair = grouped[dataset['slug']]
        baseline = pair['baseline']
        candidate = pair['candidate']
        rows.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'baseline': baseline,
            'candidate': candidate,
            'delta': {
                'sampled_AUPR': (
                    candidate['sampled']['AUPR'] - baseline['sampled']['AUPR']
                ),
                'MRR': candidate['ranking']['MRR'] - baseline['ranking']['MRR'],
                'Recall@20': (
                    candidate['ranking']['Recall@20']
                    - baseline['ranking']['Recall@20']
                ),
            },
        })
    sampled = [row['delta']['sampled_AUPR'] for row in rows]
    mrr = [row['delta']['MRR'] for row in rows]
    recall = [row['delta']['Recall@20'] for row in rows]
    gate = manifest['gate']
    checks = {
        'expected_dataset_count': len(rows) == 4,
        'outer_test_scored_false': True,
        'minimum_macro_sampled_aupr_delta': (
            float(np.mean(sampled))
            >= float(gate['minimum_macro_sampled_aupr_delta'])
        ),
        'minimum_macro_mrr_delta': (
            float(np.mean(mrr)) >= float(gate['minimum_macro_mrr_delta'])
        ),
        'minimum_macro_recall_at_20_delta': (
            float(np.mean(recall))
            >= float(gate['minimum_macro_recall_at_20_delta'])
        ),
        'maximum_dataset_sampled_aupr_drop': (
            min(sampled) >= -float(gate['maximum_dataset_sampled_aupr_drop'])
        ),
        'maximum_dataset_mrr_drop': (
            min(mrr) >= -float(gate['maximum_dataset_mrr_drop'])
        ),
    }
    if 'minimum_dataset_mrr_wins' in gate:
        checks['minimum_dataset_mrr_wins'] = (
            sum(value > 0 for value in mrr)
            >= int(gate['minimum_dataset_mrr_wins'])
        )
    if 'minimum_sampled_aupr_wins' in gate:
        checks['minimum_sampled_aupr_wins'] = (
            sum(value > 0 for value in sampled)
            >= int(gate['minimum_sampled_aupr_wins'])
        )
    return {
        'protocol': manifest['protocol'],
        'fold': int(manifest['fold']),
        'outer_test_scored': False,
        'rows': rows,
        'macro_delta': {
            'sampled_AUPR': float(np.mean(sampled)),
            'MRR': float(np.mean(mrr)),
            'Recall@20': float(np.mean(recall)),
        },
        'checks': checks,
        'gate': 'PASS' if all(checks.values()) else 'FAIL',
    }


def markdown(report):
    baseline_label = report.get('baseline_label', 'SCHPT')
    candidate_label = report.get('candidate_label', 'DG-SCHPT')
    lines = [
        '# %s' % report.get(
            'summary_title', 'DG-SCHPT Paired Inner-Validation Pilot'
        ),
        '',
        '- Fold: `%d`' % report['fold'],
        '- Outer test scored: `False`',
        '- Gate: `%s`' % report['gate'],
        '',
        '| Dataset | %s AUPR | %s AUPR | Delta | %s MRR | %s MRR | Delta | R@20 Delta |' % (
            baseline_label, candidate_label, baseline_label, candidate_label
        ),
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['rows']:
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %+.6f | %+.6f |' % (
                row['dataset'],
                row['baseline']['sampled']['AUPR'],
                row['candidate']['sampled']['AUPR'],
                row['delta']['sampled_AUPR'],
                row['baseline']['ranking']['MRR'],
                row['candidate']['ranking']['MRR'],
                row['delta']['MRR'],
                row['delta']['Recall@20'],
            )
        )
    macro = report['macro_delta']
    lines.extend([
        '',
        'Macro deltas: sampled AUPR `%+.6f`, MRR `%+.6f`, Recall@20 `%+.6f`.' % (
            macro['sampled_AUPR'], macro['MRR'], macro['Recall@20']
        ),
        '',
        '## Gate Checks',
        '',
    ])
    lines.extend(
        '- `%s`: `%s`' % (key, value)
        for key, value in report['checks'].items()
    )
    lines.extend([
        '',
        report.get(
            'gate_message',
            'A PASS permits a frozen four-dataset five-fold confirmation. '
            'A FAIL stops DG-SCHPT without reading new outer-test metrics.',
        ),
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='configs/dg_schpt_pilot_manifest.json'
    )
    parser.add_argument('--run-dir')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(
        repository_path(args.manifest).read_text(encoding='utf-8')
    )
    run_dir = (
        repository_path(args.run_dir)
        if args.run_dir else
        REPOSITORY_ROOT / 'results' / 'batch_runs' / (
            manifest.get('run_prefix', 'dg_schpt_pilot_')
            + datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(manifest, run_dir)
    (run_dir / 'worklist.json').write_text(
        json.dumps({'manifest': manifest, 'jobs': jobs}, indent=2) + '\n',
        encoding='utf-8',
    )
    if args.dry_run:
        print(json.dumps({
            'protocol': manifest['protocol'],
            'run_dir': str(run_dir),
            'outer_test_scored': False,
            'jobs': jobs,
        }, ensure_ascii=False, indent=2))
        return 0
    if not args.summarize_only:
        run_jobs(manifest, jobs, run_dir, args.python, args.resume)
    report = summarize(manifest, jobs)
    report.update({
        'summary_title': manifest.get(
            'summary_title', 'DG-SCHPT Paired Inner-Validation Pilot'
        ),
        'baseline_label': manifest.get('baseline_label', 'SCHPT'),
        'candidate_label': manifest.get('candidate_label', 'DG-SCHPT'),
        'gate_message': manifest.get(
            'gate_message',
            'A PASS permits a frozen four-dataset five-fold confirmation. '
            'A FAIL stops DG-SCHPT without reading new outer-test metrics.',
        ),
    })
    (run_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    text = markdown(report)
    (run_dir / 'summary.md').write_text(text, encoding='utf-8')
    print(text)
    print('Summary written to: %s' % (run_dir / 'summary.md'))
    return 0 if report['gate'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

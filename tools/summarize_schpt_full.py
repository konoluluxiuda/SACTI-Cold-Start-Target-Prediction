#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
SUMMARY_MARKER = 'The result of 5-fold cross validation:'
METADATA_PATTERN = re.compile(
    r'^Herb prototype metadata:\s*(.+)$', re.MULTILINE
)


def _metric_pattern(metric, require_space):
    separator = r'\s+' if require_space else r'\s*'
    return re.compile(
        r'^%s:%s([0-9]+(?:\.[0-9]+)?)' % (
            re.escape(metric), separator
        ),
        re.MULTILINE,
    )


def parse_five_fold_log(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='replace')
    if SUMMARY_MARKER not in text:
        raise ValueError('No five-fold summary found in %s.' % path)
    fold_text, summary_text = text.rsplit(SUMMARY_MARKER, 1)
    fold_metrics = {}
    summary_metrics = {}
    for metric in METRICS:
        fold_metrics[metric] = [
            float(value)
            for value in _metric_pattern(metric, True).findall(fold_text)
        ]
        summary_values = _metric_pattern(metric, False).findall(summary_text)
        if not summary_values:
            raise ValueError('No summary %s found in %s.' % (metric, path))
        summary_metrics[metric] = float(summary_values[0])
    fold_count = len(fold_metrics['AUPR'])
    if fold_count != 5:
        raise ValueError(
            'Expected 5 fold AUPR values in %s, found %d.' % (
                path, fold_count
            )
        )
    if any(len(values) != fold_count for values in fold_metrics.values()):
        raise ValueError('Fold metric counts disagree in %s.' % path)
    return {
        'text': text,
        'fold_metrics': fold_metrics,
        'summary_metrics': summary_metrics,
    }


def _load_candidate_metadata(text, expected_count):
    metadata_paths = METADATA_PATTERN.findall(text)
    if len(metadata_paths) != expected_count:
        raise ValueError(
            'Expected %d SCHPT metadata paths, found %d.' % (
                expected_count, len(metadata_paths)
            )
        )
    metadata = []
    for value in metadata_paths:
        path = Path(value.strip())
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        record = json.loads(path.read_text(encoding='utf-8'))
        record['metadata_path'] = str(path.resolve())
        metadata.append(record)
    return metadata


def summarize(run_dir, manifest_path):
    run_dir = Path(run_dir)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_folds = int(manifest['fold_count'])
    rows = []
    all_fold_deltas = []
    all_coverages = []
    all_scales = []

    for index, dataset in enumerate(manifest['datasets']):
        baseline_log = run_dir / ('%02d_%s_baseline.log' % (
            index * 2 + 1, dataset['slug']
        ))
        candidate_log = run_dir / ('%02d_%s_candidate.log' % (
            index * 2 + 2, dataset['slug']
        ))
        baseline = parse_five_fold_log(baseline_log)
        candidate = parse_five_fold_log(candidate_log)
        metadata = _load_candidate_metadata(
            candidate['text'], expected_folds
        )
        baseline_fold_aupr = np.asarray(
            baseline['fold_metrics']['AUPR'], dtype=np.float64
        )
        candidate_fold_aupr = np.asarray(
            candidate['fold_metrics']['AUPR'], dtype=np.float64
        )
        fold_deltas = candidate_fold_aupr - baseline_fold_aupr
        coverages = np.asarray([
            float(record['validation']['evidence_coverage'])
            for record in metadata
        ], dtype=np.float64)
        scales = np.asarray([
            float(record['learned_scale']) for record in metadata
        ], dtype=np.float64)
        all_fold_deltas.extend(fold_deltas.tolist())
        all_coverages.extend(coverages.tolist())
        all_scales.extend(scales.tolist())
        baseline_aupr = baseline['summary_metrics']['AUPR']
        candidate_aupr = candidate['summary_metrics']['AUPR']
        rows.append({
            'dataset': dataset['name'],
            'slug': dataset['slug'],
            'baseline_log': str(baseline_log.resolve()),
            'candidate_log': str(candidate_log.resolve()),
            'baseline_metrics': baseline['summary_metrics'],
            'candidate_metrics': candidate['summary_metrics'],
            'outer_aupr_delta': candidate_aupr - baseline_aupr,
            'baseline_fold_aupr': baseline_fold_aupr.tolist(),
            'candidate_fold_aupr': candidate_fold_aupr.tolist(),
            'fold_aupr_deltas': fold_deltas.tolist(),
            'positive_fold_count': int(np.sum(fold_deltas > 0)),
            'mean_evidence_coverage': float(np.mean(coverages)),
            'minimum_evidence_coverage': float(np.min(coverages)),
            'mean_learned_scale': float(np.mean(scales)),
            'minimum_absolute_learned_scale': float(
                np.min(np.abs(scales))
            ),
            'metadata_paths': [
                record['metadata_path'] for record in metadata
            ],
        })

    dataset_deltas = np.asarray([
        row['outer_aupr_delta'] for row in rows
    ], dtype=np.float64)
    fold_deltas = np.asarray(all_fold_deltas, dtype=np.float64)
    gate = manifest['gate']
    checks = {
        'mean_outer_aupr_delta_at_least_0.003': (
            float(np.mean(dataset_deltas))
            >= float(gate['minimum_mean_outer_aupr_delta'])
        ),
        'at_least_3_of_4_datasets_positive': (
            int(np.sum(dataset_deltas > 0))
            >= int(gate['minimum_positive_dataset_count'])
        ),
        'no_dataset_drop_below_minus_0.005': (
            float(np.min(dataset_deltas))
            >= float(gate['minimum_dataset_outer_aupr_delta'])
        ),
        'at_least_12_of_20_folds_positive': (
            int(np.sum(fold_deltas > 0))
            >= int(gate['minimum_positive_fold_count'])
            and fold_deltas.size == int(gate['expected_fold_count'])
        ),
        'all_fold_coverage_at_least_0.30': all(
            value >= float(gate['minimum_evidence_coverage_each_fold'])
            for value in all_coverages
        ),
        'all_fold_scales_nonzero': all(
            abs(value) > float(
                gate['minimum_absolute_learned_scale_each_fold']
            ) for value in all_scales
        ),
    }
    return {
        'protocol': manifest['protocol'],
        'manifest': str(manifest_path.resolve()),
        'rows': rows,
        'mean_outer_aupr_delta': float(np.mean(dataset_deltas)),
        'positive_dataset_count': int(np.sum(dataset_deltas > 0)),
        'positive_fold_count': int(np.sum(fold_deltas > 0)),
        'fold_count': int(fold_deltas.size),
        'checks': checks,
        'gate': 'PASS' if all(checks.values()) else 'NO-GO',
    }


def markdown(report):
    lines = [
        '# SCHPT Four-Dataset Five-Fold Confirmation',
        '',
        '- Protocol: frozen seed 52026, compound cold-start, five folds',
        '- Outer test: final evaluation only; model selection used inner validation',
        '- Mean outer AUPR delta: `%+.6f`' % report[
            'mean_outer_aupr_delta'
        ],
        '- Positive datasets: `%d/4`' % report['positive_dataset_count'],
        '- Positive paired folds: `%d/%d`' % (
            report['positive_fold_count'], report['fold_count']
        ),
        '- Gate: `%s`' % report['gate'],
        '',
        '| Dataset | Baseline AUPR | SCHPT AUPR | Delta | Positive folds | Min coverage | Mean scale |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in report['rows']:
        lines.append(
            '| %s | %.6f | %.6f | %+.6f | %d/5 | %.4f | %.6f |' % (
                row['dataset'],
                row['baseline_metrics']['AUPR'],
                row['candidate_metrics']['AUPR'],
                row['outer_aupr_delta'],
                row['positive_fold_count'],
                row['minimum_evidence_coverage'],
                row['mean_learned_scale'],
            )
        )
    lines.extend(['', '## Gate Checks', ''])
    lines.extend(
        '- `%s`: `%s`' % (name, value)
        for name, value in report['checks'].items()
    )
    lines.extend([
        '',
        'A PASS promotes SCHPT to complete four-dataset five-fold evidence. '
        'A NO-GO stops the candidate without dataset-specific tuning.',
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument(
        '--manifest', default='configs/schpt_full_manifest.json'
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    report = summarize(run_dir, args.manifest)
    (run_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (run_dir / 'summary.md').write_text(
        markdown(report), encoding='utf-8'
    )
    print('SCHPT five-fold confirmation: %s' % report['gate'])
    print('Summary: %s' % (run_dir / 'summary.md').resolve())
    return 0 if report['gate'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate fold-safe heuristics over the complete model protein universe."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_non_neural_cold_start_baselines import (  # noqa: E402
    METHODS,
    build_matrices_from_records,
    global_target_prior,
    hc_jaccard_label_propagation,
    herb_prototype_profiles,
    read_assignments,
    read_relation_pairs,
    repository_path,
    sha256_file,
)
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking  # noqa: E402


def fold_records(assignments, test_fold):
    outer_train = [list(row[:3]) for row in assignments if row[3] != test_fold]
    outer_test = [list(row[:3]) for row in assignments if row[3] == test_fold]
    return outer_train, outer_test


def profile_scorer(profiles, compound_ids, protein_ids):
    compound_index = {
        str(value): index for index, value in enumerate(compound_ids)
    }
    protein_index = {
        str(value): index for index, value in enumerate(protein_ids)
    }

    def score_pairs(compounds, proteins):
        rows = np.fromiter(
            (compound_index[str(value)] for value in compounds),
            dtype=np.int64,
            count=len(compounds),
        )
        columns = np.fromiter(
            (protein_index[str(value)] for value in proteins),
            dtype=np.int64,
            count=len(proteins),
        )
        return profiles[rows, columns]

    return score_pairs


def evaluate_records(
        hc_pairs, pd_pairs, train_records, test_records, fold, prior_strength, ks):
    pd_proteins = [protein_id for protein_id, _ in pd_pairs]
    matrices = build_matrices_from_records(
        hc_pairs,
        train_records,
        test_records,
        extra_protein_ids=pd_proteins,
    )
    protein_ids = matrices['protein_ids']
    test_compound_ids = matrices['test_compound_ids']

    prior, _ = global_target_prior(matrices['cp'])
    prototype, prototype_covered = herb_prototype_profiles(
        matrices['hc'],
        matrices['cp'],
        matrices['test_compounds'],
        prior_strength=prior_strength,
    )
    jaccard, jaccard_covered = hc_jaccard_label_propagation(
        matrices['hc'], matrices['cp'], matrices['test_compounds'])
    profiles = {
        'GlobalPrior': np.broadcast_to(
            prior, (len(test_compound_ids), len(protein_ids))),
        'HerbPrototype-EB': prototype,
        'HC-Jaccard-LP': jaccard,
    }
    coverage = {
        'GlobalPrior': 1.0,
        'HerbPrototype-EB': float(np.mean(prototype_covered)),
        'HC-Jaccard-LP': float(np.mean(jaccard_covered)),
    }

    methods = {}
    for method in METHODS:
        ranking = evaluate_fixed_candidate_ranking(
            protein_ids,
            train_records,
            test_records,
            profile_scorer(profiles[method], test_compound_ids, protein_ids),
            ks=ks,
            export_top=0,
        )
        methods[method] = {
            'coverage': coverage[method],
            'protocol': ranking['protocol'],
            'metrics': ranking['metrics'],
        }
    return {
        'fold': fold + 1,
        'training_positive_edges': matrices['training_positive_edges'],
        'entity_counts': matrices['entity_counts'],
        'test_compounds': len(test_compound_ids),
        'methods': methods,
    }


def evaluate_fold(
        hc_pairs, pd_pairs, assignments, fold, prior_strength, ks):
    outer_train, outer_test = fold_records(assignments, fold)
    return evaluate_records(
        hc_pairs,
        pd_pairs,
        outer_train,
        outer_test,
        fold,
        prior_strength,
        ks,
    )


def summarize_dataset(folds):
    summary = {}
    for method in METHODS:
        metric_names = [
            key for key, value in folds[0]['methods'][method]['metrics'].items()
            if isinstance(value, (int, float))
        ]
        summary[method] = {
            'coverage': {
                'mean': float(np.mean([
                    fold['methods'][method]['coverage'] for fold in folds
                ])),
                'folds': [
                    fold['methods'][method]['coverage'] for fold in folds
                ],
            },
            'metrics': {},
        }
        for metric in metric_names:
            values = [
                float(fold['methods'][method]['metrics'][metric])
                for fold in folds
            ]
            summary[method]['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'folds': values,
            }
    return summary


def render_markdown(report):
    lines = [
        '# Full-Candidate Heuristic Ranking',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Evaluation split: `%s`' % report.get('evaluation_split', 'outer'),
        '- Outer test scored: `%s`' % report.get('outer_test_scored', True),
        '- Split seed: `%d`' % report['split_seed'],
        '- Fold count: `%d`' % report['fold_count'],
        '- Candidate scope: all proteins in C-P assignments or P-D side information',
        '- Training/optimizer steps: `0`',
        '',
        '| Dataset | Method | MRR | Recall@20 | Hits@20 | Recall@50 | Hits@50 | Coverage |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in report['datasets']:
        for method in METHODS:
            row = dataset['summary'][method]
            metrics = row['metrics']
            lines.append(
                '| %s | %s | %.6f (±%.6f) | %.6f | %.6f | %.6f | %.6f | %.2f%% |' % (
                    dataset['name'],
                    method,
                    metrics['MRR']['mean'],
                    metrics['MRR']['std'],
                    metrics['Recall@20']['mean'],
                    metrics['Hits@20']['mean'],
                    metrics['Recall@50']['mean'],
                    metrics['Hits@50']['mean'],
                    100.0 * row['coverage']['mean'],
                )
            )
    lines.extend([
        '',
        'Unobserved pairs remain unlabeled. A high rank is not interpreted as a verified negative or positive.',
        '',
    ])
    return '\n'.join(lines)


def validation_folds(config_path, fold_count):
    from util.config import ModelConf
    from util.dataSplit import DataSplit
    from util.model_components import resolve_early_stopping

    conf = ModelConf(str(config_path))
    folds, _ = DataSplit.prepareStrictFolds(
        conf, conf['datapath'], fold_count)
    early_stopping = resolve_early_stopping(conf)
    if not early_stopping['enabled']:
        raise ValueError(
            'Validation ranking requires early.stopping=True in %s.' %
            config_path)
    base_seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
    validation_seed_base = (
        int(conf['validation.seed'])
        if conf.contains('validation.seed') else base_seed + 100000
    )
    result = []
    for fold, (outer_train, _) in enumerate(folds):
        model_train, validation, info = DataSplit.innerValidationSplitForConfig(
            conf,
            outer_train,
            early_stopping['ratio'],
            validation_seed_base + fold,
        )
        result.append((model_train, validation, info))
    return result


def evaluate_manifest(manifest_path, output_dir, evaluation_split='outer'):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fold_count = int(manifest['fold_count'])
    datasets = []
    for dataset in manifest['datasets']:
        hc_path = repository_path(dataset['hc_path'])
        pd_path = repository_path(dataset['pd_path'])
        split_dir = repository_path(dataset['split_dir'])
        split_manifest_path = split_dir / 'manifest.json'
        split_manifest = json.loads(
            split_manifest_path.read_text(encoding='utf-8'))
        if split_manifest.get('split_strategy') != 'compound_cold_start':
            raise ValueError('%s is not a compound cold-start split.' % split_dir)
        if int(split_manifest.get('seed')) != int(manifest['split_seed']):
            raise ValueError('%s uses the wrong split seed.' % split_dir)
        assignments_path = split_dir / 'fold_assignments.tsv'
        if sha256_file(assignments_path) != split_manifest['assignments_sha256']:
            raise ValueError('Assignment hash mismatch in %s.' % split_dir)

        hc_pairs = read_relation_pairs(hc_path)
        pd_pairs = read_relation_pairs(pd_path)
        assignments = read_assignments(assignments_path, fold_count)
        inner_folds = None
        if evaluation_split == 'validation':
            inner_folds = validation_folds(
                repository_path(dataset['config']), fold_count)
        folds = []
        for fold in range(fold_count):
            print('[%s] %s full-candidate heuristics fold %d/%d' % (
                dataset['name'], evaluation_split, fold + 1, fold_count))
            if evaluation_split == 'validation':
                model_train, validation, _ = inner_folds[fold]
                folds.append(evaluate_records(
                    hc_pairs,
                    pd_pairs,
                    model_train,
                    validation,
                    fold,
                    prior_strength=float(manifest['prior_strength']),
                    ks=manifest['ks'],
                ))
            else:
                folds.append(evaluate_fold(
                    hc_pairs,
                    pd_pairs,
                    assignments,
                    fold,
                    prior_strength=float(manifest['prior_strength']),
                    ks=manifest['ks'],
                ))
        datasets.append({
            **dataset,
            'hc_sha256': sha256_file(hc_path),
            'pd_sha256': sha256_file(pd_path),
            'assignments_sha256': sha256_file(assignments_path),
            'folds': folds,
            'summary': summarize_dataset(folds),
        })

    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': manifest['protocol'],
        'evaluation_split': evaluation_split,
        'outer_test_scored': evaluation_split == 'outer',
        'manifest': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'split_seed': int(manifest['split_seed']),
        'fold_count': fold_count,
        'prior_strength': float(manifest['prior_strength']),
        'ks': [int(value) for value in manifest['ks']],
        'methods': list(METHODS),
        'datasets': datasets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'summary.md').write_text(
        render_markdown(report), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='configs/full_candidate_ranking_gate.json')
    parser.add_argument(
        '--output-dir',
        default='results/full_candidate_ranking_gate/heuristics',
    )
    parser.add_argument(
        '--evaluation-split',
        choices=('outer', 'validation'),
        default='outer',
    )
    args = parser.parse_args()
    report = evaluate_manifest(
        repository_path(args.manifest),
        repository_path(args.output_dir),
        evaluation_split=args.evaluation_split,
    )
    print(render_markdown(report))
    print('Results written to: %s' % repository_path(args.output_dir))


if __name__ == '__main__':
    main()

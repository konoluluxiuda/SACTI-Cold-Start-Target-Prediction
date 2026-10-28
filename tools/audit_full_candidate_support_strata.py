#!/usr/bin/env python3
"""Audit where the frozen final ranker beats fold-safe H-C heuristics.

This is a descriptive, checkpoint-free audit. It reconstructs the prespecified
HerbPrototype-EB and HC-Jaccard-LP rankings inside each outer-training fold,
aligns them with saved DG-SCHPT V2 per-compound metrics, and stratifies paired
differences by observable support. It never trains a model or selects a new
configuration.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evaluate_full_candidate_heuristics import (  # noqa: E402
    fold_records,
    profile_scorer,
)
from tools.evaluate_non_neural_cold_start_baselines import (  # noqa: E402
    build_matrices_from_records,
    hc_jaccard_label_propagation,
    herb_prototype_profiles,
    read_assignments,
    read_relation_pairs,
    repository_path,
    sha256_file,
)
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking  # noqa: E402


METHODS = ('HerbPrototype-EB', 'HC-Jaccard-LP')
EPSILON = 1e-12


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def index_rows(rows):
    indexed = {}
    for row in rows:
        compound_id = str(row['compound_id'])
        if compound_id in indexed:
            raise ValueError('Duplicate compound row: %s.' % compound_id)
        indexed[compound_id] = row
    return indexed


def support_features(matrices):
    """Compute fold-local H-C support available to each test compound."""
    hc = matrices['hc'].tocsr()
    hc_csc = hc.tocsc()
    cp_degree = np.asarray(matrices['cp'].getnnz(axis=1)).reshape(-1)
    supported = np.flatnonzero(cp_degree > 0)
    supported_by_herb = np.zeros(hc.shape[0], dtype=np.float64)
    if supported.size:
        supported_by_herb = np.asarray(
            hc[:, supported].getnnz(axis=1) > 0,
            dtype=np.float64,
        ).reshape(-1)

    result = {}
    supported_hc = hc[:, supported] if supported.size else None
    supported_degrees = (
        np.asarray(supported_hc.getnnz(axis=0)).reshape(-1)
        if supported_hc is not None else np.asarray([], dtype=np.int64)
    )
    for compound_id, compound_index in zip(
            matrices['test_compound_ids'], matrices['test_compounds']):
        herbs = hc_csc[:, int(compound_index)].indices
        hc_degree = int(len(herbs))
        informative = int(np.sum(supported_by_herb[herbs])) if herbs.size else 0
        informative_fraction = (
            informative / float(hc_degree) if hc_degree else 0.0
        )

        neighbor_count = 0
        max_jaccard = 0.0
        if supported_hc is not None and hc_degree:
            shared = hc[herbs, :][:, supported].sum(axis=0)
            shared = np.asarray(shared).reshape(-1)
            neighbor_mask = shared > 0
            neighbor_count = int(np.sum(neighbor_mask))
            if neighbor_count:
                intersections = shared[neighbor_mask]
                unions = (
                    hc_degree + supported_degrees[neighbor_mask] - intersections
                )
                valid = unions > 0
                if np.any(valid):
                    max_jaccard = float(np.max(
                        intersections[valid] / unions[valid]))

        result[str(compound_id)] = {
            'hc_degree': hc_degree,
            'informative_herb_count': informative,
            'informative_herb_fraction': informative_fraction,
            'hc_train_neighbor_count': neighbor_count,
            'max_hc_jaccard': max_jaccard,
        }
    return result


def evaluate_heuristics(hc_pairs, pd_pairs, train_records, test_records, ks):
    matrices = build_matrices_from_records(
        hc_pairs,
        train_records,
        test_records,
        extra_protein_ids=[protein_id for protein_id, _ in pd_pairs],
    )
    prior_strength = 1.0
    prototype, _ = herb_prototype_profiles(
        matrices['hc'],
        matrices['cp'],
        matrices['test_compounds'],
        prior_strength=prior_strength,
    )
    jaccard, _ = hc_jaccard_label_propagation(
        matrices['hc'], matrices['cp'], matrices['test_compounds'])
    profiles = {
        'HerbPrototype-EB': prototype,
        'HC-Jaccard-LP': jaccard,
    }
    rankings = {}
    for method in METHODS:
        rankings[method] = evaluate_fixed_candidate_ranking(
            matrices['protein_ids'],
            train_records,
            test_records,
            profile_scorer(
                profiles[method],
                matrices['test_compound_ids'],
                matrices['protein_ids'],
            ),
            ks=ks,
            export_top=0,
        )
    return matrices, rankings


def fixed_bin(feature, value):
    value = float(value)
    if feature == 'hc_degree':
        if value <= 0:
            return '0'
        if value == 1:
            return '1'
        if value == 2:
            return '2'
        if value < 5:
            return '3-4'
        return '>=5'
    if feature == 'informative_herb_fraction':
        if value <= 0:
            return '0'
        if value < 0.5:
            return '(0,0.5)'
        if value < 1.0:
            return '[0.5,1)'
        return '1'
    if feature == 'hc_train_neighbor_count':
        if value <= 0:
            return '0'
        if value < 5:
            return '1-4'
        if value < 20:
            return '5-19'
        return '>=20'
    if feature == 'max_hc_jaccard':
        if value <= 0:
            return '0'
        if value < 0.25:
            return '(0,0.25)'
        if value < 0.5:
            return '[0.25,0.5)'
        return '>=0.5'
    if feature == 'test_positive_count':
        if value <= 1:
            return '1'
        if value < 5:
            return '2-4'
        if value < 10:
            return '5-9'
        return '>=10'
    raise ValueError('Unknown stratification feature: %s.' % feature)


def metric_summary(rows, heuristic_method):
    neural_rr = np.asarray([float(row['neural_rr']) for row in rows])
    heuristic_rr = np.asarray([
        float(row['%s_rr' % heuristic_method]) for row in rows
    ])
    neural_r20 = np.asarray([float(row['neural_recall20']) for row in rows])
    heuristic_r20 = np.asarray([
        float(row['%s_recall20' % heuristic_method]) for row in rows
    ])
    rr_delta = neural_rr - heuristic_rr
    r20_delta = neural_r20 - heuristic_r20
    return {
        'compounds': int(len(rows)),
        'neural_mrr': float(np.mean(neural_rr)),
        'heuristic_mrr': float(np.mean(heuristic_rr)),
        'mrr_delta': float(np.mean(rr_delta)),
        'neural_recall20': float(np.mean(neural_r20)),
        'heuristic_recall20': float(np.mean(heuristic_r20)),
        'recall20_delta': float(np.mean(r20_delta)),
        'rr_win_rate': float(np.mean(rr_delta > EPSILON)),
        'rr_tie_rate': float(np.mean(np.abs(rr_delta) <= EPSILON)),
        'rr_loss_rate': float(np.mean(rr_delta < -EPSILON)),
        'recall20_win_rate': float(np.mean(r20_delta > EPSILON)),
    }


def choose_fixed_heuristic(rows):
    means = {
        method: float(np.mean([
            float(row['%s_rr' % method]) for row in rows
        ]))
        for method in METHODS
    }
    return max(METHODS, key=lambda method: (means[method], method)), means


def build_summaries(rows):
    datasets = sorted({row['dataset'] for row in rows})
    overall = []
    selected = {}
    for dataset in datasets:
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        heuristic, means = choose_fixed_heuristic(dataset_rows)
        selected[dataset] = heuristic
        overall.append({
            'dataset': dataset,
            'fixed_heuristic': heuristic,
            'heuristic_candidate_mrr': means,
            **metric_summary(dataset_rows, heuristic),
        })

    features = (
        'hc_degree',
        'informative_herb_fraction',
        'hc_train_neighbor_count',
        'max_hc_jaccard',
        'test_positive_count',
    )
    stratum_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        heuristic = selected[dataset]
        for feature in features:
            grouped = defaultdict(list)
            for row in dataset_rows:
                label = fixed_bin(feature, row[feature])
                grouped[label].append(row)
            for label, group in grouped.items():
                stratum_rows.append({
                    'dataset': dataset,
                    'feature': feature,
                    'stratum': label,
                    'fixed_heuristic': heuristic,
                    **metric_summary(group, heuristic),
                })

    macro_strata = []
    macro_groups = defaultdict(list)
    for row in stratum_rows:
        macro_groups[(row['feature'], row['stratum'])].append(row)
    for (feature, stratum), group in sorted(macro_groups.items()):
        macro_strata.append({
            'feature': feature,
            'stratum': stratum,
            'datasets_represented': len(group),
            'compounds': int(sum(row['compounds'] for row in group)),
            'macro_mrr_delta': float(np.mean([
                row['mrr_delta'] for row in group
            ])),
            'positive_datasets_mrr': int(sum(
                row['mrr_delta'] > EPSILON for row in group
            )),
            'macro_recall20_delta': float(np.mean([
                row['recall20_delta'] for row in group
            ])),
            'positive_datasets_recall20': int(sum(
                row['recall20_delta'] > EPSILON for row in group
            )),
            'macro_rr_win_rate': float(np.mean([
                row['rr_win_rate'] for row in group
            ])),
        })

    stable = [
        row for row in macro_strata
        if row['datasets_represented'] >= 3
        and row['positive_datasets_mrr'] >= 3
        and row['macro_mrr_delta'] > 0
    ]
    verdict = (
        'stable_support_stratum_identified'
        if stable else 'no_stable_neural_support_stratum_identified'
    )
    return overall, stratum_rows, macro_strata, stable, verdict


def render_markdown(report):
    lines = [
        '# Full-Candidate Support-Stratified Audit',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Training/optimizer steps: `0`',
        '- Model selection reopened: `False`',
        '- Neural source: `%s`' % report['neural_root'],
        '- Split seed: `%d`' % report['split_seed'],
        '- Verdict: `%s`' % report['verdict'],
        '',
        'The fixed heuristic is selected once per database from the two '
        'prespecified non-neural methods using all frozen outer compounds. '
        'Strata are descriptive and must not be used to select a new model.',
        '',
        '## Overall comparison',
        '',
        '| Dataset | N | Neural MRR | Fixed heuristic | Heuristic MRR | Delta | RR win/tie/loss | Recall@20 delta |',
        '|---|---:|---:|---|---:|---:|---:|---:|',
    ]
    for row in report['overall']:
        lines.append(
            '| %s | %d | %.6f | %s | %.6f | %+.6f | %.1f%% / %.1f%% / %.1f%% | %+.6f |' % (
                row['dataset'],
                row['compounds'],
                row['neural_mrr'],
                row['fixed_heuristic'],
                row['heuristic_mrr'],
                row['mrr_delta'],
                100 * row['rr_win_rate'],
                100 * row['rr_tie_rate'],
                100 * row['rr_loss_rate'],
                row['recall20_delta'],
            )
        )

    lines.extend([
        '',
        '## Cross-database support strata',
        '',
        '| Feature | Stratum | Datasets | Compounds | Macro MRR delta | Positive datasets | Macro Recall@20 delta | RR win rate |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in report['macro_strata']:
        lines.append(
            '| %s | %s | %d | %d | %+.6f | %d/%d | %+.6f | %.1f%% |' % (
                row['feature'],
                row['stratum'],
                row['datasets_represented'],
                row['compounds'],
                row['macro_mrr_delta'],
                row['positive_datasets_mrr'],
                row['datasets_represented'],
                row['macro_recall20_delta'],
                100 * row['macro_rr_win_rate'],
            )
        )

    lines.extend(['', '## Interpretation boundary', ''])
    if report['stable_strata']:
        lines.append(
            'At least one prespecified support stratum had positive mean MRR '
            'differences in at least three databases. This identifies a '
            'descriptive operating niche, not a newly selected deployment rule.'
        )
    else:
        lines.append(
            'No prespecified support stratum showed a positive MRR difference '
            'in at least three databases. The audit therefore does not support '
            'a stable full-candidate niche for the neural ranker.'
        )
    lines.extend([
        '',
        'Unobserved candidate pairs remain unlabeled. Per-compound and '
        'dataset-specific rows are stored in TSV files for transparent review.',
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='configs/full_candidate_ranking_gate.json')
    parser.add_argument(
        '--neural-root',
        default=(
            'results/batch_runs/dg_schpt_v2_ranking_20260808_150634/v2'
        ),
    )
    parser.add_argument(
        '--output-dir', default='results/full_candidate_support_strata_audit')
    args = parser.parse_args()

    manifest_path = repository_path(args.manifest)
    neural_root = repository_path(args.neural_root)
    output_dir = repository_path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fold_count = int(manifest['fold_count'])
    all_rows = []
    input_hashes = {
        'manifest': sha256_file(manifest_path),
    }

    for dataset in manifest['datasets']:
        name = dataset['name']
        slug = dataset['slug']
        print('[%s] loading frozen support audit inputs' % name)
        hc_path = repository_path(dataset['hc_path'])
        pd_path = repository_path(dataset['pd_path'])
        assignment_path = repository_path(dataset['split_dir']) / 'fold_assignments.tsv'
        hc_pairs = read_relation_pairs(hc_path)
        pd_pairs = read_relation_pairs(pd_path)
        assignments = read_assignments(assignment_path, fold_count)
        input_hashes['%s_hc' % slug] = sha256_file(hc_path)
        input_hashes['%s_pd' % slug] = sha256_file(pd_path)
        input_hashes['%s_assignments' % slug] = sha256_file(assignment_path)

        for fold in range(fold_count):
            print('[%s] fold %d/%d' % (name, fold + 1, fold_count))
            train_records, test_records = fold_records(assignments, fold)
            matrices, rankings = evaluate_heuristics(
                hc_pairs, pd_pairs, train_records, test_records, manifest['ks'])
            support = support_features(matrices)
            heuristic_rows = {
                method: index_rows(rankings[method]['per_compound'])
                for method in METHODS
            }
            neural_path = (
                neural_root / slug / ('fold_%d' % (fold + 1)) /
                'per_compound_metrics.tsv'
            )
            if not neural_path.exists():
                raise FileNotFoundError(neural_path)
            input_hashes['%s_fold%d_neural' % (slug, fold + 1)] = sha256_file(
                neural_path)
            neural_rows = index_rows(read_tsv(neural_path))
            compound_ids = set(neural_rows)
            for method in METHODS:
                if set(heuristic_rows[method]) != compound_ids:
                    raise ValueError(
                        '%s fold %d compound mismatch for %s.' % (
                            name, fold + 1, method))
            if set(support) != compound_ids:
                raise ValueError(
                    '%s fold %d support compound mismatch.' % (name, fold + 1))

            for compound_id in sorted(compound_ids):
                neural = neural_rows[compound_id]
                row = {
                    'dataset': name,
                    'dataset_slug': slug,
                    'fold': fold + 1,
                    'compound_id': compound_id,
                    **support[compound_id],
                    'test_positive_count': int(neural['test_positive_count']),
                    'candidate_count': int(neural['candidate_count']),
                    'neural_rr': float(neural['reciprocal_rank']),
                    'neural_recall20': float(neural['recall@20']),
                }
                for method in METHODS:
                    heuristic = heuristic_rows[method][compound_id]
                    row['%s_rr' % method] = float(
                        heuristic['reciprocal_rank'])
                    row['%s_recall20' % method] = float(
                        heuristic['recall@20'])
                all_rows.append(row)

    overall, strata, macro_strata, stable, verdict = build_summaries(all_rows)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'frozen_full_candidate_support_strata_audit_v1',
        'training_optimizer_steps': 0,
        'model_selection_reopened': False,
        'split_seed': int(manifest['split_seed']),
        'fold_count': fold_count,
        'manifest': str(manifest_path),
        'neural_root': str(neural_root),
        'input_hashes': input_hashes,
        'overall': overall,
        'strata': strata,
        'macro_strata': macro_strata,
        'stable_strata': stable,
        'verdict': verdict,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'summary.md').write_text(
        render_markdown(report), encoding='utf-8')
    per_compound_fields = [
        'dataset', 'dataset_slug', 'fold', 'compound_id', 'hc_degree',
        'informative_herb_count', 'informative_herb_fraction',
        'hc_train_neighbor_count', 'max_hc_jaccard', 'test_positive_count',
        'candidate_count', 'neural_rr', 'neural_recall20',
        'HerbPrototype-EB_rr', 'HerbPrototype-EB_recall20',
        'HC-Jaccard-LP_rr', 'HC-Jaccard-LP_recall20',
    ]
    write_tsv(
        output_dir / 'per_compound.tsv', all_rows, per_compound_fields)
    stratum_fields = [
        'dataset', 'feature', 'stratum', 'fixed_heuristic', 'compounds',
        'neural_mrr', 'heuristic_mrr', 'mrr_delta', 'neural_recall20',
        'heuristic_recall20', 'recall20_delta', 'rr_win_rate',
        'rr_tie_rate', 'rr_loss_rate', 'recall20_win_rate',
    ]
    write_tsv(output_dir / 'strata.tsv', strata, stratum_fields)
    print(render_markdown(report))
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()

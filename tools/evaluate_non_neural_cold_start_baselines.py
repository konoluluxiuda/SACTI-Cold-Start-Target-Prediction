#!/usr/bin/env python3
"""Evaluate fold-safe non-neural baselines on compound cold-start splits."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
METHODS = ('GlobalPrior', 'HerbPrototype-EB', 'HC-Jaccard-LP')


def repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_relation_pairs(path):
    pairs = []
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError('Invalid relation row %d in %s.' % (
                    line_number, path))
            pairs.append((parts[0], parts[1]))
    return pairs


def read_assignments(path, fold_count):
    records = []
    with open(path, encoding='utf-8') as handle:
        header = next(handle, '').strip().split('\t')
        if header != ['left_id', 'right_id', 'label', 'fold']:
            raise ValueError('Invalid strict assignment header in %s.' % path)
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 4:
                raise ValueError('Invalid assignment row %d in %s.' % (
                    line_number, path))
            compound_id, protein_id, label_text, fold_text = parts
            label = int(label_text)
            fold = int(fold_text)
            if label not in (0, 1) or fold < 0 or fold >= fold_count:
                raise ValueError('Invalid label or fold at row %d in %s.' % (
                    line_number, path))
            records.append((compound_id, protein_id, label, fold))
    return records


def ordered_ids(values):
    return sorted(set(values), key=lambda value: (
        0, int(value)) if str(value).isdigit() else (1, str(value)))


def binary_csr(rows, columns, shape):
    matrix = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=shape,
        dtype=np.float32,
    ).tocsr()
    if matrix.nnz:
        matrix.data[:] = 1.0
        matrix.eliminate_zeros()
    return matrix


def build_matrices_from_records(
        hc_pairs, train_records, test_records, extra_protein_ids=()):
    compounds = ordered_ids(
        [compound for _, compound in hc_pairs]
        + [row[0] for row in train_records]
        + [row[0] for row in test_records]
    )
    proteins = ordered_ids(
        [row[1] for row in train_records]
        + [row[1] for row in test_records]
        + list(extra_protein_ids)
    )
    herbs = ordered_ids(herb for herb, _ in hc_pairs)
    compound_index = {value: index for index, value in enumerate(compounds)}
    protein_index = {value: index for index, value in enumerate(proteins)}
    herb_index = {value: index for index, value in enumerate(herbs)}

    hc_rows = [herb_index[herb] for herb, _ in hc_pairs]
    hc_columns = [compound_index[compound] for _, compound in hc_pairs]
    hc = binary_csr(
        hc_rows, hc_columns, (len(herbs), len(compounds)))

    train_positives = [
        row for row in train_records if float(row[2]) > 0
    ]
    cp_rows = [compound_index[row[0]] for row in train_positives]
    cp_columns = [protein_index[row[1]] for row in train_positives]
    cp = binary_csr(
        cp_rows, cp_columns, (len(compounds), len(proteins)))

    test_compound_ids = ordered_ids(row[0] for row in test_records)
    test_compounds = np.asarray(
        [compound_index[value] for value in test_compound_ids],
        dtype=np.int64,
    )
    test_compound_lookup = {
        value: index for index, value in enumerate(test_compound_ids)
    }
    record_rows = np.asarray(
        [test_compound_lookup[row[0]] for row in test_records],
        dtype=np.int64,
    )
    record_proteins = np.asarray(
        [protein_index[row[1]] for row in test_records], dtype=np.int64)
    labels = np.asarray([row[2] for row in test_records], dtype=np.int8)

    train_degrees = np.asarray(cp.getnnz(axis=1)).reshape(-1)
    if np.any(train_degrees[test_compounds] != 0):
        raise ValueError(
            'Test compounds have positive C-P support in the training records.')

    return {
        'hc': hc,
        'cp': cp,
        'compound_ids': compounds,
        'protein_ids': proteins,
        'test_compound_ids': test_compound_ids,
        'test_compounds': test_compounds,
        'record_rows': record_rows,
        'record_proteins': record_proteins,
        'labels': labels,
        'entity_counts': {
            'herbs': len(herbs),
            'compounds': len(compounds),
            'proteins': len(proteins),
        },
        'training_positive_edges': int(cp.nnz),
    }


def build_matrices(hc_pairs, assignments, test_fold, extra_protein_ids=()):
    train_records = [row[:3] for row in assignments if row[3] != test_fold]
    test_records = [row[:3] for row in assignments if row[3] == test_fold]
    return build_matrices_from_records(
        hc_pairs,
        train_records,
        test_records,
        extra_protein_ids=extra_protein_ids,
    )


def global_target_prior(cp):
    supported = np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0
    supported_count = int(np.sum(supported))
    if not supported_count:
        return np.zeros(cp.shape[1], dtype=np.float32), supported
    prior = (
        np.asarray(cp.sum(axis=0), dtype=np.float32).reshape(-1)
        / float(supported_count)
    )
    return prior.astype(np.float32), supported


def herb_prototype_profiles(hc, cp, test_compounds, prior_strength=1.0):
    prior, supported = global_target_prior(cp)
    supported_hc = hc[:, supported]
    support_counts = np.asarray(
        supported_hc.sum(axis=1), dtype=np.float32).reshape(-1)
    target_counts = supported_hc.dot(cp[supported]).toarray().astype(np.float32)
    posterior = (
        target_counts + float(prior_strength) * prior[None, :]
    ) / np.maximum(
        support_counts[:, None] + float(prior_strength),
        float(prior_strength),
    )

    test_hc = hc[:, test_compounds].transpose().tocsr()
    valid_herbs = (support_counts > 0).astype(np.float32)
    weighted_membership = test_hc.multiply(valid_herbs[None, :])
    valid_counts = np.asarray(
        weighted_membership.sum(axis=1), dtype=np.float32).reshape(-1)
    profiles = weighted_membership.dot(posterior)
    profiles = np.asarray(profiles, dtype=np.float32)
    covered = valid_counts > 0
    if np.any(covered):
        profiles[covered] /= valid_counts[covered, None]
    if np.any(~covered):
        profiles[~covered] = prior
    return profiles, covered


def hc_jaccard_label_propagation(hc, cp, test_compounds):
    prior, supported = global_target_prior(cp)
    supported_compounds = np.flatnonzero(supported)
    if not supported_compounds.size:
        return np.broadcast_to(
            prior, (len(test_compounds), len(prior))).copy(), np.zeros(
                len(test_compounds), dtype=bool)

    test_hc = hc[:, test_compounds].transpose().tocsr()
    train_hc = hc[:, supported_compounds].tocsr()
    intersections = test_hc.dot(train_hc).tocoo()
    test_degrees = np.asarray(test_hc.sum(axis=1)).reshape(-1)
    train_degrees = np.asarray(train_hc.sum(axis=0)).reshape(-1)
    unions = (
        test_degrees[intersections.row]
        + train_degrees[intersections.col]
        - intersections.data
    )
    similarities = sp.coo_matrix(
        (
            intersections.data.astype(np.float32)
            / np.maximum(unions.astype(np.float32), 1.0),
            (intersections.row, intersections.col),
        ),
        shape=intersections.shape,
        dtype=np.float32,
    ).tocsr()
    row_sums = np.asarray(similarities.sum(axis=1)).reshape(-1)
    covered = row_sums > 0
    normalizers = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float32),
        where=covered,
    )
    similarities = sp.diags(normalizers).dot(similarities)
    profiles = similarities.dot(cp[supported_compounds]).toarray().astype(
        np.float32)
    if np.any(~covered):
        profiles[~covered] = prior
    return profiles, covered


def hc_weighted_nearest_neighbor_profiles(
        hc, cp, test_compounds, neighbor_count=5, rank_decay=0.7):
    """Infer cold-compound target profiles from ranked H-C neighbors.

    This is an adapted weighted-nearest-neighbor profile baseline: H-C Jaccard
    replaces chemical similarity because a uniform molecular modality is not
    available across the four datasets. Only compounds with fold-local training
    C-P support can serve as neighbors.
    """
    neighbor_count = int(neighbor_count)
    rank_decay = float(rank_decay)
    if neighbor_count <= 0:
        raise ValueError('neighbor_count must be positive.')
    if not 0.0 < rank_decay <= 1.0:
        raise ValueError('rank_decay must be in (0, 1].')

    prior, supported = global_target_prior(cp)
    supported_compounds = np.flatnonzero(supported)
    profile_shape = (len(test_compounds), cp.shape[1])
    if not supported_compounds.size:
        return (
            np.broadcast_to(prior, profile_shape).copy(),
            np.zeros(len(test_compounds), dtype=bool),
            np.zeros(len(test_compounds), dtype=np.int64),
        )

    test_hc = hc[:, test_compounds].transpose().tocsr()
    train_hc = hc[:, supported_compounds].tocsr()
    intersections = test_hc.dot(train_hc).tocoo()
    test_degrees = np.asarray(test_hc.sum(axis=1)).reshape(-1)
    train_degrees = np.asarray(train_hc.sum(axis=0)).reshape(-1)
    unions = (
        test_degrees[intersections.row]
        + train_degrees[intersections.col]
        - intersections.data
    )
    similarities = sp.coo_matrix(
        (
            intersections.data.astype(np.float32)
            / np.maximum(unions.astype(np.float32), 1.0),
            (intersections.row, intersections.col),
        ),
        shape=intersections.shape,
        dtype=np.float32,
    ).tocsr()

    profiles = np.broadcast_to(prior, profile_shape).copy()
    covered = np.zeros(len(test_compounds), dtype=bool)
    used_neighbors = np.zeros(len(test_compounds), dtype=np.int64)
    decay = np.power(
        np.float32(rank_decay),
        np.arange(neighbor_count, dtype=np.float32),
    )
    supported_cp = cp[supported_compounds]
    for row in range(similarities.shape[0]):
        start, end = similarities.indptr[row:row + 2]
        columns = similarities.indices[start:end]
        values = similarities.data[start:end]
        if not values.size:
            continue

        # The supported compound index is the deterministic secondary key.
        order = np.lexsort((columns, -values))[:neighbor_count]
        neighbor_columns = columns[order]
        neighbor_similarities = values[order]
        normalizer = float(np.sum(neighbor_similarities))
        if normalizer <= 0.0:
            continue
        weights = neighbor_similarities * decay[:len(order)] / normalizer
        profiles[row] = np.asarray(
            weights @ supported_cp[neighbor_columns], dtype=np.float32
        ).reshape(-1)
        covered[row] = True
        used_neighbors[row] = len(order)
    return profiles, covered, used_neighbors


def metric_pair(labels, scores):
    return {
        'AUC': float(roc_auc_score(labels, scores)),
        'AUPR': float(average_precision_score(labels, scores)),
    }


def evaluate_fold(hc_pairs, assignments, fold, prior_strength):
    matrices = build_matrices(hc_pairs, assignments, fold)
    hc = matrices['hc']
    cp = matrices['cp']
    test_compounds = matrices['test_compounds']
    record_rows = matrices['record_rows']
    record_proteins = matrices['record_proteins']
    labels = matrices['labels']

    prior, _ = global_target_prior(cp)
    prototype_profiles, prototype_covered = herb_prototype_profiles(
        hc, cp, test_compounds, prior_strength=prior_strength)
    jaccard_profiles, jaccard_covered = hc_jaccard_label_propagation(
        hc, cp, test_compounds)
    scores = {
        'GlobalPrior': prior[record_proteins],
        'HerbPrototype-EB': prototype_profiles[
            record_rows, record_proteins],
        'HC-Jaccard-LP': jaccard_profiles[record_rows, record_proteins],
    }
    coverage = {
        'GlobalPrior': 1.0,
        'HerbPrototype-EB': float(np.mean(prototype_covered[record_rows])),
        'HC-Jaccard-LP': float(np.mean(jaccard_covered[record_rows])),
    }
    return {
        'fold': fold,
        'records': int(labels.size),
        'positives': int(np.sum(labels)),
        'training_positive_edges': matrices['training_positive_edges'],
        'entity_counts': matrices['entity_counts'],
        'methods': {
            method: {
                **metric_pair(labels, scores[method]),
                'coverage': coverage[method],
            }
            for method in METHODS
        },
    }


def summarize_folds(folds):
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ('AUC', 'AUPR', 'coverage'):
            values = [fold['methods'][method][metric] for fold in folds]
            summary[method][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'folds': values,
            }
    return summary


def load_ours_results(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['dataset']: {
            'mean': float(np.mean(row['candidate_fold_aupr'])),
            'folds': [float(value) for value in row['candidate_fold_aupr']],
        }
        for row in data['rows']
    }


def render_markdown(report):
    lines = [
        '# Non-Neural Compound Cold-Start Baselines',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Split seed: `%d`' % report['split_seed'],
        '- Fold count: `%d`' % report['fold_count'],
        '- Training/optimizer steps: `0`',
        '',
        'All C-P statistics are rebuilt from the current fold training positives.',
        'Test compounds have zero training C-P support; H-C side information remains available.',
        '',
        '| Dataset | GlobalPrior AUPR | HerbPrototype-EB AUPR | HC-Jaccard-LP AUPR | Ours-full AUPR | Ours - best heuristic |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for dataset in report['datasets']:
        methods = dataset['summary']
        heuristic_values = [methods[name]['AUPR']['mean'] for name in METHODS]
        lines.append(
            '| %s | %.6f (±%.6f) | %.6f (±%.6f) | %.6f (±%.6f) | %.6f | %+.6f |' % (
                dataset['name'],
                methods['GlobalPrior']['AUPR']['mean'],
                methods['GlobalPrior']['AUPR']['std'],
                methods['HerbPrototype-EB']['AUPR']['mean'],
                methods['HerbPrototype-EB']['AUPR']['std'],
                methods['HC-Jaccard-LP']['AUPR']['mean'],
                methods['HC-Jaccard-LP']['AUPR']['std'],
                dataset['ours_full_aupr'],
                dataset['ours_full_aupr'] - max(heuristic_values),
            )
        )
    lines.extend([
        '| **Macro** | %.6f | %.6f | %.6f | %.6f | %+.6f |' % (
            report['macro']['GlobalPrior'],
            report['macro']['HerbPrototype-EB'],
            report['macro']['HC-Jaccard-LP'],
            report['macro']['Ours-full'],
            report['macro']['Ours-minus-best-heuristic'],
        ),
        '',
        '## Paired Fold AUPR Delta (Ours-full - Heuristic)',
        '',
        '| Dataset | vs GlobalPrior | Positive folds | vs HerbPrototype-EB | Positive folds | vs HC-Jaccard-LP | Positive folds |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for dataset in report['datasets']:
        paired = dataset['paired_aupr_delta']
        lines.append(
            '| %s | %+.6f | %d/5 | %+.6f | %d/5 | %+.6f | %d/5 |' % (
                dataset['name'],
                paired['GlobalPrior']['mean'],
                paired['GlobalPrior']['positive_fold_count'],
                paired['HerbPrototype-EB']['mean'],
                paired['HerbPrototype-EB']['positive_fold_count'],
                paired['HC-Jaccard-LP']['mean'],
                paired['HC-Jaccard-LP']['positive_fold_count'],
            )
        )
    lines.extend([
        '',
        '## Evidence Coverage',
        '',
        '| Dataset | HerbPrototype-EB | HC-Jaccard-LP |',
        '|---|---:|---:|',
    ])
    for dataset in report['datasets']:
        methods = dataset['summary']
        lines.append('| %s | %.2f%% | %.2f%% |' % (
            dataset['name'],
            100.0 * methods['HerbPrototype-EB']['coverage']['mean'],
            100.0 * methods['HC-Jaccard-LP']['coverage']['mean'],
        ))
    lines.extend([
        '',
        '## Interpretation Boundary',
        '',
        '- `GlobalPrior` uses only fold-local protein prevalence.',
        '- `HerbPrototype-EB` averages empirical-Bayes herb-target posteriors with prior strength `%.3g`.' % report['prior_strength'],
        '- `HC-Jaccard-LP` propagates training C-P labels from H-C Jaccard-similar supported compounds.',
        '- Uncovered compounds fall back to `GlobalPrior`.',
        '- These are inference-only controls, not trainable competitors.',
        '',
    ])
    return '\n'.join(lines)


def evaluate_manifest(manifest_path, output_dir):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    fold_count = int(manifest['fold_count'])
    ours_results_path = repository_path(manifest['ours_results'])
    ours_results = load_ours_results(ours_results_path)
    dataset_reports = []

    for dataset in manifest['datasets']:
        hc_path = repository_path(dataset['hc_path'])
        split_dir = repository_path(dataset['split_dir'])
        split_manifest_path = split_dir / 'manifest.json'
        split_manifest = json.loads(
            split_manifest_path.read_text(encoding='utf-8'))
        if split_manifest.get('split_strategy') != 'compound_cold_start':
            raise ValueError('%s is not a compound cold-start split.' % split_dir)
        if split_manifest.get('seed') != manifest['split_seed']:
            raise ValueError('%s uses the wrong split seed.' % split_dir)
        assignments_path = split_dir / 'fold_assignments.tsv'
        if sha256_file(assignments_path) != split_manifest['assignments_sha256']:
            raise ValueError('Assignment hash mismatch in %s.' % split_dir)

        hc_pairs = read_relation_pairs(hc_path)
        assignments = read_assignments(assignments_path, fold_count)
        folds = []
        for fold in range(fold_count):
            print('[%s] evaluating fold %d/%d' % (
                dataset['name'], fold + 1, fold_count))
            folds.append(evaluate_fold(
                hc_pairs,
                assignments,
                fold,
                prior_strength=manifest['prior_strength'],
            ))
        if dataset['name'] not in ours_results:
            raise ValueError('Missing Ours-full result for %s.' % dataset['name'])
        ours = ours_results[dataset['name']]
        if len(ours['folds']) != fold_count:
            raise ValueError('Ours-full fold count mismatch for %s.' % dataset['name'])
        fold_summary = summarize_folds(folds)
        paired = {}
        for method in METHODS:
            method_folds = fold_summary[method]['AUPR']['folds']
            deltas = [
                ours_value - method_value
                for ours_value, method_value in zip(ours['folds'], method_folds)
            ]
            paired[method] = {
                'mean': float(np.mean(deltas)),
                'folds': deltas,
                'positive_fold_count': int(np.sum(np.asarray(deltas) > 0)),
            }
        dataset_reports.append({
            **dataset,
            'hc_sha256': sha256_file(hc_path),
            'assignments_sha256': sha256_file(assignments_path),
            'folds': folds,
            'summary': fold_summary,
            'ours_full_aupr': ours['mean'],
            'ours_full_fold_aupr': ours['folds'],
            'paired_aupr_delta': paired,
        })

    macro = {
        method: float(np.mean([
            row['summary'][method]['AUPR']['mean']
            for row in dataset_reports
        ]))
        for method in METHODS
    }
    macro['Ours-full'] = float(np.mean([
        row['ours_full_aupr'] for row in dataset_reports]))
    macro['Best-heuristic-per-dataset'] = float(np.mean([
        max(row['summary'][method]['AUPR']['mean'] for method in METHODS)
        for row in dataset_reports
    ]))
    macro['Ours-minus-best-heuristic'] = (
        macro['Ours-full'] - macro['Best-heuristic-per-dataset'])
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': manifest['protocol'],
        'manifest': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'ours_results': str(ours_results_path),
        'ours_results_sha256': sha256_file(ours_results_path),
        'split_seed': manifest['split_seed'],
        'fold_count': fold_count,
        'prior_strength': manifest['prior_strength'],
        'methods': list(METHODS),
        'datasets': dataset_reports,
        'macro': macro,
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
        '--manifest',
        default='configs/non_neural_cold_start_baselines.json',
    )
    parser.add_argument(
        '--output-dir',
        default='results/non_neural_cold_start_baselines',
    )
    args = parser.parse_args()
    manifest_path = repository_path(args.manifest)
    output_dir = repository_path(args.output_dir)
    report = evaluate_manifest(manifest_path, output_dir)
    print(render_markdown(report))
    print('Results written to: %s' % output_dir)


if __name__ == '__main__':
    main()

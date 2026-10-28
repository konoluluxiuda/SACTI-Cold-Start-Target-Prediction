#!/usr/bin/env python3
"""Evaluate all held-out ETCM Confirmed Targets with frozen checkpoints."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
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
    global_target_prior,
    hc_jaccard_label_propagation,
    herb_prototype_profiles,
    read_assignments,
    read_relation_pairs,
)
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking  # noqa: E402
from util.config import ModelConf  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', default='configs/etcm_confirmed_target_evaluation.json')
    parser.add_argument(
        '--output-dir', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def resolve_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def validate_hash(path, expected, label):
    if not path.exists():
        raise FileNotFoundError('%s not found: %s' % (label, path))
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            '%s hash mismatch: expected %s, got %s.' %
            (label, expected, actual)
        )


def relation_pairs(path):
    pairs = set()
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(
                    'Malformed relation row %d in %s.' % (line_number, path))
            pairs.add((str(parts[0]), str(parts[1])))
    return pairs


def validate_config(path, split_seed):
    conf = ModelConf(str(path))
    required = {
        'experiment.protocol': 'strict',
        'split.strategy': 'compound_cold_start',
        'split.seed': str(split_seed),
        'evaluation.outer.test': 'True',
    }
    if conf['model.name'].strip() == 'HDCTI':
        required['attention.max.nodes'] = '0'
    mismatches = {
        key: {'expected': expected, 'observed': conf[key]}
        for key, expected in required.items()
        if not conf.contains(key) or str(conf[key]) != expected
    }
    if mismatches:
        raise ValueError('Frozen config mismatch in %s: %s.' % (path, mismatches))


def validate_protocol(manifest, manifest_path):
    frozen_statuses = {
        'FROZEN_BEFORE_DATASET_WIDE_SCORING',
        'FROZEN_BEFORE_SACTI_DATASET_WIDE_SCORING',
    }
    if manifest.get('status') not in frozen_statuses:
        raise ValueError('The dataset-wide evidence protocol is not frozen.')
    if manifest.get('outer_test_scored_for_model_selection'):
        raise ValueError('Outer-test evidence cannot be used for model selection.')

    evidence_path = resolve_path(manifest['evidence']['path'])
    evidence_manifest = resolve_path(manifest['evidence']['manifest'])
    assignment_path = resolve_path(manifest['split_assignments']['path'])
    validate_hash(evidence_path, manifest['evidence']['sha256'], 'Evidence table')
    validate_hash(
        evidence_manifest, manifest['evidence']['manifest_sha256'],
        'Evidence manifest')
    validate_hash(
        assignment_path, manifest['split_assignments']['sha256'],
        'Fold assignments')

    graph_paths = {}
    for key in ('cp', 'hc', 'pd'):
        path = resolve_path(manifest['model_graph']['%s_path' % key])
        validate_hash(
            path, manifest['model_graph']['%s_sha256' % key],
            '%s relation' % key.upper())
        graph_paths[key] = path
    protein_mapping = resolve_path(manifest['model_graph']['protein_mapping'])
    validate_hash(
        protein_mapping,
        manifest['model_graph']['protein_mapping_sha256'],
        'Protein mapping')

    evidence_rows = read_tsv(evidence_path)
    evidence_pairs = [
        (str(row['compound_id']), str(row['protein_id']))
        for row in evidence_rows
    ]
    if len(evidence_pairs) != len(set(evidence_pairs)):
        raise ValueError('The frozen evidence table contains duplicate pairs.')
    if len(evidence_pairs) != int(manifest['evidence']['expected_pairs']):
        raise ValueError('Unexpected evidence-pair count: %d.' % len(evidence_pairs))
    compounds = sorted({pair[0] for pair in evidence_pairs})
    if len(compounds) != int(manifest['evidence']['expected_compounds']):
        raise ValueError('Unexpected evidence-compound count: %d.' % len(compounds))
    invalid_status = sorted({
        row.get('status', '') for row in evidence_rows
        if row.get('status') != 'unseen_confirmed'
    })
    if invalid_status:
        raise ValueError('Unexpected evidence statuses: %s.' % invalid_status)
    overlap = set(evidence_pairs) & relation_pairs(graph_paths['cp'])
    if overlap:
        raise ValueError(
            '%d Confirmed Target pairs overlap the model C-P graph.' %
            len(overlap))

    protein_rows = []
    with open(protein_mapping, newline='', encoding='utf-8') as handle:
        protein_rows = list(csv.DictReader(handle))
    protein_ids = {str(row['protein_id']) for row in protein_rows}
    if len(protein_ids) != int(manifest['model_graph']['expected_proteins']):
        raise ValueError('Unexpected mapped-protein count: %d.' % len(protein_ids))
    missing_proteins = sorted({pair[1] for pair in evidence_pairs} - protein_ids)
    if missing_proteins:
        raise ValueError(
            'Evidence contains proteins outside the candidate universe: %s.' %
            missing_proteins)

    assignments = read_assignments(assignment_path, int(manifest['fold_count']))
    compound_folds = defaultdict(set)
    positive_test_compounds = defaultdict(set)
    for compound_id, _, label, fold in assignments:
        compound_folds[str(compound_id)].add(int(fold) + 1)
        if int(label) > 0:
            positive_test_compounds[int(fold) + 1].add(str(compound_id))
    invalid_folds = {
        compound_id: sorted(compound_folds.get(compound_id, set()))
        for compound_id in compounds
        if len(compound_folds.get(compound_id, set())) != 1
    }
    if invalid_folds:
        raise ValueError(
            'Evidence compounds lack one unique outer fold: %s.' % invalid_folds)

    compounds_by_fold = defaultdict(list)
    for compound_id in compounds:
        fold = next(iter(compound_folds[compound_id]))
        if compound_id not in positive_test_compounds[fold]:
            raise ValueError(
                'Evidence compound %s has no positive row in outer fold %d.' %
                (compound_id, fold))
        compounds_by_fold[fold].append(compound_id)

    neural_methods = {}
    for method in manifest['neural_methods']:
        config_path = resolve_path(method['config'])
        validate_hash(
            config_path, method['config_sha256'],
            '%s config' % method['name'])
        validate_config(config_path, int(manifest['split_seed']))
        checkpoints = {}
        for row in method['checkpoints']:
            fold = int(row['fold'])
            prefix = resolve_path(row['prefix'])
            validate_hash(
                Path(str(prefix) + '.index'), row['index_sha256'],
                '%s fold %d checkpoint index' % (method['name'], fold))
            validate_hash(
                Path(str(prefix) + '.data-00000-of-00001'),
                row['data_sha256'],
                '%s fold %d checkpoint data' % (method['name'], fold))
            checkpoints[fold] = {'prefix': prefix, **row}
        expected_folds = set(range(1, int(manifest['fold_count']) + 1))
        if set(checkpoints) != expected_folds:
            raise ValueError(
                '%s checkpoint folds do not equal %s.' %
                (method['name'], sorted(expected_folds)))
        neural_methods[method['slug']] = {
            **method,
            'config_path': config_path,
            'checkpoints_by_fold': checkpoints,
        }

    return {
        'manifest_path': manifest_path,
        'manifest_sha256': sha256_file(manifest_path),
        'evidence_path': evidence_path,
        'evidence_rows': evidence_rows,
        'evidence_pairs': evidence_pairs,
        'evidence_targets': evidence_targets(evidence_rows),
        'assignment_path': assignment_path,
        'assignments': assignments,
        'compounds_by_fold': {
            fold: sorted(values) for fold, values in compounds_by_fold.items()
        },
        'graph_paths': graph_paths,
        'protein_ids': sorted(protein_ids),
        'neural_methods': neural_methods,
    }


def evidence_targets(rows):
    targets = defaultdict(set)
    for row in rows:
        targets[str(row['compound_id'])].add(str(row['protein_id']))
    return targets


def neural_output_dir(output_dir, method_slug, fold):
    return output_dir / 'rankings' / method_slug / ('fold_%d' % fold)


def heuristic_output_dir(output_dir, method, fold):
    return output_dir / 'rankings' / method / ('fold_%d' % fold)


def run_neural_inference(manifest, audit, output_dir, resume=False):
    ks = [str(value) for value in manifest['candidate_protocol']['ks']]
    for slug, method in audit['neural_methods'].items():
        for fold in range(1, int(manifest['fold_count']) + 1):
            fold_dir = neural_output_dir(output_dir, slug, fold)
            if resume and (fold_dir / 'report.json').exists():
                print('[resume] %s fold %d' % (method['name'], fold))
                continue
            command = [
                sys.executable,
                str(REPOSITORY_ROOT / 'tools' / 'evaluate_checkpoint_ranking.py'),
                '--config', str(method['config_path']),
                '--checkpoint', str(
                    method['checkpoints_by_fold'][fold]['prefix']),
                '--fold', str(fold),
                '--evaluation-split', 'outer',
                '--ks', *ks,
                '--export-top', '0',
                '--export-compound-ids', *audit['compounds_by_fold'][fold],
                '--output-dir', str(fold_dir),
            ]
            print('[neural] %s fold %d: %d compounds' % (
                method['name'], fold, len(audit['compounds_by_fold'][fold])))
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def run_heuristics(manifest, audit, output_dir, resume=False):
    hc_pairs = read_relation_pairs(audit['graph_paths']['hc'])
    pd_pairs = read_relation_pairs(audit['graph_paths']['pd'])
    extra_proteins = [protein_id for protein_id, _ in pd_pairs]
    methods = manifest['non_neural_methods']['names']
    prior_strength = float(manifest['non_neural_methods']['prior_strength'])
    ks = manifest['candidate_protocol']['ks']
    for fold_index in range(int(manifest['fold_count'])):
        fold = fold_index + 1
        completed = all(
            (heuristic_output_dir(output_dir, method, fold) / 'report.json').exists()
            for method in methods
        )
        if resume and completed:
            print('[resume] non-neural fold %d' % fold)
            continue
        train_records, test_records = fold_records(
            audit['assignments'], fold_index)
        requested = set(audit['compounds_by_fold'][fold])
        test_records = [
            row for row in test_records if str(row[0]) in requested
        ]
        matrices = build_matrices_from_records(
            hc_pairs, train_records, test_records,
            extra_protein_ids=extra_proteins)
        if set(matrices['protein_ids']) != set(audit['protein_ids']):
            raise ValueError(
                'Heuristic fold %d candidate universe does not match 509 proteins.' %
                fold)
        prior, _ = global_target_prior(matrices['cp'])
        prototype, prototype_covered = herb_prototype_profiles(
            matrices['hc'], matrices['cp'], matrices['test_compounds'],
            prior_strength=prior_strength)
        jaccard, jaccard_covered = hc_jaccard_label_propagation(
            matrices['hc'], matrices['cp'], matrices['test_compounds'])
        profiles = {
            'GlobalPrior': np.broadcast_to(
                prior,
                (len(matrices['test_compound_ids']), len(matrices['protein_ids']))),
            'HerbPrototype-EB': prototype,
            'HC-Jaccard-LP': jaccard,
        }
        coverage = {
            'GlobalPrior': 1.0,
            'HerbPrototype-EB': float(np.mean(prototype_covered)),
            'HC-Jaccard-LP': float(np.mean(jaccard_covered)),
        }
        for method in methods:
            ranking = evaluate_fixed_candidate_ranking(
                matrices['protein_ids'], train_records, test_records,
                profile_scorer(
                    profiles[method], matrices['test_compound_ids'],
                    matrices['protein_ids']),
                ks=ks, export_top=0, export_compound_ids=requested)
            fold_dir = heuristic_output_dir(output_dir, method, fold)
            write_tsv(
                fold_dir / 'top_candidates.tsv', ranking['top_candidates'],
                [
                    'compound_id', 'rank', 'protein_id', 'score',
                    'label_status', 'in_sampled_test_negatives',
                ])
            report = {
                'evaluation_type': 'fold_local_non_neural_pure_inference_outer',
                'created_at': datetime.now().astimezone().isoformat(),
                'method': method,
                'fold': fold,
                'training_optimizer_steps': 0,
                'coverage': coverage[method],
                'training_positive_edges': matrices['training_positive_edges'],
                'fixed_candidate_metrics_for_known_targets': ranking['metrics'],
            }
            (fold_dir / 'report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8')
            print('[heuristic] %s fold %d complete' % (method, fold))


def load_method_rankings(manifest, audit, output_dir):
    methods = {
        slug: method['name'] for slug, method in audit['neural_methods'].items()
    }
    methods.update({name: name for name in manifest['non_neural_methods']['names']})
    rankings = {}
    for slug, display_name in methods.items():
        compound_rankings = defaultdict(dict)
        for fold in range(1, int(manifest['fold_count']) + 1):
            if slug in audit['neural_methods']:
                fold_dir = neural_output_dir(output_dir, slug, fold)
                report = json.loads(
                    (fold_dir / 'report.json').read_text(encoding='utf-8'))
                if report.get('evaluation_type') != 'checkpoint_only_pure_inference_outer':
                    raise ValueError('%s fold %d is not pure outer inference.' % (
                        display_name, fold))
                expected_hash = audit['neural_methods'][slug][
                    'checkpoints_by_fold'][fold]['index_sha256']
                if report['checkpoint']['index_sha256'] != expected_hash:
                    raise ValueError('%s fold %d checkpoint mismatch.' % (
                        display_name, fold))
            else:
                fold_dir = heuristic_output_dir(output_dir, slug, fold)
                report = json.loads(
                    (fold_dir / 'report.json').read_text(encoding='utf-8'))
                if report.get('training_optimizer_steps') != 0:
                    raise ValueError('%s unexpectedly performed training.' % display_name)
            for row in read_tsv(fold_dir / 'top_candidates.tsv'):
                compound_id = str(row['compound_id'])
                protein_id = str(row['protein_id'])
                if protein_id in compound_rankings[compound_id]:
                    raise ValueError(
                        'Duplicate ranked pair for %s: %s-%s.' %
                        (display_name, compound_id, protein_id))
                compound_rankings[compound_id][protein_id] = {
                    'rank': int(row['rank']),
                    'score': float(row['score']),
                    'fold': fold,
                }
        expected_compounds = set(audit['evidence_targets'])
        if set(compound_rankings) != expected_compounds:
            raise ValueError(
                '%s ranked compound set does not match frozen evidence.' %
                display_name)
        expected_candidates = int(manifest['model_graph']['expected_proteins'])
        invalid = {
            compound_id: len(values)
            for compound_id, values in compound_rankings.items()
            if len(values) != expected_candidates
        }
        if invalid:
            raise ValueError(
                '%s has incomplete candidate rankings: %s.' %
                (display_name, invalid))
        rankings[display_name] = compound_rankings
    return rankings


def evaluate_confirmed_targets(targets, rankings, ks):
    per_compound = []
    per_pair = []
    for compound_id in sorted(targets):
        candidates = rankings[compound_id]
        candidate_count = len(candidates)
        target_ranks = []
        for protein_id in sorted(targets[compound_id]):
            if protein_id not in candidates:
                raise ValueError(
                    'Confirmed pair %s-%s is absent from ranking.' %
                    (compound_id, protein_id))
            rank = int(candidates[protein_id]['rank'])
            percentile_score = (
                1.0 if candidate_count == 1 else
                1.0 - (rank - 1) / float(candidate_count - 1)
            )
            target_ranks.append(rank)
            per_pair.append({
                'compound_id': compound_id,
                'protein_id': protein_id,
                'fold': candidates[protein_id]['fold'],
                'rank': rank,
                'candidate_count': candidate_count,
                'percentile_score': percentile_score,
                'score': candidates[protein_id]['score'],
            })
        first_rank = min(target_ranks)
        row = {
            'compound_id': compound_id,
            'fold': next(iter(candidates.values()))['fold'],
            'confirmed_target_count': len(target_ranks),
            'candidate_count': candidate_count,
            'first_rank': first_rank,
            'MRR': 1.0 / first_rank,
            'mean_rank': float(np.mean(target_ranks)),
            'median_rank': float(np.median(target_ranks)),
            'mean_percentile_score': float(np.mean([
                1.0 - (rank - 1) / float(max(candidate_count - 1, 1))
                for rank in target_ranks
            ])),
        }
        for k in ks:
            hits = sum(rank <= int(k) for rank in target_ranks)
            recall = hits / float(len(target_ranks))
            random_recall = min(int(k), candidate_count) / float(candidate_count)
            row['hits@%d' % k] = 1.0 if hits else 0.0
            row['recall@%d' % k] = recall
            row['ef@%d' % k] = recall / random_recall
        per_compound.append(row)

    summary = {
        'aggregation': 'macro_over_evidence_compounds_unless_named_micro',
        'compounds': len(per_compound),
        'confirmed_pairs': len(per_pair),
        'MRR': float(np.mean([row['MRR'] for row in per_compound])),
        'mean_pair_rank': float(np.mean([row['rank'] for row in per_pair])),
        'median_pair_rank': float(np.median([row['rank'] for row in per_pair])),
        'mean_percentile_score': float(np.mean([
            row['percentile_score'] for row in per_pair
        ])),
    }
    for k in ks:
        summary['Recall@%d' % k] = float(np.mean([
            row['recall@%d' % k] for row in per_compound
        ]))
        summary['Hits@%d' % k] = float(np.mean([
            row['hits@%d' % k] for row in per_compound
        ]))
        summary['EF@%d' % k] = float(np.mean([
            row['ef@%d' % k] for row in per_compound
        ]))
        summary['micro_Recall@%d' % k] = (
            sum(int(row['rank']) <= int(k) for row in per_pair)
            / float(len(per_pair))
        )
    return {
        'summary': summary,
        'per_compound': per_compound,
        'per_pair': per_pair,
    }


def bootstrap_delta(left_rows, right_rows, metric, iterations, seed):
    per_compound_metric = {
        'MRR': 'MRR',
        **{
            '%s@%d' % (name, k): '%s@%d' % (name.lower(), k)
            for name in ('Recall', 'Hits', 'EF')
            for k in (1, 5, 10, 20, 50)
        },
    }.get(metric, metric)
    left = {
        row['compound_id']: float(row[per_compound_metric])
        for row in left_rows
    }
    right = {
        row['compound_id']: float(row[per_compound_metric])
        for row in right_rows
    }
    if set(left) != set(right):
        raise ValueError('Bootstrap methods use different compound sets.')
    compounds = sorted(left)
    deltas = np.asarray(
        [left[value] - right[value] for value in compounds], dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    samples = np.empty(int(iterations), dtype=np.float64)
    chunk_size = 1000
    for start in range(0, int(iterations), chunk_size):
        stop = min(start + chunk_size, int(iterations))
        indices = rng.randint(
            0, len(deltas), size=(stop - start, len(deltas)))
        samples[start:stop] = np.mean(deltas[indices], axis=1)
    return {
        'metric': metric,
        'unit': 'compound',
        'compounds': len(compounds),
        'mean_delta': float(np.mean(deltas)),
        'median_delta': float(np.median(deltas)),
        'positive_compounds': int(np.sum(deltas > 0)),
        'negative_compounds': int(np.sum(deltas < 0)),
        'ties': int(np.sum(deltas == 0)),
        'bootstrap_iterations': int(iterations),
        'bootstrap_seed': int(seed),
        'bootstrap_95_ci': [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
    }


def render_markdown(report):
    primary_method = report.get('primary_method', 'Final')
    lines = [
        '# ETCM Held-out Confirmed-Target Evaluation',
        '',
        '- Protocol: `%s`' % report['protocol'],
        '- Evidence pairs: `%d`' % report['evidence']['pairs'],
        '- Evidence compounds: `%d`' % report['evidence']['compounds'],
        '- Candidate proteins: `%d`' % report['candidate_proteins'],
        '- Training/optimizer steps during evaluation: `0`',
        '- Model selection reopened: `False`',
        '',
        '| Method | MRR | Recall@10 | Recall@20 | Hits@20 | Micro R@20 | EF@20 | Mean percentile | Median rank |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method, result in report['methods'].items():
        metrics = result['summary']
        lines.append(
            '| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.3f | %.6f | %.1f |' % (
                method,
                metrics['MRR'],
                metrics['Recall@10'],
                metrics['Recall@20'],
                metrics['Hits@20'],
                metrics['micro_Recall@20'],
                metrics['EF@20'],
                metrics['mean_percentile_score'],
                metrics['median_pair_rank'],
            )
        )
    lines.extend([
        '',
        '## Paired compound-level comparisons against %s' % primary_method,
        '',
        '| Comparison | Metric | Mean delta | 95% bootstrap CI | Positive compounds |',
        '|---|---|---:|---:|---:|',
    ])
    for comparison in report['comparisons']:
        low, high = comparison['bootstrap_95_ci']
        lines.append(
            '| %s | %s | %+.6f | [%+.6f, %+.6f] | %d/%d |' % (
                comparison['comparison'], comparison['metric'],
                comparison['mean_delta'], low, high,
                comparison['positive_compounds'], comparison['compounds'],
            )
        )
    lines.extend([
        '',
        '## Interpretation boundary',
        '',
        'These Confirmed Targets were absent from the model C-P graph and were not used for training, early stopping or model selection. They originate from a held-out ETCM2.0 relation source, not an independent prospective cohort. Unrecorded candidate pairs remain unlabelled, and all 685 frozen evidence pairs are reported regardless of rank.',
        '',
    ])
    return '\n'.join(lines)


def summarize(manifest, audit, output_dir):
    rankings = load_method_rankings(manifest, audit, output_dir)
    ks = [int(value) for value in manifest['candidate_protocol']['ks']]
    evaluated = {}
    for method, method_rankings in rankings.items():
        result = evaluate_confirmed_targets(
            audit['evidence_targets'], method_rankings, ks)
        method_dir = output_dir / 'confirmed_metrics' / method
        write_tsv(
            method_dir / 'per_compound.tsv', result['per_compound'],
            list(result['per_compound'][0]))
        write_tsv(
            method_dir / 'confirmed_target_ranks.tsv', result['per_pair'],
            list(result['per_pair'][0]))
        evaluated[method] = result

    primary_method = manifest.get('primary_method', 'Final')
    if primary_method not in evaluated:
        raise ValueError('Primary method is absent from rankings: %s.' % primary_method)
    primary_rows = evaluated[primary_method]['per_compound']
    bootstrap = manifest['bootstrap']
    comparisons = []
    legacy_comparison_methods = {
        'Final_vs_NoContext': 'NoContext',
        'Final_vs_Hctx-P': 'Hctx-P',
        'Final_vs_HerbPrototype-EB': 'HerbPrototype-EB',
        'Final_vs_HC-Jaccard-LP': 'HC-Jaccard-LP',
    }
    for comparison in bootstrap['comparisons']:
        if isinstance(comparison, dict):
            name = comparison['name']
            comparison_method = comparison['method']
        else:
            name = comparison
            comparison_method = legacy_comparison_methods[name]
        if comparison_method not in evaluated:
            raise ValueError(
                'Comparison method is absent from rankings: %s.' %
                comparison_method)
        for metric in bootstrap['metrics']:
            row = bootstrap_delta(
                primary_rows, evaluated[comparison_method]['per_compound'],
                metric, bootstrap['iterations'], bootstrap['seed'])
            comparisons.append({'comparison': name, **row})

    report = {
        'created_at': datetime.now().astimezone().isoformat(),
        'protocol': manifest['protocol'],
        'primary_method': primary_method,
        'manifest': str(audit['manifest_path']),
        'manifest_sha256': audit['manifest_sha256'],
        'training_optimizer_steps': 0,
        'model_selection_reopened': False,
        'evidence': {
            'path': str(audit['evidence_path']),
            'sha256': manifest['evidence']['sha256'],
            'pairs': len(audit['evidence_pairs']),
            'compounds': len(audit['evidence_targets']),
            'fold_compounds': {
                str(fold): len(values)
                for fold, values in audit['compounds_by_fold'].items()
            },
            'fold_pairs': dict(Counter(
                str(next(
                    fold for fold, compounds in audit['compounds_by_fold'].items()
                    if compound_id in compounds
                ))
                for compound_id, _ in audit['evidence_pairs']
            )),
        },
        'candidate_proteins': int(manifest['model_graph']['expected_proteins']),
        'methods': {
            method: {'summary': result['summary']}
            for method, result in evaluated.items()
        },
        'comparisons': comparisons,
        'reporting_boundaries': manifest['reporting_boundaries'],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    (output_dir / 'summary.md').write_text(
        render_markdown(report), encoding='utf-8')
    return report


def print_audit(manifest, audit):
    print('Frozen ETCM Confirmed Target protocol validated.')
    print('  manifest: %s' % audit['manifest_path'])
    print('  manifest SHA-256: %s' % audit['manifest_sha256'])
    print('  evidence pairs: %d' % len(audit['evidence_pairs']))
    print('  evidence compounds: %d' % len(audit['evidence_targets']))
    print('  candidate proteins: %d' % len(audit['protein_ids']))
    for fold in range(1, int(manifest['fold_count']) + 1):
        pair_count = sum(
            len(audit['evidence_targets'][compound_id])
            for compound_id in audit['compounds_by_fold'][fold])
        print('  fold %d: %d compounds, %d confirmed pairs' % (
            fold, len(audit['compounds_by_fold'][fold]), pair_count))
    print('  neural checkpoint evaluations: %d' % (
        len(audit['neural_methods']) * int(manifest['fold_count'])))
    print('  non-neural methods: %s' % ', '.join(
        manifest['non_neural_methods']['names']))
    print('  training/optimizer steps: 0')
    print('  model selection reopened: False')


def main():
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    audit = validate_protocol(manifest, manifest_path)
    print_audit(manifest, audit)
    if args.dry_run:
        return 0

    configured_output = (
        args.output_dir
        or manifest.get('output_dir', 'results/etcm_confirmed_target_evaluation')
    )
    output_dir = resolve_path(configured_output)
    if not args.summarize_only:
        run_neural_inference(manifest, audit, output_dir, resume=args.resume)
        run_heuristics(manifest, audit, output_dir, resume=args.resume)
    report = summarize(manifest, audit, output_dir)
    print(render_markdown(report))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

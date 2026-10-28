#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Restore a supported CTI checkpoint and evaluate it over a fixed full-protein '
            'candidate set without training.'
        )
    )
    parser.add_argument('--config', required=True, help='Configuration used to train the checkpoint.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint prefix or checkpoint directory.')
    parser.add_argument('--fold', type=int, default=1, help='One-based strict outer fold (default: 1).')
    parser.add_argument('--ks', type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument('--export-top', type=int, default=20)
    parser.add_argument(
        '--export-compound-ids', nargs='+',
        help=(
            'Restrict candidate export to these evaluation compounds. With '
            '--export-top 0, export their complete candidate rankings.'
        ),
    )
    parser.add_argument(
        '--negative-ratios', type=int, nargs='+',
        help=(
            'Evaluate deterministic nested unlabeled samples at these '
            'negative-to-positive ratios without training.'
        ),
    )
    parser.add_argument(
        '--negative-ratio-seed', type=int, default=0,
        help='Sampling seed used with --negative-ratios.',
    )
    parser.add_argument('--negative-ratio-universe', choices=['model', 'cp_positive'],
                        default='model', help='Evaluation candidate scope; never changes model evidence.')
    parser.add_argument(
        '--skip-full-candidate', action='store_true',
        help=(
            'Skip complete candidate ranking when only frozen sampled-ratio '
            'metrics are required.'
        ),
    )
    parser.add_argument(
        '--evaluation-split',
        choices=('outer', 'validation'),
        default='outer',
        help=(
            'Evaluate the outer test fold or the inner-validation split. '
            'Validation mode never scores outer-test records.'
        ),
    )
    parser.add_argument('--output-dir', help='Output directory; a timestamped path is used by default.')
    parser.add_argument(
        '--complementarity-hc-path',
        help='H-C relation file used to construct a frozen HerbPrototype-EB prior.',
    )
    parser.add_argument(
        '--complementarity-pd-path',
        help='P-D relation file used only to preserve the full protein universe.',
    )
    parser.add_argument(
        '--complementarity-prior-strength', type=float, default=1.0,
        help='Empirical-Bayes prior strength for the frozen herb prototype.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate and print the split protocol without importing TensorFlow or restoring weights.',
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def records_sha256(records):
    lines = [
        '%s\t%s\t%d' % (str(left), str(right), int(float(label) > 0))
        for left, right, label in records
    ]
    return hashlib.sha256(
        ('\n'.join(sorted(lines)) + '\n').encode('utf-8')
    ).hexdigest()


def normalize_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        preferred = path / 'hdcti_model.ckpt.index'
        if preferred.exists():
            path = preferred
        else:
            index_files = sorted(path.glob('*.index'))
            if len(index_files) != 1:
                raise FileNotFoundError(
                    'Expected exactly one *.index file in %s, found %d.' %
                    (path, len(index_files))
                )
            path = index_files[0]
    text = str(path)
    for suffix in ('.index', '.meta'):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
    if '.data-' in text:
        text = text.split('.data-', 1)[0]
    prefix = Path(text)
    if not Path(str(prefix) + '.index').exists():
        raise FileNotFoundError('Checkpoint index not found: %s.index' % prefix)
    data_files = sorted(prefix.parent.glob(prefix.name + '.data-*'))
    if not data_files:
        raise FileNotFoundError('Checkpoint data shard not found for %s.' % prefix)
    return prefix, data_files


def write_table(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    from util.config import ModelConf, OptionConf
    from util.dataSplit import DataSplit
    from util.model_components import resolve_early_stopping, resolve_negative_sampling

    conf = ModelConf(str(config_path))
    protocol = conf['experiment.protocol'].strip().lower() if conf.contains('experiment.protocol') else 'legacy'
    if protocol != 'strict':
        raise ValueError('Checkpoint ranking evaluation requires experiment.protocol=strict.')
    evaluation = OptionConf(conf['evaluation.setup'])
    if not evaluation.contains('-cv'):
        raise ValueError('Checkpoint ranking evaluation requires evaluation.setup=-cv K.')
    fold_count = int(evaluation['-cv'])
    if args.fold < 1 or args.fold > fold_count:
        raise ValueError('--fold must be between 1 and %d.' % fold_count)

    folds, split_manifest = DataSplit.prepareStrictFolds(
        conf, conf['datapath'], fold_count
    )
    outer_train, outer_test = folds[args.fold - 1]
    early_stopping = resolve_early_stopping(conf)
    validation = []
    validation_info = None
    model_train = outer_train
    if early_stopping['enabled']:
        base_seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
        validation_seed_base = (
            int(conf['validation.seed'])
            if conf.contains('validation.seed') else base_seed + 100000
        )
        model_train, validation, validation_info = DataSplit.innerValidationSplitForConfig(
            conf,
            outer_train,
            early_stopping['ratio'],
            validation_seed_base + args.fold - 1,
        )

    if args.evaluation_split == 'validation':
        if not validation:
            raise ValueError(
                '--evaluation-split validation requires an enabled, non-empty '
                'inner-validation split.'
            )
        ranking_train = model_train
        ranking_test = validation
    else:
        ranking_train = outer_train
        ranking_test = outer_test

    checkpoint_prefix, data_files = normalize_checkpoint(args.checkpoint)
    split_audit = {
        'strict_manifest': split_manifest.get('assignments_path'),
        'strict_assignments_sha256': split_manifest.get('assignments_sha256'),
        'outer_fold_one_based': args.fold,
        'outer_train_records': len(outer_train),
        'outer_test_records': len(outer_test),
        'outer_train_sha256': records_sha256(outer_train),
        'outer_test_sha256': records_sha256(outer_test),
        'model_train_records': len(model_train),
        'model_train_sha256': records_sha256(model_train),
        'validation_records': len(validation),
        'validation_sha256': records_sha256(validation) if validation else None,
        'validation_info': validation_info,
        'training_negative_settings': resolve_negative_sampling(conf),
        'evaluation_split': args.evaluation_split,
        'outer_test_scored': args.evaluation_split == 'outer',
        'ranking_train_records': len(ranking_train),
        'ranking_train_sha256': records_sha256(ranking_train),
        'ranking_test_records': len(ranking_test),
        'ranking_test_sha256': records_sha256(ranking_test),
    }
    complementarity_enabled = bool(args.complementarity_hc_path)
    if args.skip_full_candidate and complementarity_enabled:
        raise ValueError(
            '--skip-full-candidate cannot be combined with complementarity.')
    if complementarity_enabled != bool(args.complementarity_pd_path):
        raise ValueError(
            '--complementarity-hc-path and --complementarity-pd-path must be '
            'provided together.'
        )
    if args.complementarity_prior_strength < 0:
        raise ValueError('--complementarity-prior-strength must be non-negative.')
    split_audit['complementarity_audit'] = {
        'enabled': complementarity_enabled,
        'hc_path': (
            str(Path(args.complementarity_hc_path).expanduser().resolve())
            if complementarity_enabled else None
        ),
        'pd_path': (
            str(Path(args.complementarity_pd_path).expanduser().resolve())
            if complementarity_enabled else None
        ),
        'prior_strength': float(args.complementarity_prior_strength),
        'methods': (
            [
                'neural', 'prior', 'jaccard', 'prior_tie_break',
                'percentile_mean', 'heuristic_percentile_mean',
                'three_way_percentile_mean',
            ]
            if complementarity_enabled else []
        ),
    }
    print('Pure-inference checkpoint evaluation')
    print('  config: %s' % config_path)
    print('  checkpoint: %s' % checkpoint_prefix)
    print('  strict fold: %d/%d' % (args.fold, fold_count))
    print('  model graph records: %d; %s records: %d' % (
        len(model_train), args.evaluation_split, len(ranking_test)
    ))
    print('  outer test scored: %s' % (args.evaluation_split == 'outer'))
    print('  training/optimizer steps: disabled')
    if args.dry_run:
        print(json.dumps(split_audit, ensure_ascii=False, indent=2))
        return 0

    # CUDA visibility must be set before TensorFlow and HDCTI are imported.
    from util.gpu import configure_cuda_environment
    configure_cuda_environment(conf)
    from util.reproducibility import set_global_seed

    base_seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
    set_global_seed(base_seed + args.fold - 1, reset_tensorflow_graph=True)

    import tensorflow.compat.v1 as tf
    from sklearn.metrics import average_precision_score, roc_auc_score
    from util.checkpoint_ranking import (
        assess_pu_evidence,
        evaluate_negative_ratio_sensitivity,
        evaluate_fixed_candidate_ranking,
    )
    from util.score_fusion import (
        fuse_candidate_scores,
        fuse_multiple_candidate_scores,
    )

    model_name = conf['model.name'].strip()
    if model_name == 'HDCTI':
        from HDCTI import HDCTI
        model_class = HDCTI
    elif model_name == 'SACTI':
        from SACTI import SACTI
        model_class = SACTI
    else:
        raise ValueError(
            'Checkpoint ranking does not support model.name=%s.' % model_name
        )

    model = model_class(conf, model_train, outer_test, '[%d]' % args.fold)
    model.validationData = validation
    model.readConfiguration()
    model.initModel()

    checkpoint_variables = dict(tf.train.list_variables(str(checkpoint_prefix)))
    graph_variables = {
        variable.name.split(':', 1)[0]: variable
        for variable in tf.global_variables()
    }
    missing = sorted(set(graph_variables) - set(checkpoint_variables))
    shape_mismatches = []
    for name, variable in graph_variables.items():
        if name not in checkpoint_variables:
            continue
        graph_shape = variable.shape.as_list()
        checkpoint_shape = list(checkpoint_variables[name])
        if graph_shape != checkpoint_shape:
            shape_mismatches.append({
                'name': name,
                'graph': graph_shape,
                'checkpoint': checkpoint_shape,
            })
    if missing or shape_mismatches:
        model.sess.close()
        raise ValueError(
            'Checkpoint/config mismatch: missing graph variables=%s; shape mismatches=%s.' %
            (missing, shape_mismatches)
        )

    saver = tf.train.Saver(var_list=graph_variables)
    saver.restore(model.sess, str(checkpoint_prefix))
    state = model.fetchModelState()
    if model_name == 'HDCTI':
        model.u = state['compound']
        model.i = state['protein']
        model.u_context = state['compound_context']
        model.i_context = state['protein_context']
        model.herb_edge = state['herb_edge']
        model.weight = state['weights']
    else:
        model.state = state
        model.u = state['compound_id']
        model.i = state['protein']
    print('Checkpoint restored; one inference forward pass completed.')

    def score_id_pairs(compound_ids, protein_ids):
        compound_indices = np.fromiter(
            (model.data.compound[str(value)] for value in compound_ids),
            dtype=np.int64,
            count=len(compound_ids),
        )
        protein_indices = np.fromiter(
            (model.data.protein[str(value)] for value in protein_ids),
            dtype=np.int64,
            count=len(protein_ids),
        )
        return model.predictForPairs(compound_indices, protein_indices)

    complementarity = None
    complementarity_scorers = None
    complementarity_rankings = {}
    if complementarity_enabled:
        from tools.evaluate_full_candidate_heuristics import profile_scorer
        from tools.evaluate_non_neural_cold_start_baselines import (
            build_matrices_from_records,
            hc_jaccard_label_propagation,
            herb_prototype_profiles,
            read_relation_pairs,
        )

        hc_path = Path(args.complementarity_hc_path).expanduser().resolve()
        pd_path = Path(args.complementarity_pd_path).expanduser().resolve()
        hc_pairs = read_relation_pairs(hc_path)
        pd_pairs = read_relation_pairs(pd_path)
        matrices = build_matrices_from_records(
            hc_pairs,
            ranking_train,
            ranking_test,
            extra_protein_ids=(
                [protein_id for protein_id, _ in pd_pairs]
                + [
                    str(model.data.id2protein[index])
                    for index in range(model.num_proteins)
                ]
            ),
        )
        prototype, prototype_covered = herb_prototype_profiles(
            matrices['hc'],
            matrices['cp'],
            matrices['test_compounds'],
            prior_strength=float(args.complementarity_prior_strength),
        )
        jaccard, jaccard_covered = hc_jaccard_label_propagation(
            matrices['hc'], matrices['cp'], matrices['test_compounds'])
        model_protein_ids = [
            str(model.data.id2protein[index])
            for index in range(model.num_proteins)
        ]
        missing_model = sorted(
            set(model_protein_ids) - set(matrices['protein_ids']))
        prior_only_proteins = sorted(
            set(matrices['protein_ids']) - set(model_protein_ids))
        if missing_model:
            model.sess.close()
            raise ValueError(
                'The frozen prior is missing %d proteins from the model candidate '
                'universe.' % len(missing_model)
            )
        prior_score_pairs = profile_scorer(
            prototype,
            matrices['test_compound_ids'],
            matrices['protein_ids'],
        )
        jaccard_score_pairs = profile_scorer(
            jaccard,
            matrices['test_compound_ids'],
            matrices['protein_ids'],
        )

        def fused_scorer(method):
            def score_pairs(compound_ids, protein_ids):
                return fuse_candidate_scores(
                    prior_score_pairs(compound_ids, protein_ids),
                    score_id_pairs(compound_ids, protein_ids),
                    method,
                )
            return score_pairs

        def three_way_percentile_scorer(compound_ids, protein_ids):
            return fuse_multiple_candidate_scores([
                prior_score_pairs(compound_ids, protein_ids),
                jaccard_score_pairs(compound_ids, protein_ids),
                score_id_pairs(compound_ids, protein_ids),
            ])

        def heuristic_percentile_scorer(compound_ids, protein_ids):
            return fuse_multiple_candidate_scores([
                prior_score_pairs(compound_ids, protein_ids),
                jaccard_score_pairs(compound_ids, protein_ids),
            ])

        complementarity_scorers = {
            'neural': score_id_pairs,
            'prior': prior_score_pairs,
            'jaccard': jaccard_score_pairs,
            'prior_tie_break': fused_scorer('prior_tie_break'),
            'percentile_mean': fused_scorer('percentile_mean'),
            'heuristic_percentile_mean': heuristic_percentile_scorer,
            'three_way_percentile_mean': three_way_percentile_scorer,
        }
        complementarity = {
            'protocol': 'frozen_inner_candidate_score_complementarity_v1',
            'training_optimizer_steps': 0,
            'outer_test_scored': args.evaluation_split == 'outer',
            'hc_path': str(hc_path),
            'hc_sha256': sha256_file(hc_path),
            'pd_path': str(pd_path),
            'pd_sha256': sha256_file(pd_path),
            'prior_strength': float(args.complementarity_prior_strength),
            'prototype_coverage': float(np.mean(prototype_covered)),
            'jaccard_coverage': float(np.mean(jaccard_covered)),
            'model_candidate_proteins': len(model_protein_ids),
            'prior_proteins': len(matrices['protein_ids']),
            'prior_only_proteins_excluded_from_ranking': len(prior_only_proteins),
            'methods': {},
        }

    try:
        sampled_compounds = [str(row[0]) for row in ranking_test]
        sampled_proteins = [str(row[1]) for row in ranking_test]
        sampled_labels = np.asarray([int(float(row[2]) > 0) for row in ranking_test])
        sampled_logits = np.asarray(
            score_id_pairs(sampled_compounds, sampled_proteins), dtype=np.float64
        )
        sampled_scores = 1.0 / (1.0 + np.exp(-np.clip(sampled_logits, -50, 50)))
        sampled_metrics = {
            'protocol': 'fixed_strict_sampled_%s_pairs' % args.evaluation_split,
            'records': len(ranking_test),
            'positives': int(np.sum(sampled_labels)),
            'negatives': int(len(sampled_labels) - np.sum(sampled_labels)),
            'AUC': float(roc_auc_score(sampled_labels, sampled_scores)),
            'AUPR': float(average_precision_score(sampled_labels, sampled_scores)),
        }

        negative_ratio_sensitivity = None
        if args.negative_ratios:
            from util.ratio_candidate_universe import ratio_candidate_proteins
            ratio_proteins = ratio_candidate_proteins(
                [model.data.id2protein[index] for index in range(model.num_proteins)],
                outer_train, outer_test, args.negative_ratio_universe)
            negative_ratio_sensitivity = evaluate_negative_ratio_sensitivity(
                ratio_proteins,
                ranking_train,
                ranking_test,
                score_id_pairs,
                ratios=args.negative_ratios,
                seed=args.negative_ratio_seed + args.fold - 1,
            )
            negative_ratio_sensitivity['candidate_universe'] = args.negative_ratio_universe
            negative_ratio_sensitivity['candidate_protein_count'] = len(ratio_proteins)

        protein_ids = [
            model.data.id2protein[index] for index in range(model.num_proteins)
        ]
        ranking = None
        pu_assessment = None
        if not args.skip_full_candidate:
            ranking = evaluate_fixed_candidate_ranking(
                protein_ids,
                ranking_train,
                ranking_test,
                score_id_pairs,
                ks=args.ks,
                export_top=args.export_top,
                export_compound_ids=args.export_compound_ids,
            )
            pu_assessment = assess_pu_evidence(
                ranking['metrics'], sampled_metrics=sampled_metrics
            )
        if complementarity_scorers is not None:
            for method, scorer in complementarity_scorers.items():
                method_ranking = evaluate_fixed_candidate_ranking(
                    protein_ids,
                    ranking_train,
                    ranking_test,
                    scorer,
                    ks=args.ks,
                    export_top=args.export_top,
                    export_compound_ids=args.export_compound_ids,
                )
                complementarity['methods'][method] = {
                    'metrics': method_ranking['metrics'],
                }
                complementarity_rankings[method] = method_ranking
    finally:
        model.sess.close()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else
        REPOSITORY_ROOT / 'results' / 'checkpoint_ranking' / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'evaluation_type': (
            'checkpoint_only_pure_inference_%s' % args.evaluation_split
        ),
        'training_optimizer_steps': 0,
        'created_at': datetime.now().astimezone().isoformat(),
        'config': {
            'path': str(config_path),
            'sha256': sha256_file(config_path),
            'model_name': model_name,
            'model_variant': conf['model.variant'] if conf.contains('model.variant') else conf['model.name'],
            'pair_decoder': conf['pair.decoder'] if conf.contains('pair.decoder') else 'dot',
        },
        'checkpoint': {
            'prefix': str(checkpoint_prefix),
            'index_sha256': sha256_file(str(checkpoint_prefix) + '.index'),
            'data_shards': {path.name: sha256_file(path) for path in data_files},
            'restored_graph_variables': len(graph_variables),
            'unused_checkpoint_variables': len(set(checkpoint_variables) - set(graph_variables)),
        },
        'split': split_audit,
        'entity_universe': {
            'compounds': model.num_compounds,
            'proteins': model.num_proteins,
        },
        'sampled_pair_metrics_for_comparison_only': sampled_metrics,
        'negative_ratio_sensitivity': negative_ratio_sensitivity,
        'fixed_candidate_protocol': ranking['protocol'] if ranking else None,
        'fixed_candidate_metrics': ranking['metrics'] if ranking else None,
        'pu_assessment': pu_assessment,
        'score_complementarity': complementarity,
    }
    (output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    per_compound_fields = [
        'compound_id', 'candidate_count', 'filtered_train_positives',
        'test_positive_count', 'first_positive_rank', 'reciprocal_rank',
    ]
    for k in sorted(set(args.ks)):
        per_compound_fields.extend([
            'precision@%d' % k, 'recall@%d' % k, 'hits@%d' % k,
        ])
    if ranking is not None:
        write_table(
            output_dir / 'per_compound_metrics.tsv',
            ranking['per_compound'],
            per_compound_fields,
        )
        write_table(
            output_dir / 'top_candidates.tsv',
            ranking['top_candidates'],
            [
                'compound_id', 'rank', 'protein_id', 'score', 'label_status',
                'in_sampled_test_negatives',
            ],
        )
    for method, method_ranking in complementarity_rankings.items():
        method_dir = output_dir / 'complementarity' / method
        write_table(
            method_dir / 'per_compound_metrics.tsv',
            method_ranking['per_compound'],
            per_compound_fields,
        )
        write_table(
            method_dir / 'top_candidates.tsv',
            method_ranking['top_candidates'],
            [
                'compound_id', 'rank', 'protein_id', 'score',
                'label_status', 'in_sampled_test_negatives',
            ],
        )

    if ranking is not None:
        print('\nFixed-candidate ranking (macro over compounds):')
        for key, value in ranking['metrics'].items():
            if key == 'aggregation' or isinstance(value, dict):
                continue
            if isinstance(value, float):
                print('  %s: %.6f' % (key, value))
            else:
                print('  %s: %s' % (key, value))
        print('\nPU assessment: %s' % pu_assessment['trial_priority'])
        print('  %s' % pu_assessment['rationale'])
        print('  Necessity verdict: %s' % pu_assessment['necessity_conclusion'])
    else:
        print('\nFixed-candidate ranking skipped; frozen E8 metrics remain authoritative.')
    if complementarity is not None:
        print('\nFrozen score complementarity (no optimizer steps):')
        for method, values in complementarity['methods'].items():
            metrics = values['metrics']
            print('  %s: MRR=%.6f Recall@20=%.6f' % (
                method, metrics['MRR'], metrics.get('Recall@20', float('nan'))
            ))
    print('Results written to: %s' % output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

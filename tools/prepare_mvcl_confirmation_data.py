#!/usr/bin/env python3
"""Audit frozen five-fold partitions or export one fold; no model or metric execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse
from util.config import ModelConf
from util.dataSplit import DataSplit
from util.mvcl_cti_data import build_mvcl_data
from util.mvcl_features import fixed_node_features
from tools.experiment_io import relation_matrix
from tools.evaluate_non_neural_cold_start_baselines import ordered_ids, read_relation_pairs
from tools.evaluate_checkpoint_ranking import records_sha256
from tools.summarize_mvcl_sacti_matched import read, digest, require


def check_partitions(train, validation, test):
    groups = []
    for records in (train, validation, test):
        require(records and {float(r[2]) for r in records} == {0., 1.}, 'Each split needs both classes.')
        require(len({(str(c), str(p)) for c, p, _ in records}) == len(records), 'Duplicate pair.')
        groups.append({str(r[0]) for r in records})
    require(not any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))),
            'Train/validation/test compound overlap.')
    return dict(train_sha256=records_sha256(train), validation_sha256=records_sha256(validation),
                outer_test_sha256=records_sha256(test),
                record_counts=[len(x) for x in (train, validation, test)],
                compound_counts=[len(x) for x in groups])


def export_bundle(out, ids, hc, pd, train, validation, test, report):
    data = build_mvcl_data(ids, relation_matrix(hc, ids['herb'], ids['compound']),
                          relation_matrix(pd, ids['protein'], ids['disease']), train,
                          [r[0] for r in validation], [r[0] for r in test],
                          max_nnz=5000000, allow_factorized=True)
    ci = {c: i for i, c in enumerate(ids['compound'])}
    held = [ci[c] for c in {str(r[0]) for r in validation + test}]
    require(data['relations']['C_P'][held].nnz == data['train_mask'][held].nnz == 0,
            'Held-out row present in training graph/mask.')
    rr, cc = data['relations']['C_P'].nonzero()
    actual = {(ids['compound'][c], ids['protein'][p]) for c, p in zip(rr, cc)}
    expected = {(str(c), str(p)) for c, p, y in train if float(y) == 1}
    require(actual == expected, 'Training graph differs from exact positive pairs.')
    features, meta = fixed_node_features(ids)
    data['report'].update(report, features=meta, feature_policy=meta['policy'],
                          outer_test_scored=False, exact_train_positive_match=True,
                          held_out_training_mask_entries=0,
                          scope='formal matched data bundle; no training or metrics')
    for name, matrix in {**data['relations'], **data['metapaths'], 'train_mask': data['train_mask'],
                        **{k+'_factor': m for k, m in data['metapath_factors'].items()}}.items():
        sparse.save_npz(out/(name+'.npz'), matrix)
    np.savez_compressed(out/'features.npz', **features)
    for filename, value in [('ids.json', ids), ('summary.json', data['report']),
                            ('validation.json', validation), ('outer_test.json', test)]:
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')
    return data['report']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', choices=['tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10'])
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--prepare', action='store_true', help='Export exactly one specified dataset/fold.')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from repository root.')
    require(not args.output.exists(), 'Use a new output directory.')
    require(not args.prepare or (args.dataset and args.fold), 'Preparation requires one dataset and fold.')
    seal_path = ROOT/'results/mvcl_formal_confirmation_protocol_v1/seal.json'
    seal = read(seal_path)
    for filename, expected in seal['source_hashes'].items():
        require(digest(filename) == expected, 'Frozen protocol source changed: '+filename)
    policy = seal['protocol']
    reference = read(ROOT/policy['matched_sacti_protocol'])
    rows = []
    for d in reference['datasets']:
        if args.dataset and d['slug'] != args.dataset:
            continue
        config = ROOT/d['candidate_config']
        conf = ModelConf(str(config))
        conf.config.update({'split.seed': str(policy['split_seed']), 'split.dir': d['split_dir'],
                            'split.reuse': 'True', 'evaluation.outer.test': 'False'})
        require(float(conf['validation.ratio']) == policy['validation_ratio'], 'Validation ratio changed.')
        directory = ROOT/d['split_dir']
        require((directory/'manifest.json').is_file() and (directory/'fold_assignments.tsv').is_file(),
                'Refuse to create missing split.')
        folds, manifest = DataSplit.prepareStrictFolds(conf, conf['datapath'], 5)
        sources = {str(p): digest(p) for p in (seal_path, config, directory/'manifest.json', directory/'fold_assignments.tsv')}
        for relation in ('H_C', 'C_P', 'P_D'):
            item = manifest['sources']['side_relations'][relation]
            require(digest(item['path']) == item['sha256'], 'Side input changed.')
            sources[item['path']] = item['sha256']
        for fold in policy['folds']:
            if args.fold and fold != args.fold:
                continue
            outer_train, test = folds[fold-1]
            validation_seed = policy['validation_seed'] + fold - 1
            train, validation, _ = DataSplit.innerValidationSplitForConfig(
                conf, outer_train, policy['validation_ratio'], validation_seed)
            info = check_partitions(train, validation, test)
            historical = ROOT/'results/batch_runs/sacti_outer_lockbox_seed901026/ranking'/d['slug']/'candidate'/('fold_%d'%fold)/'report.json'
            # Only partition metadata is used; historical predictive metrics are not consumed.
            split = read(historical)['split']
            for key, old in [('train_sha256', 'model_train_sha256'), ('validation_sha256', 'validation_sha256'),
                             ('outer_test_sha256', 'outer_test_sha256')]:
                require(info[key] == split[old], 'Historical SACTI fold mismatch: '+d['slug']+' '+str(fold))
            require(records_sha256(outer_train) == split['outer_train_sha256'], 'Outer training mismatch.')
            sources[str(historical)] = digest(historical)
            info.update(dataset=d['slug'], fold=fold, split_seed=policy['split_seed'],
                        validation_seed=validation_seed, training_seed=policy['training_seed']+fold-1,
                        sacti_partition_hashes_match=True, outer_test_scored=False)
            rows.append(info)
            if args.prepare:
                side = manifest['sources']['side_relations']
                hc, pd, full_cp = [read_relation_pairs(side[k]['path']) for k in ('H_C', 'P_D', 'C_P')]
                records = outer_train + test
                ids = dict(herb=ordered_ids(h for h, _ in hc), disease=ordered_ids(v for _, v in pd),
                    compound=ordered_ids([c for _, c in hc]+[c for c, _ in full_cp]+[str(r[0]) for r in records]),
                    protein=ordered_ids([p for p, _ in pd]+[p for _, p in full_cp]+[str(r[1]) for r in records]))
                for f in (Path(__file__), ROOT/'rating.py', ROOT/'util/dataSplit.py', ROOT/'tools/experiment_io.py'):
                    sources[str(f)] = digest(f)
                args.output.mkdir(parents=True, exist_ok=False)
                (args.output/'preparation_state.json').write_text('{"complete": false}\n')
                export_bundle(args.output, ids, hc, pd, train, validation, test, dict(info, source_hashes=dict(sources)))
                require(all(digest(p) == h for p, h in sources.items()), 'Source changed during export.')
                (args.output/'preparation_state.json').write_text('{"complete": true}\n')
        require(all(digest(p) == h for p, h in sources.items()), 'Source changed during partition audit.')
    if not args.prepare:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'summary.json').write_text(json.dumps(dict(folds=rows, graph_bundles_prepared=False,
            optimizer_steps=0, outer_test_scored=False, protocol_seal_sha256=digest(seal_path)), indent=2)+'\n')
    lines = ['# MVCL Confirmation Data Checks', '',
             '| Dataset | Fold | Validation seed | Train/validation/test records | SACTI hashes |',
             '|---|---:|---:|---|---|']
    lines += ['| %s | %d | %d | %s | PASS |' % (r['dataset'],r['fold'],r['validation_seed'],r['record_counts']) for r in rows]
    lines += ['', 'No training, model forward or metric calculation.',
              'Outer records are accessed only for split isolation, identity checks and later evaluation export.',
              'Graph bundle exported: '+str(args.prepare),
              'Original first-fold pilot loader is not a multi-fold formal training entry point.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Summary:', args.output/'summary.md')


if __name__ == '__main__':
    main()

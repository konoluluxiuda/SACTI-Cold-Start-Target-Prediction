#!/usr/bin/env python3
"""Export a frozen first-fold MVCL-CTI graph without training or scoring."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse

from tools.prepare_query_path_data import require_existing_split
from tools.evaluate_checkpoint_ranking import records_sha256, sha256_file
from tools.evaluate_non_neural_cold_start_baselines import ordered_ids, read_relation_pairs
from util.config import ModelConf
from util.dataSplit import DataSplit
from util.mvcl_cti_data import build_mvcl_data
from util.mvcl_features import fixed_node_features


def relation_matrix(pairs, left, right):
    li, ri = {x: i for i, x in enumerate(left)}, {x: i for i, x in enumerate(right)}
    return sparse.csr_matrix(([1.] * len(pairs),
                             ([li[str(a)] for a, _ in pairs], [ri[str(b)] for _, b in pairs])),
                            shape=(len(left), len(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/SACTI_tcmsuite_inner_pilot.conf')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-path-nnz', type=int, default=5000000)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from repository root.')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new output directory.')
    config = Path(args.config).resolve()
    manifest, sources = require_existing_split(config)
    conf = ModelConf(str(config))
    folds, _ = DataSplit.prepareStrictFolds(conf, conf['datapath'], 5)
    train, validation, _ = DataSplit.innerValidationSplitForConfig(
        conf, folds[0][0], float(conf['validation.ratio']), int(conf['validation.seed']))
    source_relations = manifest['sources']['side_relations']
    hc = read_relation_pairs(source_relations['H_C']['path'])
    pd = read_relation_pairs(source_relations['P_D']['path'])
    # Full C-P is used only to preserve the established entity universe.
    full_cp = read_relation_pairs(source_relations['C_P']['path'])
    records = folds[0][0] + folds[0][1]
    ids = dict(herb=ordered_ids(h for h, _ in hc),
               compound=ordered_ids([c for _, c in hc] + [c for c, _ in full_cp]
                                    + [str(r[0]) for r in records]),
               protein=ordered_ids([p for p, _ in pd] + [p for _, p in full_cp]
                                   + [str(r[1]) for r in records]),
               disease=ordered_ids(d for _, d in pd))
    data = build_mvcl_data(ids, relation_matrix(hc, ids['herb'], ids['compound']),
                           relation_matrix(pd, ids['protein'], ids['disease']), train,
                           [r[0] for r in validation], [r[0] for r in folds[0][1]],
                           max_nnz=args.max_path_nnz, allow_factorized=True)
    for path, digest in sources.items():
        if sha256_file(path) != digest:
            raise ValueError('Input changed during preparation: ' + path)
    report = data['report']
    features, feature_metadata = fixed_node_features(ids)
    report.update(feature_policy=feature_metadata['policy'], features=feature_metadata)
    report.update(source_hashes=sources, train_sha256=records_sha256(train),
                  validation_sha256=records_sha256(validation), fold=1,
                  upstream_commit='9851641f1c82d12d0e35aa2403505e3f51f92651',
                  scope='data adapter only; not an implemented or evaluated MVCL model')
    out.mkdir(parents=True, exist_ok=False)
    for name, matrix in {**data['relations'], **data['metapaths'], 'train_mask': data['train_mask'],
                        **{k + '_factor': v for k, v in data['metapath_factors'].items()}}.items():
        sparse.save_npz(out / (name + '.npz'), matrix)
    np.savez_compressed(out / 'features.npz', **features)
    (out / 'validation.json').write_text(json.dumps(
        [[str(c), str(p), int(float(y) > 0)] for c, p, y in validation], indent=2) + '\n')
    (out / 'ids.json').write_text(json.dumps(data['ids'], indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Data-only summary: ' + str(out / 'summary.json'))
    if report['deferred_metapaths']:
        print('Factorized paths (exact factors stored): '
              + ', '.join(report['deferred_metapaths']))


if __name__ == '__main__':
    main()

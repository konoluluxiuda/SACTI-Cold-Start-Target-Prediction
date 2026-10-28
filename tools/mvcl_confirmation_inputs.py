"""Strict multi-fold bundle loading, separate from the historical first-fold pilot."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from util.mvcl_cti_model import RELATIONS
from util.mvcl_cti_data import matrix_hash, NODE_TYPES
from util.mvcl_features import fixed_node_features
from tools.prepare_mvcl_confirmation_data import check_partitions
from tools.summarize_mvcl_sacti_matched import read, require, digest


def load_confirmation_bundle(directory, dataset, fold):
    directory = Path(directory)
    require(read(directory/'preparation_state.json').get('complete') is True, 'Incomplete data preparation.')
    report, ids = read(directory/'summary.json'), read(directory/'ids.json')
    require(fold in range(1, 6) and report['fold'] == fold and report['dataset'] == dataset,
            'Wrong dataset/fold.')
    require(report['split_seed'] == 901026 and report['validation_seed'] == 152026+fold-1
            and report['training_seed'] == 52026+fold-1, 'Frozen seed mismatch.')
    require(not report['outer_test_scored'] and report['sacti_partition_hashes_match'], 'Invalid data scope.')
    for filename, expected in report['source_hashes'].items():
        require(digest(filename) == expected, 'Data source changed: '+filename)
    require(set(ids) == set(NODE_TYPES), 'Wrong node types.')
    for values in ids.values():
        require(values and all(isinstance(x, str) for x in values) and len(values) == len(set(values)),
                'Invalid or duplicate IDs.')
    require(hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest() == report['ids_sha256'],
            'Entity identity mismatch.')
    require(hashlib.sha256(json.dumps(ids['protein']).encode()).hexdigest() == report['candidate_ids_sha256'],
            'Candidate identity mismatch.')
    relations = {r: sparse.load_npz(directory/(r+'.npz')).tocsr() for r in RELATIONS}
    mask = sparse.load_npz(directory/'train_mask.npz').tocsr()
    for name, matrix in dict(relations, train_mask=mask).items():
        require(matrix.has_canonical_format and np.all(matrix.data == 1), 'Non-binary/noncanonical matrix: '+name)
        require(matrix_hash(matrix) == report['matrices'][name]['sha256'], 'Matrix identity mismatch: '+name)
    for name, (a, b) in RELATIONS.items():
        require(relations[name].shape == (len(ids[a]), len(ids[b])), 'Matrix shape mismatch.')
    for a, b in [('H_C', 'C_H'), ('C_P', 'P_C'), ('P_D', 'D_P')]:
        require((relations[a] != relations[b].T).nnz == 0, 'Reverse relation mismatch.')
    cp = relations['C_P']
    require(mask.shape == cp.shape and (cp-cp.multiply(mask)).nnz == 0, 'Positive edge outside mask.')
    features, meta = fixed_node_features(ids)
    require(meta == report['features'], 'Feature metadata mismatch.')
    with np.load(directory/'features.npz', allow_pickle=False) as stored:
        require(set(stored.files) == set(features) and all(np.array_equal(stored[k], v) for k,v in features.items()),
                'Fixed node features changed.')
    rr, cc = mask.nonzero()
    train = [[ids['compound'][c], ids['protein'][p], int(cp[c,p])] for c,p in zip(rr,cc)]
    validation, outer = read(directory/'validation.json'), read(directory/'outer_test.json')
    checked = check_partitions(train, validation, outer)
    for key, value in checked.items():
        require(value == report[key], 'Partition identity mismatch: '+key)
    ci = {v:i for i,v in enumerate(ids['compound'])}
    pi = {v:i for i,v in enumerate(ids['protein'])}
    require(all(c in ci and p in pi for c,p,_ in validation+outer), 'Held-out entity outside universe.')
    held = [ci[c] for c in {r[0] for r in validation+outer}]
    require(mask[held].nnz == 0 and cp[held].nnz == 0, 'Held-out compound has training support.')
    pairs = np.asarray([(ci[c],pi[p]) for c,p,_ in validation], dtype=np.int64)
    labels = np.asarray([int(y) for _,_,y in validation], dtype=np.int64)
    # Outer labels are read only for integrity/isolation; never returned to training.
    return dict(ids=ids, relations=relations, train_mask=mask, report=report), pairs, labels

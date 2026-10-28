"""Relation-only MVCL input contract; no model fitting or held-out labels."""

import hashlib
import json

import numpy as np
from scipy import sparse

NODE_TYPES = ('herb', 'compound', 'protein', 'disease')


class PathBudgetExceeded(ValueError):
    pass


def matrix_hash(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.sort_indices()
    digest = hashlib.sha256(str(matrix.shape).encode())
    for values in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.asarray(values, dtype='<i8').tobytes())
    return digest.hexdigest()


def binary_matrix(matrix, shape):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if matrix.shape != shape or not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('Invalid relation matrix.')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data[:] = 1
    matrix.sort_indices()
    return matrix


def shared_neighbor_path(incidence, block_size=128, max_nnz=5000000):
    """Boolean two-hop reachability, including supported diagonal entries."""
    if block_size < 1 or max_nnz < 0:
        raise ValueError('Invalid construction budget.')
    blocks, count = [], 0
    for start in range(0, incidence.shape[0], block_size):
        block = (incidence[start:start + block_size] @ incidence.T).tocsr()
        block.eliminate_zeros()
        block.data[:] = 1
        count += block.nnz
        if count > max_nnz:
            raise PathBudgetExceeded('Metapath nnz exceeds %d (observed at least %d); no truncation performed.'
                                     % (max_nnz, count))
        blocks.append(block)
    return sparse.vstack(blocks, format='csr') if blocks else sparse.csr_matrix((0, 0))


def build_mvcl_data(ids, hc, pd, train, validation_compounds=(), outer_compounds=(),
                    block_size=128, max_nnz=5000000, allow_factorized=False):
    ids = {kind: [str(x) for x in ids[kind]] for kind in NODE_TYPES}
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise ValueError('Duplicate IDs within a node type.')
    index = {kind: {value: i for i, value in enumerate(ids[kind])} for kind in NODE_TYPES}
    validation, outer = set(map(str, validation_compounds)), set(map(str, outer_compounds))
    if validation & outer or not (validation | outer) <= set(ids['compound']):
        raise ValueError('Invalid held-out compound partition.')
    rows, cols, labels, seen = [], [], [], {}
    for c, p, label in train:
        c, p, label = str(c), str(p), float(label)
        if c in validation | outer:
            raise ValueError('Training compound overlaps held-out partition.')
        if c not in index['compound'] or p not in index['protein'] or label not in (0, 1):
            raise ValueError('Invalid training pair.')
        if (c, p) in seen:
            if seen[c, p] != label:
                raise ValueError('Conflicting pair labels.')
            continue
        seen[c, p] = label
        rows.append(index['compound'][c])
        cols.append(index['protein'][p])
        labels.append(label)
    shape = (len(ids['compound']), len(ids['protein']))
    cp = sparse.csr_matrix((labels, (rows, cols)), shape=shape)
    cp.eliminate_zeros()
    mask = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    hc = binary_matrix(hc, (len(ids['herb']), shape[0]))
    pd = binary_matrix(pd, (shape[1], len(ids['disease'])))
    relations = dict(H_C=hc, C_H=hc.T.tocsr(), C_P=cp, P_C=cp.T.tocsr(),
                     P_D=pd, D_P=pd.T.tocsr())
    paths, factors, deferred = {}, {}, {}
    for name, value in dict(CHC=hc.T.tocsr(), CPC=cp, PDP=pd, PCP=cp.T.tocsr()).items():
        try:
            paths[name] = shared_neighbor_path(value, block_size, max_nnz)
        except PathBudgetExceeded as error:
            if not allow_factorized:
                raise PathBudgetExceeded(name + ': ' + str(error)) from error
            factors[name] = value
            deferred[name] = str(error)
    offsets, offset = {}, 0
    for kind in NODE_TYPES:
        offsets[kind] = offset
        offset += len(ids[kind])
    report = dict(node_counts={k: len(v) for k, v in ids.items()}, offsets=offsets,
                  node_order=list(NODE_TYPES), feature_policy='not_generated_pending_pilot_protocol',
                  optimizer_steps=0, outer_test_scored=False,
                  deferred_metapaths=deferred,
                  factorized_semantics='A = (B @ B.T > 0); counts are NOT equivalent to Boolean reachability',
                  ids_sha256=hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
                  candidate_ids_sha256=hashlib.sha256(json.dumps(ids['protein']).encode()).hexdigest(),
                  matrices={name: dict(shape=list(m.shape), nnz=m.nnz, sha256=matrix_hash(m))
                            for name, m in {**relations, **paths, 'train_mask': mask,
                                            **{k + '_factor': v for k, v in factors.items()}}.items()})
    return dict(ids=ids, relations=relations, metapaths=paths, metapath_factors=factors,
                train_mask=mask, report=report)

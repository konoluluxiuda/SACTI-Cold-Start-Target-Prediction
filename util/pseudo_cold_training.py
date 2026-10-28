"""Training-only query blocks; no access to validation labels or model weights."""

import numpy as np
import scipy.sparse as sp

from util.dual_granularity_transfer import build_loco_jaccard_transfer
from util.herb_prototype_transfer import build_support_calibrated_herb_prototypes


def query_blocks(cp, count, seed):
    if count < 2:
        raise ValueError('Pseudo-cold block count must be at least two.')
    supported = np.flatnonzero(np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0)
    if supported.size < count:
        raise ValueError('Pseudo-cold blocks require at least one supported compound each.')
    return np.array_split(np.random.RandomState(seed).permutation(supported), count)


def masked_transfer_state(hc, cp, queries):
    queries = np.asarray(queries)
    if (queries.ndim != 1 or queries.dtype.kind not in 'iu'
            or np.any(queries < 0) or np.any(queries >= cp.shape[0])
            or np.unique(queries).size != queries.size):
        raise ValueError('Query indices must be unique, valid integer indices.')
    keep = np.ones(cp.shape[0], dtype=np.float32)
    keep[queries] = 0
    masked = sp.diags(keep).dot(cp).tocsr()
    masked.eliminate_zeros()
    return (
        build_support_calibrated_herb_prototypes(hc, masked),
        build_loco_jaccard_transfer(hc, masked),
    )


def paired_support_views(hc, training_cp, queries):
    """Prepare two evidence views and shared labels; not a training routine.

    The caller must supply only inner-training positives. Query indices refer
    to its compound rows. Labels are supervision, never masked-view evidence.
    """
    hc = sp.csr_matrix(hc, dtype=np.float32).copy()
    cp = sp.csr_matrix(training_cp, dtype=np.float32).copy()
    for matrix in (hc, cp):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data != 1):
            raise ValueError('Support views require binary membership/positive matrices.')
    if hc.shape[1] != cp.shape[0] or cp.shape[1] == 0:
        raise ValueError('Support view dimensions disagree or there are no targets.')
    queries = np.asarray(queries)
    if (queries.ndim != 1 or queries.size == 0 or queries.dtype.kind not in 'iu'
            or np.any(queries < 0) or np.any(queries >= cp.shape[0])
            or np.unique(queries).size != queries.size):
        raise ValueError('Queries must be nonempty, unique valid integer indices.')
    labels = cp[queries].copy()
    if np.any(labels.getnnz(axis=1) == 0):
        raise ValueError('Each training query must have at least one positive.')

    # Rebuild every masked statistic before exposing either view to a decoder.
    masked = masked_transfer_state(hc, cp, queries)
    supported = (
        build_support_calibrated_herb_prototypes(hc, cp),
        build_loco_jaccard_transfer(hc, cp),
    )
    return {
        'queries': queries.astype(np.int64, copy=True),
        'positive_labels': labels,
        'supported': {
            'transfer_state': supported,
            'identity_gate': supported[0]['compound_supported'][queries].copy(),
        },
        'unsupported': {
            'transfer_state': masked,
            'identity_gate': masked[0]['compound_supported'][queries].copy(),
        },
    }

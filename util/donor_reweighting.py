"""Sparse donor evidence primitives; no checkpoint or validation-label inputs."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


def binary_matrix(value):
    result = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result.data)) or np.any(result.data < 0):
        raise ValueError('Relations must be finite and nonnegative.')
    result.sum_duplicates()
    result.eliminate_zeros()
    result.data[:] = 1
    result.sort_indices()
    return result


def indices(value, size):
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError('Expected integer row indices.')
    result = raw.astype(np.int64)
    if np.any(result < 0) or np.any(result >= size):
        raise ValueError('Compound index out of bounds.')
    return np.unique(result)


@dataclass
class DonorQuery:
    donors: np.ndarray
    jaccard: np.ndarray
    features: np.ndarray
    labels: sparse.csr_matrix
    prior: np.ndarray

    def score(self, offsets=None):
        """Offsets are query-donor logits, shared over all target candidates."""
        offsets = np.zeros(len(self.donors)) if offsets is None else np.asarray(offsets)
        if offsets.shape != self.jaccard.shape or not np.all(np.isfinite(offsets)):
            raise ValueError('One finite offset is required per donor.')
        if not len(self.donors):
            return self.prior.copy(), np.empty(0)
        logits = np.log(self.jaccard) + offsets
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
        return np.asarray(self.labels.T.dot(weights)).reshape(-1), weights


class DonorEvidence:
    """One masked training block. Rebuild for every different masked block.

    H-C is herbs x compounds; C-P contains ONLY inner-training positives.
    Masked labels are removed before support and fallback prior are computed.
    """

    feature_names = ('jaccard', 'query_overlap_fraction', 'donor_overlap_fraction')

    def __init__(self, hc, cp, masked_compounds):
        self.hc = binary_matrix(hc)
        cp = binary_matrix(cp)
        if self.hc.shape[1] != cp.shape[0] or cp.shape[1] == 0:
            raise ValueError('Incompatible relation dimensions.')
        masked = indices(masked_compounds, cp.shape[0])
        keep = np.ones(cp.shape[0])
        keep[masked] = 0
        self.cp = sparse.diags(keep).dot(cp).tocsr()
        self.cp.eliminate_zeros()
        self.supported = np.diff(self.cp.indptr) > 0
        count = int(self.supported.sum())
        self.prior = np.asarray(self.cp.sum(axis=0)).reshape(-1) / max(count, 1)
        self.memberships = self.hc.T.tocsr()
        self.degree = np.diff(self.memberships.indptr)

    def query(self, compound):
        indices([compound], self.cp.shape[0])
        if self.supported[compound]:
            raise ValueError('Query must be cold or explicitly masked as a whole block.')
        overlap = self.memberships.dot(self.memberships[compound].T).tocoo()
        valid = self.supported[overlap.row] & (overlap.data > 0)
        donors, intersection = overlap.row[valid], overlap.data[valid]
        order = np.argsort(donors)
        donors, intersection = donors[order], intersection[order]
        union = self.degree[compound] + self.degree[donors] - intersection
        jaccard = intersection / union
        features = np.column_stack((jaccard,
            intersection / max(self.degree[compound], 1),
            intersection / np.maximum(self.degree[donors], 1)))
        return DonorQuery(donors, jaccard, features, self.cp[donors], self.prior.copy())

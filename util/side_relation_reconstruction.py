"""Masked H-C inputs and paired sampling for auxiliary reconstruction."""

import numpy as np
from scipy import sparse

from util.donor_reweighting import binary_matrix, indices


class HerbReconstructionSampler:
    """Balanced pairs sharing one masked context, restricted to training entities."""

    def __init__(self, hc, training_compounds, seed=152026):
        self.memberships = binary_matrix(hc).T.tocsr()
        training = indices(training_compounds, self.memberships.shape[0])
        self.num_herbs = self.memberships.shape[1]
        degree = np.diff(self.memberships.indptr)[training]
        eligible = (degree >= 2) & (degree < self.num_herbs)
        self.eligible = training[eligible]
        self.rng = np.random.RandomState(seed)
        self.metadata = dict(training_compounds=len(training),
                             eligible_compounds=len(self.eligible),
                             insufficient_context=int(np.sum(degree < 2)),
                             no_nonmember=int(np.sum((degree >= 2) & (degree == self.num_herbs))),
                             seed=int(seed), drawn_pairs=0)

    def sample(self, count):
        if not isinstance(count, int) or count <= 0:
            raise ValueError('Positive integer sample count required.')
        if not len(self.eligible):
            raise ValueError('No training compounds have both masked context and nonmember herbs.')
        compounds = self.rng.choice(self.eligible, size=count, replace=True)
        positives, negatives = [], []
        for c in compounds:
            start, stop = self.memberships.indptr[c:c + 2]
            herbs = self.memberships.indices[start:stop]
            positives.append(int(self.rng.choice(herbs)))
            # Bounded uniform sampling of the complement, without rejection loops.
            rank = self.rng.randint(self.num_herbs - len(herbs))
            negative = rank + np.searchsorted(herbs - np.arange(len(herbs)), rank, side='right')
            negatives.append(int(negative))
        self.metadata['drawn_pairs'] += 2 * count
        return (np.repeat(compounds, 2),
                np.column_stack((positives, negatives)).ravel(),
                np.tile([1., 0.], count).astype(np.float32),
                np.repeat(positives, 2))


def leave_one_herb_out_memberships(hc, compounds, herbs):
    """Return pair-specific mean-pooling rows with the candidate herb removed.

    hc is herbs x compounds. Each output row describes one (compound, herb)
    reconstruction pair, whether observed or unobserved. Empty remaining contexts
    are marked invalid, not supplied with a compound-ID fallback. No C-P labels
    or learnable parameters are used here.
    """
    hc = binary_matrix(hc)
    compounds, herbs = np.asarray(compounds), np.asarray(herbs)
    if compounds.ndim != 1 or herbs.ndim != 1 or compounds.shape != herbs.shape:
        raise ValueError('Compound and herb indices must be matching vectors.')
    for values, size in ((compounds, hc.shape[1]), (herbs, hc.shape[0])):
        if values.size and (not np.issubdtype(values.dtype, np.integer)
                            or np.any(values < 0) or np.any(values >= size)):
            raise ValueError('Invalid entity indices.')
    compounds, herbs = compounds.astype(np.int64), herbs.astype(np.int64)
    rows = hc.T.tocsr()[compounds].copy()
    if rows.nnz:
        pair_rows = np.repeat(np.arange(len(compounds)), np.diff(rows.indptr))
        rows.data[rows.indices == herbs[pair_rows]] = 0
        rows.eliminate_zeros()
    degree = np.asarray(rows.sum(axis=1)).ravel()
    valid = degree > 0
    normalizer = np.divide(1., degree, out=np.zeros_like(degree), where=valid)
    return (sparse.diags(normalizer) @ rows).tocsr(), valid

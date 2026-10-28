"""CPU reference operator for exact, blockwise Boolean metapath propagation."""

import numpy as np

from util.mvcl_cti_data import binary_matrix


class BooleanMetapath:
    """Represent A=(B B^T>0) without retaining the complete square matrix.

    This NumPy/SciPy reference is not a differentiable training layer.
    Normalization matches the audited upstream normalize_adj for symmetric A:
    D^-1/2 A D^-1/2 + diag(1/(1+degree)), not standard A+I normalization.
    """

    def __init__(self, incidence, block_size=128):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError('block_size must be a positive integer.')
        self.incidence = binary_matrix(incidence, incidence.shape)
        self.block_size = block_size
        self.degree = np.zeros(incidence.shape[0], dtype=np.float64)
        self.peak_block_nnz = 0
        for start, block in self.blocks():
            self.degree[start:start + block.shape[0]] = np.diff(block.indptr)
            self.peak_block_nnz = max(self.peak_block_nnz, block.nnz)
        self.inv_sqrt = np.zeros_like(self.degree)
        np.divide(1., np.sqrt(self.degree), out=self.inv_sqrt, where=self.degree > 0)

    def blocks(self):
        for start in range(0, self.incidence.shape[0], self.block_size):
            block = (self.incidence[start:start + self.block_size] @ self.incidence.T).tocsr()
            block.eliminate_zeros()
            block.data[:] = 1.
            yield start, block

    def matmul(self, features, normalized=True):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != len(self.degree):
            raise ValueError('Expected node-by-feature matrix.')
        if not np.isfinite(features).all():
            raise ValueError('Non-finite features.')
        source = features * self.inv_sqrt[:, None] if normalized else features
        result = np.zeros_like(features)
        for start, block in self.blocks():
            stop = start + block.shape[0]
            result[start:stop] = block @ source
            if normalized:
                result[start:stop] *= self.inv_sqrt[start:stop, None]
                result[start:stop] += features[start:stop] / (1 + self.degree[start:stop, None])
        return result

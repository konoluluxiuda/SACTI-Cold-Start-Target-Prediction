"""Sparse storage helpers for fixed, typed MVCL-CTI graphs."""

import numpy as np
from scipy import sparse
import torch


def torch_sparse(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(torch.from_numpy(np.stack((coo.row, coo.col))).long(),
                                   torch.from_numpy(coo.data.astype(np.float32)), coo.shape).coalesce()


def row_mean(matrix, features):
    if matrix.layout == torch.strided:
        return (matrix / (matrix.sum(1, keepdim=True) + 1e-12)) @ features
    degree = torch.sparse.sum(matrix, dim=1).to_dense()
    values = matrix.values() / (degree[matrix.indices()[0]] + 1e-12)
    normalized = torch.sparse_coo_tensor(matrix.indices(), values, matrix.shape).coalesce()
    return torch.sparse.mm(normalized, features)


def homogeneous(relations, endpoints, order, counts):
    blocks = [[None for _ in order] for _ in order]
    index = {kind: i for i, kind in enumerate(order)}
    for relation, (left, right) in endpoints.items():
        blocks[index[left]][index[right]] = relations[relation]
    for i, left in enumerate(order):
        for j, right in enumerate(order):
            if blocks[i][j] is None:
                blocks[i][j] = sparse.csr_matrix((counts[left], counts[right]))
    return sparse.bmat(blocks, format='csr')

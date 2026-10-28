"""Exact fixed-graph autograd operator; no PyTorch/DGL dependency in data export."""

import numpy as np
import torch

from util.mvcl_metapath import BooleanMetapath


def _multiply(operator, features, normalized):
    if features.ndim != 2 or features.shape[0] != len(operator.degree):
        raise ValueError('Expected node-by-feature tensor.')
    if features.dtype not in (torch.float32, torch.float64):
        raise ValueError('Only float32/float64 propagation is supported.')
    inv = torch.as_tensor(operator.inv_sqrt, dtype=features.dtype, device=features.device)
    degree = torch.as_tensor(operator.degree, dtype=features.dtype, device=features.device)
    source = features * inv[:, None] if normalized else features
    result = torch.empty_like(features)
    for start, block in operator.blocks():
        coo = block.tocoo()
        indices = torch.as_tensor(np.stack([coo.row, coo.col]), dtype=torch.long,
                                  device=features.device)
        values = torch.ones(coo.nnz, dtype=features.dtype, device=features.device)
        matrix = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
        stop = start + block.shape[0]
        value = torch.sparse.mm(matrix, source)
        if normalized:
            value = value * inv[start:stop, None] + features[start:stop] / (1 + degree[start:stop, None])
        result[start:stop] = value
    return result


class _FixedSymmetricPropagation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, operator, normalized):
        ctx.operator = operator
        ctx.normalized = normalized
        return _multiply(operator, features, normalized)

    @staticmethod
    def backward(ctx, grad_output):
        # A and its normalization are symmetric. Recompute blocks instead of
        # retaining every block's sparse tensor until backward.
        return _FixedSymmetricPropagation.apply(grad_output, ctx.operator, ctx.normalized), None, None


class TorchBooleanMetapath:
    """Differentiable w.r.t. node features, not the fixed incidence matrix.

    Owns a private copy of the CPU reference so forward/backward share one graph.
    CPU builds exact Boolean blocks; sparse feature propagation follows the
    input tensor device. CUDA performance and support require separate checks.
    """

    def __init__(self, incidence, block_size=128):
        self._operator = BooleanMetapath(incidence, block_size)

    def __call__(self, features, normalized=True):
        return _FixedSymmetricPropagation.apply(features, self._operator, bool(normalized))

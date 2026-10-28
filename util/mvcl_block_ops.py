"""Exact row-block operations with activation recomputation for MVCL-CTI."""

import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _size(block_size):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('block_size must be a positive integer.')


def _recompute(function, *inputs):
    if torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
        return checkpoint(function, *inputs, use_reentrant=False)
    return function(*inputs)


def block_attention(q, k, v, block_size):
    """All keys remain visible to each query; only query rows are partitioned."""
    _size(block_size)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError('Incompatible Q/K/V shapes.')
    if not q.shape[0]:
        return q.sum() * 0 + v.sum() * 0 + q.new_empty((0, v.shape[1]))
    if not k.shape[0] or not q.shape[1]:
        raise ValueError('Nonempty queries require keys and positive feature dimension.')

    def attend(query, keys, values):
        return (query @ keys.T / math.sqrt(query.shape[1])).softmax(1) @ values

    return torch.cat([_recompute(attend, q[start:start + block_size], k, v)
                      for start in range(0, len(q), block_size)])


def block_identity_contrast(a, b, tau, block_size):
    """Symmetric identity-positive loss: softmax(exp(cos/tau)), not InfoNCE.

    Inputs are already unit-normalized. Row blocks cover ALL negative nodes.
    """
    _size(block_size)
    if a.ndim != 2 or a.shape != b.shape or tau <= 0:
        raise ValueError('Aligned 2-D views and positive tau required.')
    if not len(a):
        return (a.sum() + b.sum()) * 0
    loss = (a.sum() + b.sum()) * 0
    for left, right in ((a, b), (b, a)):
        for start in range(0, len(left), block_size):
            # Bind the offset now: checkpoint replays this callable in backward.
            def term(rows, other, offset=start):
                logits = torch.exp(rows @ other.T / tau)
                positives = torch.arange(len(rows), device=rows.device) + offset
                return F.cross_entropy(logits, positives, reduction='sum')
            loss = loss + _recompute(term, left[start:start + block_size], right) / len(a)
    return loss


def block_reconstruction(left, relation, right, target, block_size, mask=None):
    """Full relation squared error, optionally masked to observed training pairs.

    Targets/masks may be dense tensors or CPU SciPy sparse matrices. No relation
    score matrix is returned; exact zeros remain part of unmasked side losses.
    """
    _size(block_size)
    if left.ndim != 2 or right.ndim != 2 or relation.shape != (left.shape[1], right.shape[1]):
        raise ValueError('Incompatible decoder shapes.')
    if target.shape != (len(left), len(right)) or (mask is not None and mask.shape != target.shape):
        raise ValueError('Incompatible reconstruction target or mask.')

    def slice_constant(matrix, start, stop):
        block = matrix[start:stop]
        if isinstance(block, torch.Tensor):
            if block.requires_grad:
                raise ValueError('Relation targets must be fixed.')
            return block.to(device=left.device, dtype=left.dtype)
        return torch.as_tensor(block.toarray(), device=left.device, dtype=left.dtype)

    loss = (left.sum() + relation.sum() + right.sum()) * 0
    for start in range(0, len(left), block_size):
        stop = min(start + block_size, len(left))

        def term(rows, weights, others, begin=start, end=stop):
            residual = rows @ weights @ others.T - slice_constant(target, begin, end)
            if mask is not None:
                residual = residual * slice_constant(mask, begin, end)
            return residual.square().sum()

        loss = loss + _recompute(term, left[start:stop], relation, right)
    return loss

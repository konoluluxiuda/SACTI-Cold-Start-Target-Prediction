"""Backend-neutral path graph from explicit fold-local binary relations."""

import hashlib
import json
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from util.rgcn import RELATION_SPECS


def _binary_relation(matrix, name):
    if not sp.issparse(matrix) or matrix.ndim != 2:
        raise ValueError('%s must be a sparse binary matrix.' % name)
    values = matrix.tocoo(copy=True)
    if not np.all(np.isfinite(values.data)) or not np.all(
            (values.data == 0) | (values.data == 1)):
        raise ValueError('%s must contain only zero or one.' % name)
    values = values.tocsr()
    values.sum_duplicates()
    values.eliminate_zeros()
    values.data[:] = 1
    values.sort_indices()
    return values


class QueryPathGraph:
    """HC is H x C, training CP is C x P, and PD is P x D.

    The caller must supply inner-training CP, never the full labelled CP.
    Matrix values cannot establish split provenance; that is the caller's job.
    No normalization or target statistics are cached before query masking.
    """

    def __init__(self, hc, training_cp, pd):
        hc = _binary_relation(hc, 'H-C')
        cp = _binary_relation(training_cp, 'training C-P')
        pd = _binary_relation(pd, 'P-D')
        if hc.shape[1] != cp.shape[0] or cp.shape[1] != pd.shape[0]:
            raise ValueError('H-C, C-P and P-D entity dimensions disagree.')
        self.node_counts = dict(zip(
            ('herb', 'compound', 'protein', 'disease'),
            (hc.shape[0], cp.shape[0], cp.shape[1], pd.shape[1]),
        ))
        if min(self.node_counts.values()) <= 0:
            raise ValueError('All entity sets must be non-empty.')
        self.offsets = {}
        offset = 0
        for name, count in self.node_counts.items():
            self.offsets[name] = offset
            offset += count
        self.num_nodes = offset

        # Adjacent forward/inverse relations share the same local edge order.
        triples = []
        for relation_id, matrix in ((0, hc), (2, cp), (4, pd)):
            src, dst = matrix.nonzero()
            _, src_type, dst_type = RELATION_SPECS[relation_id]
            source = src + self.offsets[src_type]
            target = dst + self.offsets[dst_type]
            triples.extend((
                np.column_stack((source, target, np.full(src.size, relation_id))),
                np.column_stack((target, source, np.full(src.size, relation_id + 1))),
            ))
        self._triples = np.concatenate(triples).astype(np.int64)
        self._triples.setflags(write=False)
        self._cp_support = np.diff(cp.indptr).copy()
        self._cp_support.setflags(write=False)

    def cold_query_node(self, compound_index):
        """Validate zero training support before sharing any unmasked topology."""
        if (isinstance(compound_index, (bool, np.bool_)) or
                not isinstance(compound_index, Integral) or
                not 0 <= compound_index < self.node_counts['compound']):
            raise ValueError('Query must be one valid local compound integer ID.')
        if self._cp_support[int(compound_index)] != 0:
            raise ValueError('Shared cold inputs require zero training C-P support.')
        return int(compound_index) + self.offsets['compound']

    def for_query(self, compound_index):
        """Return one isolated graph, with the query's entire CP row removed.

        Queries must not share a union graph with different masking semantics.
        Returned triples are (global source, global destination, relation ID).
        Candidate position p always corresponds to original local protein p.
        """
        if (isinstance(compound_index, (bool, np.bool_))
                or not isinstance(compound_index, Integral)
                or not 0 <= compound_index < self.node_counts['compound']):
            raise ValueError('Query must be one valid local compound integer ID.')
        query = int(compound_index) + self.offsets['compound']
        edges = self._triples
        remove = ((edges[:, 2] == 2) & (edges[:, 0] == query)) | (
            (edges[:, 2] == 3) & (edges[:, 1] == query))
        masked = edges[~remove].copy()
        candidates = np.arange(self.node_counts['protein'], dtype=np.int64)
        candidates += self.offsets['protein']
        signature = hashlib.sha256()
        signature.update(json.dumps(self.node_counts, sort_keys=True).encode('ascii'))
        signature.update(masked.astype('<i8', copy=False).tobytes())
        return {
            'triples': masked,
            'query_node': query,
            'query_relation': 2,
            'candidate_nodes': candidates,
            'num_nodes': self.num_nodes,
            'node_counts': dict(self.node_counts),
            'offsets': dict(self.offsets),
            'relation_names': [spec[0] for spec in RELATION_SPECS],
            'removed_cp_positives': int(np.count_nonzero(remove) // 2),
            'masked_graph_sha256': signature.hexdigest(),
        }

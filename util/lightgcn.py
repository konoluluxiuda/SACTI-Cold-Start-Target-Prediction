import hashlib

import numpy as np


def build_normalized_bipartite_adjacency(
        num_compounds, num_proteins, edges):
    num_compounds = int(num_compounds)
    num_proteins = int(num_proteins)
    if num_compounds <= 0 or num_proteins <= 0:
        raise ValueError('LightGCN requires non-empty compound and protein sets.')

    unique_edges = sorted({
        (int(compound_index), int(protein_index))
        for compound_index, protein_index in edges
    })
    if not unique_edges:
        raise ValueError('LightGCN training graph contains no positive C-P edge.')
    for compound_index, protein_index in unique_edges:
        if not 0 <= compound_index < num_compounds:
            raise ValueError(
                'Compound index %d is outside [0, %d).' %
                (compound_index, num_compounds)
            )
        if not 0 <= protein_index < num_proteins:
            raise ValueError(
                'Protein index %d is outside [0, %d).' %
                (protein_index, num_proteins)
            )

    compound_indices = np.asarray(
        [edge[0] for edge in unique_edges], dtype=np.int64
    )
    protein_indices = np.asarray(
        [edge[1] for edge in unique_edges], dtype=np.int64
    )
    shifted_proteins = protein_indices + num_compounds
    rows = np.concatenate([compound_indices, shifted_proteins])
    columns = np.concatenate([shifted_proteins, compound_indices])
    node_count = num_compounds + num_proteins
    degrees = np.bincount(rows, minlength=node_count).astype(np.float32)
    values = 1.0 / np.sqrt(degrees[rows] * degrees[columns])
    indices = np.column_stack([rows, columns])
    order = np.lexsort((indices[:, 1], indices[:, 0]))

    edge_lines = [
        '%d\t%d' % (compound_index, protein_index)
        for compound_index, protein_index in unique_edges
    ]
    return {
        'indices': indices[order].astype(np.int64),
        'values': values[order].astype(np.float32),
        'shape': (node_count, node_count),
        'edge_count': len(unique_edges),
        'active_compounds': int(np.count_nonzero(degrees[:num_compounds])),
        'active_proteins': int(np.count_nonzero(degrees[num_compounds:])),
        'edge_sha256': hashlib.sha256(
            ('\n'.join(edge_lines) + '\n').encode('utf-8')
        ).hexdigest(),
    }

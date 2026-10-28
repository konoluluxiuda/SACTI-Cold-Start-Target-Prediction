import hashlib

import numpy as np


RELATION_SPECS = (
    ('herb_to_compound', 'herb', 'compound'),
    ('compound_to_herb', 'compound', 'herb'),
    ('compound_to_protein', 'compound', 'protein'),
    ('protein_to_compound', 'protein', 'compound'),
    ('protein_to_disease', 'protein', 'disease'),
    ('disease_to_protein', 'disease', 'protein'),
)


def _validated_edges(source_count, destination_count, edges, relation_name):
    unique_edges = sorted({
        (int(source_index), int(destination_index))
        for source_index, destination_index in edges
    })
    if not unique_edges:
        raise ValueError(
            'R-GCN relation %s contains no edge.' % relation_name
        )
    for source_index, destination_index in unique_edges:
        if not 0 <= source_index < source_count:
            raise ValueError(
                '%s source index %d is outside [0, %d).' % (
                    relation_name, source_index, source_count
                )
            )
        if not 0 <= destination_index < destination_count:
            raise ValueError(
                '%s destination index %d is outside [0, %d).' % (
                    relation_name, destination_index, destination_count
                )
            )
    return unique_edges


def _build_directed_relation(
        source_type,
        destination_type,
        source_count,
        destination_count,
        edges,
        relation_name):
    unique_edges = _validated_edges(
        source_count,
        destination_count,
        edges,
        relation_name,
    )
    sources = np.asarray(
        [edge[0] for edge in unique_edges], dtype=np.int64
    )
    destinations = np.asarray(
        [edge[1] for edge in unique_edges], dtype=np.int64
    )
    destination_degrees = np.bincount(
        destinations, minlength=destination_count
    ).astype(np.float32)
    values = 1.0 / destination_degrees[destinations]
    indices = np.column_stack([destinations, sources])
    order = np.lexsort((indices[:, 1], indices[:, 0]))
    edge_text = '\n'.join(
        '%d\t%d' % edge for edge in unique_edges
    ) + '\n'
    return {
        'source_type': source_type,
        'destination_type': destination_type,
        'indices': indices[order].astype(np.int64),
        'values': values[order].astype(np.float32),
        'shape': (destination_count, source_count),
        'edge_count': len(unique_edges),
        'active_sources': int(len(set(sources.tolist()))),
        'active_destinations': int(np.count_nonzero(destination_degrees)),
        'edge_sha256': hashlib.sha256(
            edge_text.encode('utf-8')
        ).hexdigest(),
    }


def build_rgcn_relations(
        num_herbs,
        num_compounds,
        num_proteins,
        num_diseases,
        herb_compound_edges,
        compound_protein_edges,
        protein_disease_edges):
    node_counts = {
        'herb': int(num_herbs),
        'compound': int(num_compounds),
        'protein': int(num_proteins),
        'disease': int(num_diseases),
    }
    for entity_type, count in node_counts.items():
        if count <= 0:
            raise ValueError(
                'R-GCN requires a non-empty %s entity set.' % entity_type
            )

    canonical_edges = {
        'H_C': list(herb_compound_edges),
        'C_P': list(compound_protein_edges),
        'P_D': list(protein_disease_edges),
    }
    directed_edges = {
        'herb_to_compound': canonical_edges['H_C'],
        'compound_to_herb': [
            (compound, herb)
            for herb, compound in canonical_edges['H_C']
        ],
        'compound_to_protein': canonical_edges['C_P'],
        'protein_to_compound': [
            (protein, compound)
            for compound, protein in canonical_edges['C_P']
        ],
        'protein_to_disease': canonical_edges['P_D'],
        'disease_to_protein': [
            (disease, protein)
            for protein, disease in canonical_edges['P_D']
        ],
    }

    relations = {}
    for relation_name, source_type, destination_type in RELATION_SPECS:
        relations[relation_name] = _build_directed_relation(
            source_type,
            destination_type,
            node_counts[source_type],
            node_counts[destination_type],
            directed_edges[relation_name],
            relation_name,
        )
    return {
        'normalization': 'destination_relation_mean',
        'node_counts': node_counts,
        'source_edge_counts': {
            'H_C': relations['herb_to_compound']['edge_count'],
            'C_P': relations['compound_to_protein']['edge_count'],
            'P_D': relations['protein_to_disease']['edge_count'],
        },
        'relations': relations,
    }

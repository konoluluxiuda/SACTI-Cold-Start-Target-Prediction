import hashlib

import numpy as np

from util.rgcn import RELATION_SPECS, build_rgcn_relations


def _sample_relation(relation_name, relation, max_neighbors, seed):
    indices = np.asarray(relation['indices'], dtype=np.int64)
    destinations = indices[:, 0]
    sources = indices[:, 1]
    original_edge_count = len(indices)
    keep = np.ones(original_edge_count, dtype=bool)

    if max_neighbors > 0:
        keep[:] = False
        relation_salt = int.from_bytes(
            hashlib.sha256(relation_name.encode('utf-8')).digest()[:8],
            byteorder='little',
        )
        scores = (
            sources.astype(np.uint64) * np.uint64(11400714819323198485)
            + destinations.astype(np.uint64) * np.uint64(14029467366897019727)
            + np.uint64(int(seed))
            + np.uint64(relation_salt)
        )
        starts = np.flatnonzero(np.r_[
            True,
            destinations[1:] != destinations[:-1],
        ])
        ends = np.r_[starts[1:], original_edge_count]
        for start, end in zip(starts, ends):
            if end - start <= max_neighbors:
                keep[start:end] = True
                continue
            local = np.argsort(
                scores[start:end],
                kind='mergesort',
            )[:max_neighbors]
            keep[start + local] = True

    sampled_sources = sources[keep]
    sampled_destinations = destinations[keep]
    edge_text = ''.join(
        '%d\t%d\n' % (source, destination)
        for source, destination in zip(
            sampled_sources,
            sampled_destinations,
        )
    )
    return {
        'source_type': relation['source_type'],
        'destination_type': relation['destination_type'],
        'source_indices': sampled_sources,
        'destination_indices': sampled_destinations,
        'shape': relation['shape'],
        'original_edge_count': original_edge_count,
        'edge_count': int(len(sampled_sources)),
        'active_sources': int(len(set(sampled_sources.tolist()))),
        'active_destinations': int(
            len(set(sampled_destinations.tolist()))
        ),
        'retention_ratio': (
            float(len(sampled_sources)) / original_edge_count
        ),
        'edge_sha256': hashlib.sha256(
            edge_text.encode('utf-8')
        ).hexdigest(),
    }


def build_hgt_relations(
        num_herbs,
        num_compounds,
        num_proteins,
        num_diseases,
        herb_compound_edges,
        compound_protein_edges,
        protein_disease_edges,
        max_neighbors=64,
        seed=2026):
    max_neighbors = int(max_neighbors)
    seed = int(seed)
    if max_neighbors < 0:
        raise ValueError('hgt.max.neighbors must be non-negative.')

    graph = build_rgcn_relations(
        num_herbs,
        num_compounds,
        num_proteins,
        num_diseases,
        herb_compound_edges,
        compound_protein_edges,
        protein_disease_edges,
    )
    sampled_relations = {
        relation_name: _sample_relation(
            relation_name,
            graph['relations'][relation_name],
            max_neighbors,
            seed,
        )
        for relation_name, _, _ in RELATION_SPECS
    }
    return {
        'attention_normalization': 'destination_segment_softmax',
        'sampling': {
            'mode': (
                'deterministic_relation_destination_cap'
                if max_neighbors > 0 else 'all_edges'
            ),
            'max_neighbors': max_neighbors,
            'seed': seed,
        },
        'node_counts': graph['node_counts'],
        'source_edge_counts': graph['source_edge_counts'],
        'sampled_source_edge_counts': {
            'H_C': sampled_relations['herb_to_compound']['edge_count'],
            'C_P': sampled_relations['compound_to_protein']['edge_count'],
            'P_D': sampled_relations['protein_to_disease']['edge_count'],
        },
        'original_directed_edges': int(sum(
            relation['original_edge_count']
            for relation in sampled_relations.values()
        )),
        'sampled_directed_edges': int(sum(
            relation['edge_count']
            for relation in sampled_relations.values()
        )),
        'relations': sampled_relations,
    }

import numpy as np
import scipy.sparse as sp

from util.listwise_expert_mixture import (
    absolute_jaccard_scores,
    loco_absolute_prototype_scores,
)


SUPPORTED_BASE_EXPERTS = ('prototype', 'jaccard')


def within_query_rank_percentile(scores):
    """Map scores to [0, 1], breaking ties by stable candidate position."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not scores.size:
        return scores.astype(np.float32)
    if not np.all(np.isfinite(scores)):
        raise ValueError('Scores contain NaN or infinity.')
    if scores.size == 1:
        return np.ones(1, dtype=np.float32)

    order = np.lexsort((np.arange(scores.size, dtype=np.int64), -scores))
    result = np.empty(scores.size, dtype=np.float64)
    denominator = float(scores.size - 1)
    result[order] = 1.0 - np.arange(scores.size, dtype=np.float64) / denominator
    return result.astype(np.float32)


def score_base_expert(
        expert, prototypes, jaccard_transfer, compound_herb_indices,
        compound_herb_mask, compound_indices, protein_indices,
        prior_strength=1.0):
    """Score requested fold-local pairs with one frozen H-C expert."""
    expert = str(expert).strip().lower()
    if expert not in SUPPORTED_BASE_EXPERTS:
        raise ValueError('Unknown base expert: %s.' % expert)
    if expert == 'prototype':
        return loco_absolute_prototype_scores(
            prototypes,
            compound_herb_indices,
            compound_herb_mask,
            compound_indices,
            protein_indices,
            prior_strength=prior_strength,
        )
    return absolute_jaccard_scores(
        jaccard_transfer, compound_indices, protein_indices
    )


def full_candidate_base_scores(
        expert, compound, num_proteins, prototypes, jaccard_transfer,
        compound_herb_indices, compound_herb_mask, prior_strength=1.0):
    proteins = np.arange(int(num_proteins), dtype=np.int64)
    compounds = np.full(proteins.size, int(compound), dtype=np.int64)
    absolute = score_base_expert(
        expert,
        prototypes,
        jaccard_transfer,
        compound_herb_indices,
        compound_herb_mask,
        compounds,
        proteins,
        prior_strength=prior_strength,
    )
    return {
        'absolute': np.asarray(absolute, dtype=np.float32),
        'rank_percentile': within_query_rank_percentile(absolute),
    }


def build_support_gate_features(hc, cp):
    """Build four fold-local compound features without test C-P labels."""
    hc = sp.csr_matrix(hc, dtype=np.float32)
    cp = sp.csr_matrix(cp, dtype=np.float32)
    if hc.shape[1] != cp.shape[0]:
        raise ValueError('H-C and C-P matrices disagree on compound count.')

    compound_herbs = hc.transpose().tocsr()
    compound_supported = np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0
    supported_compounds = np.flatnonzero(compound_supported).astype(np.int64)
    supported_hc = hc[:, supported_compounds].tocsr()
    source_degrees = np.asarray(
        compound_herbs[supported_compounds].getnnz(axis=1)
    ).reshape(-1).astype(np.float32)
    supported_local = {
        int(compound): index
        for index, compound in enumerate(supported_compounds)
    }
    herb_support_counts = np.asarray(
        supported_hc.getnnz(axis=1)
    ).reshape(-1).astype(np.int64)

    raw = np.zeros((hc.shape[1], 4), dtype=np.float32)
    for compound in range(hc.shape[1]):
        start = compound_herbs.indptr[compound]
        stop = compound_herbs.indptr[compound + 1]
        herbs = compound_herbs.indices[start:stop]
        degree = int(herbs.size)
        self_supported = int(compound_supported[compound])
        raw[compound, 0] = np.log1p(degree)
        if degree:
            informative = herb_support_counts[herbs] - self_supported > 0
            raw[compound, 1] = float(np.mean(informative))

        if not degree or not supported_compounds.size:
            continue
        local_parts = []
        for herb in herbs:
            source_start = supported_hc.indptr[herb]
            source_stop = supported_hc.indptr[herb + 1]
            local_parts.append(supported_hc.indices[source_start:source_stop])
        if not local_parts:
            continue
        all_local = np.concatenate(local_parts)
        if not all_local.size:
            continue
        intersections = np.bincount(
            all_local, minlength=supported_compounds.size
        ).astype(np.float32)
        neighbor_local = np.flatnonzero(intersections > 0)
        if self_supported:
            self_local = supported_local[int(compound)]
            neighbor_local = neighbor_local[neighbor_local != self_local]
        raw[compound, 2] = np.log1p(neighbor_local.size)
        if neighbor_local.size:
            unions = (
                float(degree) + source_degrees[neighbor_local]
                - intersections[neighbor_local]
            )
            raw[compound, 3] = float(np.max(
                intersections[neighbor_local] / np.maximum(unions, 1.0)
            ))

    return {
        'values': raw,
        'feature_names': (
            'log1p_hc_degree',
            'informative_herb_fraction',
            'log1p_hc_train_neighbor_count',
            'max_hc_jaccard',
        ),
    }


def bounded_residual_scores(base_scores, gate, residual, tau):
    """Apply the registered bounded correction to base rank scores."""
    base_scores = np.asarray(base_scores, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if base_scores.shape != residual.shape:
        raise ValueError('Base scores and residuals must have matching shapes.')
    try:
        gate = np.broadcast_to(gate, base_scores.shape)
    except ValueError as error:
        raise ValueError('Gate does not broadcast to pair scores.') from error
    tau = float(tau)
    if tau <= 0:
        raise ValueError('tau must be positive.')
    if np.any((gate < 0) | (gate > 1)):
        raise ValueError('Gate values must be in [0, 1].')
    return (
        base_scores + gate * tau * np.tanh(residual)
    ).astype(np.float32)


def _positive_target_arrays(positive_targets, compound):
    values = positive_targets[int(compound)]
    return np.asarray(sorted(set(int(value) for value in values)), dtype=np.int64)


def build_safe_listwise_candidate_batch(
        rng, positive_targets, num_proteins, prototypes, jaccard_transfer,
        compound_herb_indices, compound_herb_mask, base_expert,
        support_features, query_count=32, hard_unlabeled_count=256,
        random_unlabeled_count=64, prior_strength=1.0,
        base_rank_cache=None):
    """Build padded multi-positive query groups from the full protein universe."""
    if int(query_count) <= 0:
        raise ValueError('query_count must be positive.')
    if int(hard_unlabeled_count) < 0 or int(random_unlabeled_count) < 0:
        raise ValueError('Candidate counts cannot be negative.')
    if int(hard_unlabeled_count) + int(random_unlabeled_count) <= 0:
        raise ValueError('At least one unlabeled candidate is required.')
    num_proteins = int(num_proteins)
    eligible = np.asarray([
        compound for compound in range(len(positive_targets))
        if len(positive_targets[compound]) > 0
    ], dtype=np.int64)
    if not eligible.size:
        raise ValueError('No compound has a training positive target.')
    support_features = np.asarray(support_features, dtype=np.float32)
    if support_features.shape != (len(positive_targets), 4):
        raise ValueError('support_features must have shape [compounds, 4].')

    compounds = rng.choice(
        eligible,
        size=int(query_count),
        replace=eligible.size < int(query_count),
    ).astype(np.int64)
    rows = []
    all_proteins = np.arange(num_proteins, dtype=np.int64)
    for compound in compounds:
        positives = _positive_target_arrays(positive_targets, compound)
        if np.any((positives < 0) | (positives >= num_proteins)):
            raise ValueError('Positive target index is outside the protein universe.')
        cache_key = int(compound)
        rank_percentile = (
            None if base_rank_cache is None
            else base_rank_cache.get(cache_key)
        )
        if rank_percentile is None:
            rank_percentile = full_candidate_base_scores(
                base_expert,
                compound,
                num_proteins,
                prototypes,
                jaccard_transfer,
                compound_herb_indices,
                compound_herb_mask,
                prior_strength=prior_strength,
            )['rank_percentile']
            if base_rank_cache is not None:
                base_rank_cache[cache_key] = rank_percentile
        unlabeled_mask = np.ones(num_proteins, dtype=bool)
        unlabeled_mask[positives] = False
        unlabeled = all_proteins[unlabeled_mask]
        hard_count = min(int(hard_unlabeled_count), int(unlabeled.size))
        hard_order = np.lexsort((
            unlabeled,
            -rank_percentile[unlabeled],
        ))
        hard = unlabeled[hard_order[:hard_count]]
        remaining = unlabeled[hard_order[hard_count:]]
        random_count = min(int(random_unlabeled_count), int(remaining.size))
        random_candidates = (
            rng.choice(remaining, size=random_count, replace=False)
            if random_count else np.empty(0, dtype=np.int64)
        )
        candidates = np.concatenate([positives, hard, random_candidates])
        labels = np.concatenate([
            np.ones(positives.size, dtype=np.float32),
            np.zeros(hard.size + random_candidates.size, dtype=np.float32),
        ])
        hard_mask = np.concatenate([
            np.zeros(positives.size, dtype=np.float32),
            np.ones(hard.size, dtype=np.float32),
            np.zeros(random_candidates.size, dtype=np.float32),
        ])
        permutation = rng.permutation(candidates.size)
        rows.append({
            'compound': int(compound),
            'proteins': candidates[permutation],
            'labels': labels[permutation],
            'hard_mask': hard_mask[permutation],
            'base_scores': rank_percentile[candidates][permutation],
        })

    width = max(len(row['proteins']) for row in rows)
    protein_matrix = np.zeros((len(rows), width), dtype=np.int32)
    label_matrix = np.zeros((len(rows), width), dtype=np.float32)
    hard_matrix = np.zeros((len(rows), width), dtype=np.float32)
    base_matrix = np.zeros((len(rows), width), dtype=np.float32)
    valid_matrix = np.zeros((len(rows), width), dtype=np.float32)
    for index, row in enumerate(rows):
        count = len(row['proteins'])
        protein_matrix[index, :count] = row['proteins']
        label_matrix[index, :count] = row['labels']
        hard_matrix[index, :count] = row['hard_mask']
        base_matrix[index, :count] = row['base_scores']
        valid_matrix[index, :count] = 1.0

    return {
        'compound_indices': compounds.astype(np.int32),
        'protein_indices': protein_matrix,
        'labels': label_matrix,
        'hard_unlabeled_mask': hard_matrix,
        'base_rank_scores': base_matrix,
        'valid_mask': valid_matrix,
        'support_features': support_features[compounds],
        'query_count': int(len(rows)),
        'candidate_width': int(width),
        'positive_count': int(np.sum(label_matrix)),
        'hard_unlabeled_count': int(np.sum(hard_matrix)),
    }

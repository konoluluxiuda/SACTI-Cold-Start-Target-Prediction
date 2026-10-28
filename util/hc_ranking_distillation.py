import numpy as np

from util.listwise_expert_mixture import (
    absolute_jaccard_scores,
    loco_absolute_prototype_scores,
)
from util.score_fusion import percentile_scores


def consensus_teacher_scores(
        prototypes, jaccard_transfer, compound_herb_indices,
        compound_herb_mask, compound_indices, protein_indices,
        prior_strength=1.0):
    """Return fixed equal-weight rank-percentile H-C teacher scores."""
    compound_indices = np.asarray(compound_indices, dtype=np.int64).reshape(-1)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must match.')
    prototype = loco_absolute_prototype_scores(
        prototypes,
        compound_herb_indices,
        compound_herb_mask,
        compound_indices,
        protein_indices,
        prior_strength=prior_strength,
    )
    jaccard = absolute_jaccard_scores(
        jaccard_transfer, compound_indices, protein_indices
    )
    return (
        0.5 * percentile_scores(prototype)
        + 0.5 * percentile_scores(jaccard)
    ).astype(np.float32)


def build_distillation_candidate_batch(
        rng, positive_pairs, positive_targets, num_proteins,
        prototypes, jaccard_transfer, compound_herb_indices,
        compound_herb_mask, query_count=32, hard_negative_count=8,
        random_negative_count=8, hard_pool_size=64, prior_strength=1.0):
    """Build deterministic fold-local LOCO teacher candidate groups."""
    positive_pairs = np.asarray(positive_pairs, dtype=np.int64)
    if positive_pairs.ndim != 2 or positive_pairs.shape[1] != 2:
        raise ValueError('positive_pairs must have shape [N, 2].')
    if not len(positive_pairs):
        raise ValueError('positive_pairs cannot be empty.')
    if query_count <= 0:
        raise ValueError('query_count must be positive.')
    if hard_negative_count < 0 or random_negative_count < 0:
        raise ValueError('Negative counts cannot be negative.')
    required_negatives = hard_negative_count + random_negative_count
    if required_negatives <= 0:
        raise ValueError('At least one candidate negative is required.')
    if hard_pool_size < required_negatives:
        raise ValueError('hard_pool_size must cover all candidate negatives.')

    draw = rng.choice(
        len(positive_pairs), size=int(query_count),
        replace=len(positive_pairs) < int(query_count),
    )
    queries = positive_pairs[draw]
    all_proteins = np.arange(int(num_proteins), dtype=np.int64)
    compound_rows = []
    protein_rows = []
    teacher_rows = []
    jaccard_rows = []

    for compound, positive in queries:
        known = set(int(value) for value in positive_targets[int(compound)])
        available = np.asarray(
            [value for value in all_proteins if int(value) not in known],
            dtype=np.int64,
        )
        if available.size < required_negatives:
            continue
        pool_size = min(int(hard_pool_size), int(available.size))
        pool = rng.choice(available, size=pool_size, replace=False)
        pool_compounds = np.full(pool_size, int(compound), dtype=np.int64)
        pool_teacher = consensus_teacher_scores(
            prototypes,
            jaccard_transfer,
            compound_herb_indices,
            compound_herb_mask,
            pool_compounds,
            pool,
            prior_strength=prior_strength,
        )
        hard_order = np.argsort(-pool_teacher, kind='mergesort')
        hard_positions = hard_order[:int(hard_negative_count)]
        hard = pool[hard_positions]
        remaining_positions = hard_order[int(hard_negative_count):]
        random_positions = rng.choice(
            remaining_positions,
            size=int(random_negative_count),
            replace=False,
        )
        negatives = np.concatenate([hard, pool[random_positions]])
        candidates = np.concatenate([
            np.asarray([positive], dtype=np.int64), negatives
        ])
        candidates = candidates[rng.permutation(candidates.size)]
        compounds = np.full(candidates.size, int(compound), dtype=np.int64)
        teacher = consensus_teacher_scores(
            prototypes,
            jaccard_transfer,
            compound_herb_indices,
            compound_herb_mask,
            compounds,
            candidates,
            prior_strength=prior_strength,
        )
        jaccard = absolute_jaccard_scores(
            jaccard_transfer, compounds, candidates
        ) - np.asarray(jaccard_transfer['target_prior'], dtype=np.float32)[
            candidates
        ]
        compound_rows.append(compounds)
        protein_rows.append(candidates)
        teacher_rows.append(teacher)
        jaccard_rows.append(jaccard.astype(np.float32))

    if not compound_rows:
        raise ValueError('No eligible distillation query could be sampled.')
    return {
        'compound_indices': np.concatenate(compound_rows).astype(np.int32),
        'protein_indices': np.concatenate(protein_rows).astype(np.int32),
        'teacher_scores': np.stack(teacher_rows).astype(np.float32),
        'jaccard_residuals': np.stack(jaccard_rows).astype(np.float32),
        'query_count': len(compound_rows),
        'candidate_count': 1 + required_negatives,
    }

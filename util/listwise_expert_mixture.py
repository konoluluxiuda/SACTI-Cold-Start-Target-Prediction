import numpy as np

from util.dual_granularity_transfer import loco_jaccard_pair_residuals


def loco_absolute_prototype_scores(
        prototypes, compound_herb_indices, compound_herb_mask,
        compound_indices, protein_indices, prior_strength=1.0):
    """Return empirical-Bayes scores with a pair-specific LOCO prior."""
    prior_strength = float(prior_strength)
    if prior_strength <= 0:
        raise ValueError('prior_strength must be positive.')
    compound_indices = np.asarray(compound_indices, dtype=np.int64).reshape(-1)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must match.')

    herb_indices = np.asarray(compound_herb_indices, dtype=np.int64)[
        compound_indices]
    herb_mask = np.asarray(compound_herb_mask, dtype=np.float32)[
        compound_indices]
    herb_target_counts = np.asarray(
        prototypes['herb_target_counts'], dtype=np.float32)
    herb_support_counts = np.asarray(
        prototypes['herb_support_counts'], dtype=np.float32)
    sentinel = herb_target_counts.shape[0]
    safe_herbs = np.where(herb_indices == sentinel, 0, herb_indices)
    pair_targets = np.broadcast_to(protein_indices[:, None], herb_indices.shape)

    compound_supported = np.asarray(
        prototypes['compound_supported'], dtype=np.float32
    )[compound_indices]
    pair_keys = (
        compound_indices * np.int64(prototypes['num_proteins'])
        + protein_indices
    )
    training_keys = np.asarray(prototypes['training_edge_keys'], dtype=np.int64)
    self_positive = np.zeros(pair_keys.shape, dtype=np.float32)
    if training_keys.size:
        positions = np.searchsorted(training_keys, pair_keys)
        in_range = positions < training_keys.size
        self_positive[in_range] = (
            training_keys[positions[in_range]] == pair_keys[in_range]
        ).astype(np.float32)

    supported_count = float(prototypes['supported_compound_count'])
    target_totals = (
        np.asarray(prototypes['target_prevalence'], dtype=np.float32)
        * supported_count
    )
    loco_count = supported_count - compound_supported
    loco_prior = np.divide(
        target_totals[protein_indices] - self_positive,
        loco_count,
        out=np.zeros_like(self_positive),
        where=loco_count > 0,
    )
    source_counts = herb_support_counts[safe_herbs] - compound_supported[:, None]
    target_counts = (
        herb_target_counts[safe_herbs, pair_targets]
        - self_positive[:, None]
    )
    valid = (herb_mask > 0) & (source_counts > 0)
    posterior = (
        target_counts + prior_strength * loco_prior[:, None]
    ) / np.maximum(source_counts + prior_strength, prior_strength)
    valid_counts = np.sum(valid, axis=1)
    return np.divide(
        np.sum(posterior * valid.astype(np.float32), axis=1),
        valid_counts,
        out=loco_prior.astype(np.float32).copy(),
        where=valid_counts > 0,
    ).astype(np.float32)


def absolute_jaccard_scores(transfer, compound_indices, protein_indices):
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    residual = loco_jaccard_pair_residuals(
        transfer, compound_indices, protein_indices)
    prior = np.asarray(
        transfer['target_prior'], dtype=np.float32
    )[protein_indices]
    return (prior + residual).astype(np.float32)


def compound_support_features(hc, cp, jaccard_transfer):
    """Build label-safe compound query features for an expert router."""
    compound_herbs = hc.transpose().tocsr()
    supported = np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0
    supported_compounds = np.flatnonzero(supported)
    supported_hc = hc[:, supported_compounds].tocsr()
    supported_local = {
        int(compound): position
        for position, compound in enumerate(supported_compounds)
    }
    herb_has_support = np.asarray(
        supported_hc.getnnz(axis=1)
    ).reshape(-1) > 0
    raw = np.zeros((hc.shape[1], 3), dtype=np.float32)
    for compound in range(hc.shape[1]):
        start = compound_herbs.indptr[compound]
        stop = compound_herbs.indptr[compound + 1]
        herbs = compound_herbs.indices[start:stop]
        raw[compound, 0] = len(herbs)
        raw[compound, 1] = int(np.sum(herb_has_support[herbs]))
        neighbor_local = set()
        for herb in herbs:
            source_start = supported_hc.indptr[herb]
            source_stop = supported_hc.indptr[herb + 1]
            neighbor_local.update(
                int(value)
                for value in supported_hc.indices[source_start:source_stop]
            )
        if supported[compound]:
            neighbor_local.discard(supported_local[compound])
        raw[compound, 2] = len(neighbor_local)

    transformed = np.log1p(raw)
    reference = transformed[supported]
    if not len(reference):
        reference = transformed
    mean = np.mean(reference, axis=0).astype(np.float32)
    scale = np.std(reference, axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    normalized = ((transformed - mean) / scale).astype(np.float32)
    return {
        'raw': raw,
        'normalized': normalized,
        'mean': mean,
        'scale': scale,
        'feature_names': (
            'log_herb_count',
            'log_supported_herb_count',
            'log_jaccard_neighbor_count',
        ),
    }


def combine_expert_scores(prototype, jaccard, neural, weights):
    prototype = np.asarray(prototype, dtype=np.float32)
    jaccard = np.asarray(jaccard, dtype=np.float32)
    neural = np.asarray(neural, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if prototype.shape != jaccard.shape or prototype.shape != neural.shape:
        raise ValueError('Expert score arrays must have matching shapes.')
    if weights.shape[-1] != 3:
        raise ValueError('Expert weights must contain three columns.')
    if weights.shape[:-1] == prototype.shape:
        aligned_weights = weights
    elif (
        prototype.ndim >= 2
        and weights.shape[:-1] == prototype.shape[:-1]
    ):
        aligned_weights = np.expand_dims(weights, axis=-2)
    else:
        raise ValueError('Expert weights do not match score rows.')
    stacked = np.stack([prototype, jaccard, neural], axis=-1)
    return np.sum(stacked * aligned_weights, axis=-1).astype(np.float32)

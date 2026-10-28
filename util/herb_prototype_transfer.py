import numpy as np
import scipy.sparse as sp


def _binary_csr(matrix):
    result = sp.csr_matrix(matrix, dtype=np.float32).copy()
    if result.nnz:
        result.data[:] = 1.0
        result.eliminate_zeros()
    return result


def build_support_calibrated_herb_prototypes(hc, cp):
    """Build fold-local herb-to-target statistics from supported compounds."""
    hc = _binary_csr(hc)
    cp = _binary_csr(cp)
    if hc.shape[1] != cp.shape[0]:
        raise ValueError('H-C and C-P matrices disagree on compound count.')

    compound_supported = (
        np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0
    ).astype(np.float32)
    supported_hc = hc.dot(sp.diags(compound_supported, dtype=np.float32))
    herb_target_counts = supported_hc.dot(cp).toarray().astype(np.float32)
    herb_support_counts = np.asarray(
        supported_hc.sum(axis=1), dtype=np.float32
    ).reshape(-1)
    supported_compound_count = int(np.sum(compound_supported))
    if supported_compound_count:
        target_prevalence = (
            np.asarray(cp.sum(axis=0), dtype=np.float32).reshape(-1)
            / float(supported_compound_count)
        )
    else:
        target_prevalence = np.zeros(cp.shape[1], dtype=np.float32)
    cp_rows, cp_columns = cp.nonzero()
    training_edge_keys = np.sort(
        cp_rows.astype(np.int64) * np.int64(cp.shape[1])
        + cp_columns.astype(np.int64)
    )

    return {
        'herb_target_counts': herb_target_counts,
        'herb_support_counts': herb_support_counts,
        'target_prevalence': target_prevalence.astype(np.float32),
        'target_totals': np.asarray(cp.sum(axis=0), dtype=np.float32).reshape(-1),
        'compound_supported': compound_supported,
        'training_edge_keys': training_edge_keys,
        'num_proteins': int(cp.shape[1]),
        'supported_compound_count': supported_compound_count,
        'training_positive_edges': int(cp.nnz),
    }


def support_calibrated_herb_prototype_scores(
        prototypes,
        compound_herb_indices,
        compound_herb_mask,
        compound_indices,
        protein_indices,
        prior_strength=1.0,
        return_diagnostics=False,
        exclude_query_prior=False,
        raw_scores=False):
    """Compute LOCO prototype residuals, or raw posteriors with excluded priors."""
    prior_strength = float(prior_strength)
    if raw_scores and not exclude_query_prior:
        raise ValueError('Raw transfer requires query-excluded priors.')
    if prior_strength <= 0:
        raise ValueError('prior_strength must be positive.')

    compound_indices = np.asarray(compound_indices, dtype=np.int64).reshape(-1)
    protein_indices = np.asarray(protein_indices, dtype=np.int64).reshape(-1)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must have matching shapes.')

    herb_indices = np.asarray(compound_herb_indices, dtype=np.int64)[
        compound_indices
    ]
    herb_mask = np.asarray(compound_herb_mask, dtype=np.float32)[
        compound_indices
    ]
    herb_target_counts = np.asarray(
        prototypes['herb_target_counts'], dtype=np.float32
    )
    herb_support_counts = np.asarray(
        prototypes['herb_support_counts'], dtype=np.float32
    )
    sentinel = herb_target_counts.shape[0]
    if np.any((herb_indices < 0) | (herb_indices > sentinel)):
        raise ValueError('Compound-herb indices contain an invalid herb ID.')

    safe_herb_indices = np.where(
        herb_indices == sentinel, 0, herb_indices
    )
    pair_targets = np.broadcast_to(
        protein_indices[:, None], herb_indices.shape
    )
    target_counts = herb_target_counts[safe_herb_indices, pair_targets]
    support_counts = herb_support_counts[safe_herb_indices]

    compound_supported = np.asarray(
        prototypes['compound_supported'], dtype=np.float32
    )[compound_indices]
    pair_keys = (
        compound_indices * np.int64(prototypes['num_proteins'])
        + protein_indices
    )
    training_edge_keys = np.asarray(
        prototypes['training_edge_keys'], dtype=np.int64
    )
    self_positive = np.zeros(pair_keys.shape, dtype=np.float32)
    if training_edge_keys.size:
        positions = np.searchsorted(training_edge_keys, pair_keys)
        in_range = positions < training_edge_keys.size
        self_positive[in_range] = (
            training_edge_keys[positions[in_range]] == pair_keys[in_range]
        ).astype(np.float32)
    source_counts = support_counts - compound_supported[:, None]
    target_counts = target_counts - self_positive[:, None]
    target_prior = np.asarray(
        prototypes['target_prevalence'], dtype=np.float32
    )[protein_indices, None]
    if exclude_query_prior:
        # Match rebuilding statistics after removing the entire query C-P row.
        denominator = prototypes['supported_compound_count'] - compound_supported
        numerator = prototypes['target_totals'][protein_indices] - self_positive
        target_prior = np.divide(
            numerator, denominator, out=np.zeros_like(numerator),
            where=denominator > 0,
        )[:, None]

    valid = (herb_mask > 0) & (source_counts > 0)
    posterior = (
        target_counts + prior_strength * target_prior
    ) / np.maximum(source_counts + prior_strength, prior_strength)
    values = posterior if raw_scores else posterior - target_prior
    residuals = values * valid.astype(np.float32)
    valid_counts = np.sum(valid, axis=1)
    pair_scores = np.divide(
        np.sum(residuals, axis=1),
        valid_counts,
        out=np.zeros(compound_indices.shape[0], dtype=np.float32),
        where=valid_counts > 0,
    )
    if raw_scores:
        pair_scores = np.where(valid_counts > 0, pair_scores, target_prior[:, 0])

    if not return_diagnostics:
        return pair_scores.astype(np.float32)
    evidence_mask = valid_counts > 0
    return pair_scores.astype(np.float32), {
        'pairs': int(pair_scores.size),
        'evidence_pairs': int(np.sum(evidence_mask)),
        'evidence_coverage': (
            float(np.mean(evidence_mask)) if pair_scores.size else 0.0
        ),
        'mean_abs_residual': (
            float(np.mean(np.abs(pair_scores[evidence_mask])))
            if np.any(evidence_mask) else 0.0
        ),
    }

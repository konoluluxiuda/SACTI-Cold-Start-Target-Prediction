import numpy as np
import scipy.sparse as sp


def _binary_csr(matrix):
    result = sp.csr_matrix(matrix, dtype=np.float32).copy()
    if result.nnz:
        result.data[:] = 1.0
        result.eliminate_zeros()
    return result


def build_loco_jaccard_transfer(hc, cp):
    """Build fold-local state for leave-one-compound-out H-C propagation."""
    hc = _binary_csr(hc)
    cp = _binary_csr(cp)
    if hc.shape[1] != cp.shape[0]:
        raise ValueError('H-C and C-P matrices disagree on compound count.')
    compound_herbs = hc.transpose().tocsr()
    compound_supported = np.asarray(cp.getnnz(axis=1)).reshape(-1) > 0
    supported_compounds = np.flatnonzero(compound_supported).astype(np.int64)
    supported_hc = hc[:, supported_compounds].tocsr()
    supported_count = int(supported_compounds.size)
    if supported_count:
        target_prior = (
            np.asarray(cp.sum(axis=0), dtype=np.float32).reshape(-1)
            / float(supported_count)
        )
    else:
        target_prior = np.zeros(cp.shape[1], dtype=np.float32)
    return {
        'cp': cp,
        'compound_herbs': compound_herbs,
        'supported_hc': supported_hc,
        'supported_compounds': supported_compounds,
        'supported_compound_herb_degrees': np.asarray(
            compound_herbs[supported_compounds].sum(axis=1),
            dtype=np.float32,
        ).reshape(-1),
        'target_prior': target_prior.astype(np.float32),
        'target_totals': np.asarray(cp.sum(axis=0), dtype=np.float32).reshape(-1),
        'training_positive_edges': int(cp.nnz),
    }


def loco_jaccard_pair_residuals(
        transfer,
        compound_indices,
        protein_indices,
        return_diagnostics=False,
        exclude_query_prior=False,
        raw_scores=False):
    """Compute LOCO Jaccard residuals or raw propagation with prior fallback."""
    if raw_scores and not exclude_query_prior:
        raise ValueError('Raw transfer requires query-excluded priors.')
    compound_indices = np.asarray(
        compound_indices, dtype=np.int64
    ).reshape(-1)
    protein_indices = np.asarray(
        protein_indices, dtype=np.int64
    ).reshape(-1)
    if compound_indices.shape != protein_indices.shape:
        raise ValueError('Compound and protein indices must have matching shapes.')

    cp = transfer['cp']
    compound_herbs = transfer['compound_herbs']
    supported_hc = transfer['supported_hc']
    supported_compounds = transfer['supported_compounds']
    source_degrees = transfer['supported_compound_herb_degrees']
    target_prior = transfer['target_prior']
    if np.any((compound_indices < 0) | (compound_indices >= cp.shape[0])):
        raise ValueError('Invalid compound index.')
    if np.any((protein_indices < 0) | (protein_indices >= cp.shape[1])):
        raise ValueError('Invalid protein index.')

    residuals = np.zeros(compound_indices.size, dtype=np.float32)
    evidence_pairs = 0
    for compound in np.unique(compound_indices):
        pair_positions = np.flatnonzero(compound_indices == compound)
        pair_prior = target_prior[protein_indices[pair_positions]]
        if exclude_query_prior:
            supported = int(cp.indptr[compound + 1] > cp.indptr[compound])
            denominator = supported_compounds.size - supported
            self_labels = cp[compound, protein_indices[pair_positions]].toarray().reshape(-1)
            pair_prior = np.divide(
                transfer['target_totals'][protein_indices[pair_positions]] - self_labels,
                denominator, out=np.zeros(pair_positions.size, dtype=np.float32),
                where=denominator > 0,
            )
        if raw_scores:
            residuals[pair_positions] = pair_prior
        herb_start = compound_herbs.indptr[compound]
        herb_stop = compound_herbs.indptr[compound + 1]
        herbs = compound_herbs.indices[herb_start:herb_stop]
        if not herbs.size or not supported_compounds.size:
            continue

        neighbor_local_parts = []
        for herb in herbs:
            start = supported_hc.indptr[herb]
            stop = supported_hc.indptr[herb + 1]
            neighbor_local_parts.append(supported_hc.indices[start:stop])
        if not neighbor_local_parts:
            continue
        all_local = np.concatenate(neighbor_local_parts)
        if not all_local.size:
            continue
        intersections = np.bincount(
            all_local, minlength=supported_compounds.size
        ).astype(np.float32)
        neighbor_local = np.flatnonzero(intersections > 0)
        neighbor_compounds = supported_compounds[neighbor_local]
        not_self = neighbor_compounds != compound
        neighbor_local = neighbor_local[not_self]
        neighbor_compounds = neighbor_compounds[not_self]
        if not neighbor_compounds.size:
            continue

        candidate_degree = float(herbs.size)
        overlap = intersections[neighbor_local]
        unions = candidate_degree + source_degrees[neighbor_local] - overlap
        similarities = overlap / np.maximum(unions, 1.0)
        similarity_sum = float(np.sum(similarities))
        if similarity_sum <= 0:
            continue

        requested_targets, target_inverse = np.unique(
            protein_indices[pair_positions], return_inverse=True
        )
        propagated = np.asarray(
            cp[neighbor_compounds][:, requested_targets].transpose().dot(
                similarities
            ),
            dtype=np.float32,
        ).reshape(-1) / similarity_sum
        residuals[pair_positions] = (
            propagated[target_inverse]
            if raw_scores else propagated[target_inverse] - pair_prior
        )
        evidence_pairs += int(pair_positions.size)

    if not return_diagnostics:
        return residuals
    return residuals, {
        'pairs': int(residuals.size),
        'evidence_pairs': evidence_pairs,
        'evidence_coverage': (
            evidence_pairs / float(residuals.size) if residuals.size else 0.0
        ),
        'mean_abs_residual': (
            float(np.mean(np.abs(residuals))) if residuals.size else 0.0
        ),
    }

import numpy as np


def percentile_scores(values):
    """Map scores to average empirical percentiles while preserving ties."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if not np.all(np.isfinite(values)):
        raise ValueError('Scores contain NaN or infinity.')

    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    percentiles = np.empty(values.size, dtype=np.float64)
    start = 0
    denominator = float(values.size)
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        percentiles[order[start:stop]] = average_rank / denominator
        start = stop
    return percentiles


def fuse_candidate_scores(prior_scores, neural_scores, method):
    """Fuse two frozen candidate score vectors with no learned parameters."""
    prior_scores = np.asarray(prior_scores, dtype=np.float64).reshape(-1)
    neural_scores = np.asarray(neural_scores, dtype=np.float64).reshape(-1)
    if prior_scores.shape != neural_scores.shape:
        raise ValueError('Prior and neural score shapes do not match.')

    method = str(method).strip().lower()
    if method == 'prior':
        return prior_scores.copy()
    if method == 'neural':
        return neural_scores.copy()

    prior_percentile = percentile_scores(prior_scores)
    neural_percentile = percentile_scores(neural_scores)
    if method == 'prior_tie_break':
        # Distinct prior percentile groups differ by at least 1/N. The neural
        # perturbation is smaller than that gap, so it can only resolve ties.
        return prior_percentile + neural_percentile / float(prior_scores.size + 1)
    if method == 'percentile_mean':
        return 0.5 * (prior_percentile + neural_percentile)
    raise ValueError('Unknown score fusion method: %s.' % method)


def fuse_multiple_candidate_scores(score_vectors, method='percentile_mean'):
    """Fuse frozen expert rankings with fixed, parameter-free weights."""
    vectors = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in score_vectors
    ]
    if len(vectors) < 2:
        raise ValueError('At least two expert score vectors are required.')
    reference_shape = vectors[0].shape
    if any(values.shape != reference_shape for values in vectors[1:]):
        raise ValueError('Expert score shapes do not match.')
    method = str(method).strip().lower()
    if method != 'percentile_mean':
        raise ValueError('Unknown multi-expert fusion method: %s.' % method)
    percentiles = np.stack(
        [percentile_scores(values) for values in vectors], axis=0)
    return np.mean(percentiles, axis=0)

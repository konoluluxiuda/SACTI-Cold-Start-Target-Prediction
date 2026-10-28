import numpy as np


def global_token_attention_complexity(
        node_count, hyperedge_count, token_count, head_count=1):
    node_count = int(node_count)
    hyperedge_count = int(hyperedge_count)
    token_count = int(token_count)
    head_count = int(head_count)
    if node_count <= 0 or hyperedge_count <= 0:
        raise ValueError('Node and hyperedge counts must be positive.')
    if token_count <= 0 or head_count <= 0:
        raise ValueError('Token and head counts must be positive.')
    dense_pairs = node_count * node_count * head_count
    token_pairs = (
        hyperedge_count * token_count
        + node_count * token_count * head_count
    )
    return {
        'dense_node_pairs': dense_pairs,
        'token_attention_pairs': token_pairs,
        'pair_ratio': float(token_pairs) / float(dense_pairs),
    }


def _normalized_entropy(probabilities, axis, support_size):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=axis,
    )
    if int(support_size) <= 1:
        return np.zeros_like(entropy)
    return entropy / np.log(float(support_size))


def summarize_global_token_attention(
        edge_assignments, node_attention, token_embeddings):
    edge_assignments = np.asarray(edge_assignments, dtype=np.float64)
    node_attention = np.asarray(node_attention, dtype=np.float64)
    token_embeddings = np.asarray(token_embeddings, dtype=np.float64)
    if edge_assignments.ndim != 2:
        raise ValueError('edge_assignments must have shape [edges, tokens].')
    if node_attention.ndim != 3:
        raise ValueError(
            'node_attention must have shape [heads, nodes, tokens].'
        )
    edge_count, token_count = edge_assignments.shape
    if node_attention.shape[2] != token_count:
        raise ValueError('Node and edge attention token counts do not match.')
    if token_embeddings.shape[0] != token_count:
        raise ValueError('token_embeddings has the wrong token count.')
    if not (
            np.all(np.isfinite(edge_assignments))
            and np.all(np.isfinite(node_attention))
            and np.all(np.isfinite(token_embeddings))):
        raise ValueError('Global token diagnostics contain non-finite values.')

    active_edges = np.sum(edge_assignments, axis=1) > 1e-12
    active_edge_count = max(1, int(np.sum(active_edges)))
    edge_distributions = edge_assignments.transpose()
    edge_entropy = _normalized_entropy(
        edge_distributions, axis=1, support_size=active_edge_count
    )
    node_entropy = _normalized_entropy(
        node_attention, axis=2, support_size=token_count
    )

    normalized_tokens = token_embeddings / np.maximum(
        np.linalg.norm(token_embeddings, axis=1, keepdims=True), 1e-12
    )
    token_cosine = normalized_tokens.dot(normalized_tokens.transpose())
    off_diagonal = token_cosine[~np.eye(token_count, dtype=bool)]
    mean_abs_off_diagonal = (
        float(np.mean(np.abs(off_diagonal)))
        if off_diagonal.size else 0.0
    )
    return {
        'active_hyperedges': active_edge_count,
        'edge_assignment_entropy_mean': float(np.mean(edge_entropy)),
        'edge_assignment_max_weight_mean': float(
            np.mean(np.max(edge_distributions, axis=1))
        ),
        'node_attention_entropy_mean': float(np.mean(node_entropy)),
        'node_attention_max_weight_mean': float(
            np.mean(np.max(node_attention, axis=2))
        ),
        'token_mean_abs_off_diagonal_cosine': mean_abs_off_diagonal,
    }

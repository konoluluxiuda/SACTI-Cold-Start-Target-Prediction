import numpy as np
import scipy.sparse as sp


def hypergraph_pagerank(
        incidence, alpha=0.85, max_iter=100, tol=1e-8):
    incidence = sp.csr_matrix(incidence, dtype=np.float64)
    node_count, hyperedge_count = incidence.shape
    if node_count <= 0 or hyperedge_count <= 0:
        raise ValueError('Hypergraph incidence must have positive dimensions.')
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError('PageRank alpha must be between 0 and 1.')
    if int(max_iter) <= 0 or float(tol) <= 0:
        raise ValueError('PageRank max_iter and tol must be positive.')
    if incidence.nnz and np.any(incidence.data < 0):
        raise ValueError('Hypergraph incidence cannot contain negative values.')

    node_degree = np.asarray(incidence.sum(axis=1)).reshape(-1)
    edge_degree = np.asarray(incidence.sum(axis=0)).reshape(-1)
    node_inverse = np.zeros_like(node_degree)
    edge_inverse = np.zeros_like(edge_degree)
    np.divide(1.0, node_degree, out=node_inverse, where=node_degree > 0)
    np.divide(1.0, edge_degree, out=edge_inverse, where=edge_degree > 0)
    dangling = node_degree <= 0

    rank = np.full(node_count, 1.0 / node_count, dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        edge_mass = incidence.transpose().dot(rank * node_inverse)
        propagated = incidence.dot(edge_mass * edge_inverse)
        if np.any(dangling):
            propagated += np.sum(rank[dangling]) / node_count
        updated = (1.0 - float(alpha)) / node_count
        updated = updated + float(alpha) * propagated
        updated /= np.sum(updated)
        iterations = iteration
        if np.sum(np.abs(updated - rank)) <= float(tol):
            rank = updated
            converged = True
            break
        rank = updated

    mean_normalized = (rank * node_count).astype(np.float32)
    diagnostics = {
        'nodes': int(node_count),
        'hyperedges': int(hyperedge_count),
        'incidences': int(incidence.nnz),
        'zero_degree_nodes': int(np.sum(dangling)),
        'iterations': int(iterations),
        'converged': bool(converged),
        'probability_sum': float(np.sum(rank)),
        'prior_mean': float(np.mean(mean_normalized)),
        'prior_min': float(np.min(mean_normalized)),
        'prior_max': float(np.max(mean_normalized)),
    }
    return mean_normalized, diagnostics


def pagerank_linear_attention_numpy(
        queries, keys, values, pagerank_prior, epsilon=1e-6):
    queries = np.asarray(queries, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    prior = np.asarray(pagerank_prior, dtype=np.float64).reshape(-1)
    if queries.ndim != 3:
        raise ValueError('Queries must have shape [heads, nodes, dimensions].')
    if not (queries.shape == keys.shape == values.shape):
        raise ValueError('Queries, keys, and values must have matching shapes.')
    if prior.shape[0] != queries.shape[1]:
        raise ValueError('PageRank prior has the wrong node count.')
    if np.any(prior < 0) or not np.all(np.isfinite(prior)):
        raise ValueError('PageRank prior must be finite and non-negative.')
    if float(epsilon) <= 0:
        raise ValueError('Attention epsilon must be positive.')

    query_features = np.empty_like(queries)
    key_features = np.empty_like(keys)
    query_positive = queries > 0
    key_positive = keys > 0
    query_features[query_positive] = queries[query_positive] + 1.0
    query_features[~query_positive] = np.exp(queries[~query_positive])
    key_features[key_positive] = keys[key_positive] + 1.0
    key_features[~key_positive] = np.exp(keys[~key_positive])
    weighted_values = values * prior[None, :, None]
    kernel_state = np.matmul(
        np.swapaxes(key_features, 1, 2), weighted_values
    )
    key_sum = np.sum(key_features * prior[None, :, None], axis=1)
    numerator = np.matmul(query_features, kernel_state)
    denominator = np.sum(
        query_features * key_sum[:, None, :], axis=2, keepdims=True
    )
    return numerator / np.maximum(denominator, float(epsilon))


def pagerank_linear_attention_tf(
        tf_module, node_embeddings, pagerank_prior, weights,
        head_count, epsilon, name):
    embedding_size = node_embeddings.get_shape().as_list()[1]
    if embedding_size is None or embedding_size % int(head_count) != 0:
        raise ValueError('HPLGA requires an embedding size divisible by heads.')
    head_dimension = embedding_size // int(head_count)

    with tf_module.name_scope(name):
        projected = {}
        for projection in ('q', 'k', 'v'):
            tensor = tf_module.matmul(
                node_embeddings,
                weights[projection],
                name=projection + '_projection',
            )
            projected[projection] = tf_module.transpose(
                tf_module.reshape(
                    tensor, [-1, int(head_count), head_dimension]
                ),
                [1, 0, 2],
                name=projection + '_heads',
            )

        query_features = tf_module.nn.elu(projected['q']) + 1.0
        key_features = tf_module.nn.elu(projected['k']) + 1.0
        prior = tf_module.reshape(
            pagerank_prior, [1, -1, 1], name='pagerank_prior'
        )
        weighted_values = projected['v'] * prior
        kernel_state = tf_module.matmul(
            key_features,
            weighted_values,
            transpose_a=True,
            name='kernel_state',
        )
        key_sum = tf_module.reduce_sum(
            key_features * prior, axis=1, name='weighted_key_sum'
        )
        numerator = tf_module.matmul(
            query_features, kernel_state, name='numerator'
        )
        denominator = tf_module.reduce_sum(
            query_features * tf_module.expand_dims(key_sum, axis=1),
            axis=2,
            keepdims=True,
            name='denominator',
        )
        attended = numerator / tf_module.maximum(
            denominator, float(epsilon)
        )
        attended = tf_module.reshape(
            tf_module.transpose(attended, [1, 0, 2]),
            [-1, embedding_size],
            name='merge_heads',
        )
        update = tf_module.matmul(
            attended, weights['output'], name='output_projection'
        )
        residual_scale = tf_module.tanh(weights['gamma'][0])
        output = tf_module.add(
            node_embeddings,
            residual_scale * update,
            name='residual',
        )
    return output, {
        'kernel_state': kernel_state,
        'denominator': denominator,
        'update': update,
    }


def hplga_complexity(node_count, embedding_size, head_count):
    node_count = int(node_count)
    embedding_size = int(embedding_size)
    head_count = int(head_count)
    if node_count <= 0 or embedding_size <= 0 or head_count <= 0:
        raise ValueError('HPLGA complexity dimensions must be positive.')
    if embedding_size % head_count != 0:
        raise ValueError('Embedding size must be divisible by head count.')
    head_dimension = embedding_size // head_count
    return {
        'dense_attention_pairs': int(node_count * node_count * head_count),
        'quadratic_attention_elements': 0,
        'linear_node_features': int(
            3 * node_count * embedding_size
        ),
        'kernel_state_elements': int(
            head_count * head_dimension * head_dimension
        ),
    }

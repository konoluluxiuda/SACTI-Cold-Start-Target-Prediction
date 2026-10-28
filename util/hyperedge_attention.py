import numpy as np


def ordered_incidence_ids(edge_ids, node_ids):
    """Return forward (edge,node) and reverse (node,edge) sorted IDs."""
    edge_ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    if edge_ids.shape != node_ids.shape:
        raise ValueError("edge_ids and node_ids must have identical shapes.")
    forward_order = np.lexsort((node_ids, edge_ids))
    reverse_order = np.lexsort((edge_ids, node_ids))
    return {
        "forward_edge_ids": edge_ids[forward_order],
        "forward_node_ids": node_ids[forward_order],
        "reverse_edge_ids": edge_ids[reverse_order],
        "reverse_node_ids": node_ids[reverse_order],
    }


def hyperedge_specificity_prior(edge_degrees, node_count, clip=3.0):
    """Return a standardized log inverse-degree prior for active hyperedges."""
    degrees = np.asarray(edge_degrees, dtype=np.float64).reshape(-1)
    if int(node_count) <= 0:
        raise ValueError("node_count must be positive.")
    if np.any(degrees < 0):
        raise ValueError("Hyperedge degrees cannot be negative.")
    active = degrees > 0
    prior = np.zeros_like(degrees, dtype=np.float64)
    if not np.any(active):
        return prior.astype(np.float32)
    raw = np.log1p(float(node_count) / degrees[active])
    standard_deviation = float(np.std(raw))
    if standard_deviation > 1e-12:
        prior[active] = (raw - float(np.mean(raw))) / standard_deviation
    if clip is not None:
        prior = np.clip(prior, -float(clip), float(clip))
    return prior.astype(np.float32)


def segment_softmax(values, segment_ids, segment_count):
    """NumPy reference for softmax independently normalized per segment."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    segment_ids = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    if values.shape != segment_ids.shape:
        raise ValueError("values and segment_ids must have identical shapes.")
    if int(segment_count) <= 0:
        raise ValueError("segment_count must be positive.")
    if np.any(segment_ids < 0) or np.any(segment_ids >= int(segment_count)):
        raise ValueError("segment_ids contain an out-of-range value.")
    result = np.zeros_like(values)
    for segment in np.unique(segment_ids):
        mask = segment_ids == segment
        shifted = values[mask] - np.max(values[mask])
        exponentials = np.exp(shifted)
        result[mask] = exponentials / np.sum(exponentials)
    return result.astype(np.float32)


def factorized_incidence_attention(
        node_logits,
        edge_logits,
        edge_ids,
        node_ids,
        edge_count,
        node_count,
        specificity_prior=None,
        temperature=1.0,
        prior_scale=0.1):
    """Reference incidence weights for factorized node-edge attention."""
    node_logits = np.asarray(node_logits, dtype=np.float64).reshape(-1)
    edge_logits = np.asarray(edge_logits, dtype=np.float64).reshape(-1)
    edge_ids = np.asarray(edge_ids, dtype=np.int64).reshape(-1)
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    if edge_ids.shape != node_ids.shape:
        raise ValueError("edge_ids and node_ids must have identical shapes.")
    if node_logits.size != int(node_count):
        raise ValueError("node_logits length does not match node_count.")
    if edge_logits.size != int(edge_count):
        raise ValueError("edge_logits length does not match edge_count.")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive.")
    if float(prior_scale) < 0:
        raise ValueError("prior_scale cannot be negative.")
    if specificity_prior is None:
        specificity_prior = np.zeros(int(edge_count), dtype=np.float64)
    specificity_prior = np.asarray(
        specificity_prior, dtype=np.float64
    ).reshape(-1)
    if specificity_prior.size != int(edge_count):
        raise ValueError("specificity_prior length does not match edge_count.")

    node_to_edge = segment_softmax(
        node_logits[node_ids] / float(temperature),
        edge_ids,
        edge_count,
    )
    guided_edge_logits = edge_logits + float(prior_scale) * specificity_prior
    edge_to_node = segment_softmax(
        guided_edge_logits[edge_ids] / float(temperature),
        node_ids,
        node_count,
    )
    return node_to_edge, edge_to_node

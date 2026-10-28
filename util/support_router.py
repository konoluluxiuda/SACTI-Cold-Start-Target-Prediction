import hashlib
import math
from collections import Counter

import numpy as np


def _stable_rank(seed, compound_id):
    payload = "%d\t%s" % (int(seed), str(compound_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_pseudo_cold_compounds(
        training_records, eligible_compounds, ratio=0.1, seed=62026):
    """Select whole compounds for deterministic C-P graph masking."""
    ratio = float(ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("Pseudo-cold ratio must be between 0 and 1.")
    eligible = {str(value) for value in eligible_compounds}
    positive_degree = Counter()
    for compound_id, _, label in training_records:
        compound_id = str(compound_id)
        if float(label) > 0 and compound_id in eligible:
            positive_degree[compound_id] += 1
    candidates = sorted(positive_degree)
    if len(candidates) < 2:
        raise ValueError(
            "Support routing requires at least two H-C-supported training compounds."
        )
    selected_count = int(round(len(candidates) * ratio))
    selected_count = max(1, min(len(candidates) - 1, selected_count))
    ranked = sorted(candidates, key=lambda value: (_stable_rank(seed, value), value))
    selected = sorted(ranked[:selected_count])
    selected_set = set(selected)
    excluded_positive_edges = sum(
        degree for compound_id, degree in positive_degree.items()
        if compound_id in selected_set
    )
    assignment_lines = [
        "%s\t%d" % (compound_id, positive_degree[compound_id])
        for compound_id in selected
    ]
    assignment_sha256 = hashlib.sha256(
        ("\n".join(assignment_lines) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "selected_compounds": selected,
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "excluded_positive_edges": int(excluded_positive_edges),
        "ratio_requested": ratio,
        "ratio_actual": float(len(selected)) / float(len(candidates)),
        "seed": int(seed),
        "assignments_sha256": assignment_sha256,
    }


def monotonic_context_gate(degrees, context_available, slope):
    """Return (1 + degree)^(-slope), masked by H-C availability."""
    degrees = np.asarray(degrees, dtype=np.float64)
    available = np.asarray(context_available, dtype=np.float64)
    slope = float(slope)
    if degrees.shape != available.shape:
        raise ValueError("Support degrees and context availability must match.")
    if np.any(degrees < 0):
        raise ValueError("Support degrees cannot be negative.")
    if slope <= 0:
        raise ValueError("Support-router slope must be positive.")
    return available * np.exp(-slope * np.log1p(degrees))


def softplus(value):
    value = float(value)
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


def inverse_softplus(value):
    value = float(value)
    if value <= 0:
        raise ValueError("Softplus target must be positive.")
    return value + math.log(-math.expm1(-value))

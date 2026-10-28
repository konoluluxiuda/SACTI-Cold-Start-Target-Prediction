"""Load frozen target-cold and double-cold evaluation units."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def stable_int(seed, role, value):
    content = "%d|%s|%s" % (int(seed), str(role), str(value))
    return int(hashlib.sha256(content.encode("utf-8")).hexdigest()[:16], 16)


def record_lines(records):
    return [
        "%s\t%s\t%d" % (
            str(left_id), str(right_id), int(float(label) > 0)
        )
        for left_id, right_id, label in records
    ]


def records_sha256(records):
    content = "\n".join(sorted(record_lines(records))) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path):
    pairs = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(
                    "Invalid relation row %d in %s." % (line_number, path)
                )
            pairs.add((str(parts[0]), str(parts[1])))
    return pairs


def read_records(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    "Invalid test record row %d in %s." % (
                        line_number, path
                    )
                )
            record = [str(parts[0]), str(parts[1]), float(parts[2])]
            pair = (record[0], record[1])
            if pair in seen:
                raise ValueError("Duplicate test pair %s in %s." % (pair, path))
            seen.add(pair)
            records.append(record)
    return records


def read_groups(path, expected_entity_type):
    groups = {}
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        header = next(handle, "").strip().split("\t")
        if header != ["entity_type", "entity_id", "group"]:
            raise ValueError("Invalid entity group header in %s." % path)
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(
                    "Invalid entity group row %d in %s." % (
                        line_number, path
                    )
                )
            entity_type, entity_id, group_value = parts
            if entity_type != expected_entity_type:
                raise ValueError(
                    "Expected %s group in %s; found %s."
                    % (expected_entity_type, path, entity_type)
                )
            if entity_id in seen:
                raise ValueError(
                    "Entity %s appears in multiple groups in %s."
                    % (entity_id, path)
                )
            seen.add(entity_id)
            groups.setdefault(int(group_value), set()).add(entity_id)
    return groups


def _coprime_stride(size, seed_value):
    if size <= 1:
        return 1
    stride = 1 + seed_value % (size - 1)
    while math.gcd(stride, size) != 1:
        stride += 1
        if stride >= size:
            stride = 1
    return stride


def build_compound_matched_training_negatives(
        training_positive_edges, allowed_proteins, all_positive_pairs,
        seed, unit_key):
    """Build deterministic 1:1 negatives without materializing a Cartesian pool."""
    positives_by_compound = {}
    for compound_id, protein_id in training_positive_edges:
        positives_by_compound.setdefault(compound_id, set()).add(protein_id)
    allowed_proteins = sorted(str(value) for value in allowed_proteins)
    if not allowed_proteins:
        raise ValueError("Training negative generation has no allowed proteins.")

    negative_records = []
    for compound_id in sorted(positives_by_compound):
        required = len(positives_by_compound[compound_id])
        available = sum(
            (compound_id, protein_id) not in all_positive_pairs
            for protein_id in allowed_proteins
        )
        if available < required:
            raise ValueError(
                "Compound %s has %d allowed unobserved proteins; %d are "
                "required for unit %s."
                % (compound_id, available, required, unit_key)
            )
        base_seed = stable_int(
            seed, "training_negative", "%s|%s" % (unit_key, compound_id)
        )
        start = base_seed % len(allowed_proteins)
        stride = _coprime_stride(
            len(allowed_proteins),
            stable_int(
                seed,
                "training_negative_stride",
                "%s|%s" % (unit_key, compound_id),
            ),
        )
        selected = 0
        for offset in range(len(allowed_proteins)):
            protein_id = allowed_proteins[
                (start + offset * stride) % len(allowed_proteins)
            ]
            if (compound_id, protein_id) in all_positive_pairs:
                continue
            negative_records.append([compound_id, protein_id, 0.0])
            selected += 1
            if selected == required:
                break
        if selected != required:
            raise AssertionError(
                "Deterministic negative traversal ended before capacity."
            )
    return negative_records


def _ordered_entities(values, seed, role):
    return sorted(
        {str(value) for value in values},
        key=lambda value: (stable_int(seed, role, value), value),
    )


def _heldout_entities(values, ratio, seed, role):
    ordered = _ordered_entities(values, seed, role)
    if len(ordered) < 2:
        raise ValueError(
            "Support-state inner validation requires at least two %s."
            % role
        )
    heldout_count = int(round(len(ordered) * float(ratio)))
    heldout_count = max(1, min(len(ordered) - 1, heldout_count))
    return set(ordered[:heldout_count])


def _protein_matched_validation_negatives(
        validation_positive_edges, allowed_compounds, all_positive_pairs,
        seed, unit_key):
    positives_by_protein = {}
    for compound_id, protein_id in validation_positive_edges:
        positives_by_protein.setdefault(protein_id, set()).add(compound_id)
    allowed_compounds = sorted(str(value) for value in allowed_compounds)
    if not allowed_compounds:
        raise ValueError("Validation negative generation has no compounds.")

    negative_records = []
    for protein_id in sorted(positives_by_protein):
        required = len(positives_by_protein[protein_id])
        available = sum(
            (compound_id, protein_id) not in all_positive_pairs
            for compound_id in allowed_compounds
        )
        if available < required:
            raise ValueError(
                "Protein %s has %d allowed unobserved compounds; %d are "
                "required for %s."
                % (protein_id, available, required, unit_key)
            )
        base_seed = stable_int(
            seed, "validation_negative", "%s|%s" % (unit_key, protein_id)
        )
        start = base_seed % len(allowed_compounds)
        stride = _coprime_stride(
            len(allowed_compounds),
            stable_int(
                seed,
                "validation_negative_stride",
                "%s|%s" % (unit_key, protein_id),
            ),
        )
        selected = 0
        for offset in range(len(allowed_compounds)):
            compound_id = allowed_compounds[
                (start + offset * stride) % len(allowed_compounds)
            ]
            if (compound_id, protein_id) in all_positive_pairs:
                continue
            negative_records.append([compound_id, protein_id, 0.0])
            selected += 1
            if selected == required:
                break
    return negative_records


def _block_validation_negatives(
        required, heldout_compounds, heldout_proteins, all_positive_pairs,
        seed, unit_key):
    compound_ids = sorted(str(value) for value in heldout_compounds)
    protein_ids = sorted(str(value) for value in heldout_proteins)
    capacity = sum(
        (compound_id, protein_id) not in all_positive_pairs
        for compound_id in compound_ids
        for protein_id in protein_ids
    )
    if capacity < required:
        raise ValueError(
            "Double-cold validation block has %d unobserved pairs; %d are "
            "required for %s." % (capacity, required, unit_key)
        )

    candidates = [
        (compound_id, protein_id)
        for compound_id in compound_ids
        for protein_id in protein_ids
        if (compound_id, protein_id) not in all_positive_pairs
    ]
    candidates.sort(
        key=lambda pair: (
            stable_int(
                seed, "validation_negative",
                "%s|%s|%s" % (unit_key, pair[0], pair[1]),
            ),
            pair,
        )
    )
    return [
        [compound_id, protein_id, 0.0]
        for compound_id, protein_id in candidates[:required]
    ]


def _rectangle_negatives(
        required, allowed_compounds, allowed_proteins, all_positive_pairs,
        seed, unit_key, forbidden_pairs=None):
    """Sample deterministic negatives without materializing the full rectangle."""
    compound_ids = sorted(str(value) for value in allowed_compounds)
    protein_ids = sorted(str(value) for value in allowed_proteins)
    forbidden_pairs = {
        (str(left), str(right))
        for left, right in (forbidden_pairs or set())
    }
    if not compound_ids or not protein_ids:
        raise ValueError(
            "Negative rectangle for %s has an empty endpoint set." % unit_key
        )
    rectangle_size = len(compound_ids) * len(protein_ids)
    start = stable_int(seed, "rectangle_start", unit_key) % rectangle_size
    stride = _coprime_stride(
        rectangle_size,
        stable_int(seed, "rectangle_stride", unit_key),
    )
    selected = []
    for offset in range(rectangle_size):
        flat_index = (start + offset * stride) % rectangle_size
        compound_id = compound_ids[flat_index // len(protein_ids)]
        protein_id = protein_ids[flat_index % len(protein_ids)]
        if (compound_id, protein_id) in all_positive_pairs:
            continue
        if (compound_id, protein_id) in forbidden_pairs:
            continue
        selected.append([compound_id, protein_id, 0.0])
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(
            "Negative rectangle for %s has %d candidates; %d are required."
            % (unit_key, len(selected), required)
        )
    return selected


def _entity_preserving_pair_holdout(edges, ratio, seed, unit_key):
    """Hold out warm-warm edges while retaining every endpoint in training."""
    edges = {(str(left), str(right)) for left, right in edges}
    if len(edges) < 2:
        raise ValueError("Warm-warm holdout requires at least two edges.")
    desired = max(1, int(round(len(edges) * float(ratio))))
    compound_degree = Counter(left for left, _ in edges)
    protein_degree = Counter(right for _, right in edges)
    ordered = sorted(
        edges,
        key=lambda edge: (
            stable_int(
                seed,
                "warm_pair_holdout",
                "%s|%s|%s" % (unit_key, edge[0], edge[1]),
            ),
            edge,
        ),
    )
    heldout = set()
    for edge in ordered:
        compound_id, protein_id = edge
        if compound_degree[compound_id] <= 1:
            continue
        if protein_degree[protein_id] <= 1:
            continue
        heldout.add(edge)
        compound_degree[compound_id] -= 1
        protein_degree[protein_id] -= 1
        if len(heldout) == desired:
            break
    if not heldout:
        raise ValueError(
            "No entity-preserving warm-warm edge can be held out for %s."
            % unit_key
        )
    return edges - heldout, heldout


def build_four_state_support_unit(
        manifest_path, compound_group, protein_group,
        warm_holdout_ratio=0.1, seed=None):
    """Build one shared training graph and four disjoint support-state tests."""
    warm_holdout_ratio = float(warm_holdout_ratio)
    if not 0.0 < warm_holdout_ratio < 1.0:
        raise ValueError("warm_holdout_ratio must be between 0 and 1.")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError("Four-state support unit requires manifest version 2.")
    if manifest.get("protocol") != "support_complete_cold_start":
        raise ValueError("Unexpected support-complete protocol.")
    _verify_manifest_files(manifest_path, manifest)
    root = manifest_path.parent
    seed = int(manifest["seed"] if seed is None else seed)
    unit_key = "four_state_c%d_p%d" % (
        int(compound_group), int(protein_group)
    )

    all_positive_pairs = read_pairs(manifest["sources"]["C_P"]["path"])
    compound_groups = read_groups(
        root / manifest["artifacts"][
            "double_cold_compound_groups"
        ]["path"],
        "compound",
    )
    protein_groups = read_groups(
        root / manifest["artifacts"]["cold_target_groups"]["path"],
        "protein",
    )
    cold_compounds = set(compound_groups[int(compound_group)])
    cold_proteins = set(protein_groups[int(protein_group)])

    warm_warm_pool = {
        edge for edge in all_positive_pairs
        if edge[0] not in cold_compounds and edge[1] not in cold_proteins
    }
    training_positive_edges, warm_warm_positive_edges = (
        _entity_preserving_pair_holdout(
            warm_warm_pool,
            warm_holdout_ratio,
            seed,
            unit_key,
        )
    )
    warm_compounds = {left for left, _ in training_positive_edges}
    warm_proteins = {right for _, right in training_positive_edges}

    state_positive_edges = {
        "warm_warm": warm_warm_positive_edges,
        "cold_warm": {
            edge for edge in all_positive_pairs
            if edge[0] in cold_compounds and edge[1] in warm_proteins
        },
        "warm_cold": {
            edge for edge in all_positive_pairs
            if edge[0] in warm_compounds and edge[1] in cold_proteins
        },
        "cold_cold": {
            edge for edge in all_positive_pairs
            if edge[0] in cold_compounds and edge[1] in cold_proteins
        },
    }
    empty_states = [
        state for state, edges in state_positive_edges.items() if not edges
    ]
    if empty_states:
        raise ValueError(
            "Four-state unit %s has no positives for: %s."
            % (unit_key, ", ".join(empty_states))
        )

    training_negative_records = build_compound_matched_training_negatives(
        training_positive_edges,
        warm_proteins,
        all_positive_pairs,
        seed,
        unit_key,
    )
    training_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in sorted(training_positive_edges)
    ]
    training_records = (
        training_positive_records + training_negative_records
    )

    state_endpoints = {
        "warm_warm": (warm_compounds, warm_proteins),
        "cold_warm": (cold_compounds, warm_proteins),
        "warm_cold": (warm_compounds, cold_proteins),
        "cold_cold": (cold_compounds, cold_proteins),
    }
    state_records = {}
    state_metadata = {}
    seen_test_pairs = set()
    train_pairs = {(row[0], row[1]) for row in training_records}
    for state in (
            "warm_warm", "cold_warm", "warm_cold", "cold_cold"):
        positive_records = [
            [compound_id, protein_id, 1.0]
            for compound_id, protein_id in sorted(
                state_positive_edges[state]
            )
        ]
        negative_records = _rectangle_negatives(
            len(positive_records),
            state_endpoints[state][0],
            state_endpoints[state][1],
            all_positive_pairs,
            seed,
            "%s|%s" % (unit_key, state),
            forbidden_pairs=train_pairs | seen_test_pairs,
        )
        records = positive_records + negative_records
        pairs = {(row[0], row[1]) for row in records}
        if train_pairs & pairs:
            raise ValueError(
                "Four-state %s test overlaps training." % state
            )
        if seen_test_pairs & pairs:
            raise ValueError(
                "Four-state test pair appears in multiple states."
            )
        if any(
                float(row[2]) <= 0
                and (row[0], row[1]) in all_positive_pairs
                for row in records):
            raise ValueError(
                "Four-state %s sampled a known positive as negative." % state
            )
        seen_test_pairs.update(pairs)
        state_records[state] = records
        state_metadata[state] = {
            "positive_count": len(positive_records),
            "negative_count": len(negative_records),
            "records_sha256": records_sha256(records),
        }

    assignment_lines = [
        "%s\t%s\t%d\t%s" % (
            row[0], row[1], int(float(row[2]) > 0), partition
        )
        for partition, records in (
            [("train", training_records)]
            + [
                ("test_%s" % state, state_records[state])
                for state in (
                    "warm_warm", "cold_warm",
                    "warm_cold", "cold_cold",
                )
            ]
        )
        for row in records
    ]
    metadata = {
        "strategy": "support_complete_four_state",
        "unit_key": unit_key,
        "seed": seed,
        "warm_holdout_ratio": warm_holdout_ratio,
        "heldout_compounds": len(cold_compounds),
        "heldout_proteins": len(cold_proteins),
        "warm_compounds": len(warm_compounds),
        "warm_proteins": len(warm_proteins),
        "training_positive_count": len(training_positive_records),
        "training_negative_count": len(training_negative_records),
        "training_records_sha256": records_sha256(training_records),
        "states": state_metadata,
        "assignments_sha256": hashlib.sha256(
            ("\n".join(sorted(assignment_lines)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return training_records, state_records, metadata


def build_four_state_inner_validation(
        outer_training_records, all_positive_pairs, ratio, seed, unit_key):
    """Build four support-state validation sets from one outer training graph."""
    ratio = float(ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            "Four-state inner validation ratio must be between 0 and 1."
        )
    all_positive_pairs = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id in all_positive_pairs
    }
    outer_positive_edges = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id, label in outer_training_records
        if float(label) > 0
    }
    if len(outer_positive_edges) < 2:
        raise ValueError(
            "Four-state inner validation requires at least two positives."
        )

    outer_compounds = {edge[0] for edge in outer_positive_edges}
    outer_proteins = {edge[1] for edge in outer_positive_edges}
    heldout_compounds = _heldout_entities(
        outer_compounds, ratio, seed, "four_state_compound"
    )
    heldout_proteins = _heldout_entities(
        outer_proteins, ratio, seed, "four_state_protein"
    )
    inner_key = "%s|inner_four_state" % unit_key
    warm_warm_pool = {
        edge for edge in outer_positive_edges
        if edge[0] not in heldout_compounds
        and edge[1] not in heldout_proteins
    }
    inner_positive_edges, warm_warm_positive_edges = (
        _entity_preserving_pair_holdout(
            warm_warm_pool,
            ratio,
            seed,
            inner_key,
        )
    )
    warm_compounds = {left for left, _ in inner_positive_edges}
    warm_proteins = {right for _, right in inner_positive_edges}
    state_positive_edges = {
        "warm_warm": warm_warm_positive_edges,
        "cold_warm": {
            edge for edge in outer_positive_edges
            if edge[0] in heldout_compounds and edge[1] in warm_proteins
        },
        "warm_cold": {
            edge for edge in outer_positive_edges
            if edge[0] in warm_compounds and edge[1] in heldout_proteins
        },
        "cold_cold": {
            edge for edge in outer_positive_edges
            if edge[0] in heldout_compounds
            and edge[1] in heldout_proteins
        },
    }
    empty_states = [
        state for state, edges in state_positive_edges.items() if not edges
    ]
    if empty_states:
        raise ValueError(
            "Four-state inner validation for %s has no positives for: %s."
            % (unit_key, ", ".join(empty_states))
        )

    inner_negative_records = build_compound_matched_training_negatives(
        inner_positive_edges,
        warm_proteins,
        all_positive_pairs,
        seed,
        inner_key,
    )
    inner_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in sorted(inner_positive_edges)
    ]
    inner_records = inner_positive_records + inner_negative_records
    inner_pairs = {(row[0], row[1]) for row in inner_records}
    state_endpoints = {
        "warm_warm": (warm_compounds, warm_proteins),
        "cold_warm": (heldout_compounds, warm_proteins),
        "warm_cold": (warm_compounds, heldout_proteins),
        "cold_cold": (heldout_compounds, heldout_proteins),
    }
    state_records = {}
    state_metadata = {}
    seen_validation_pairs = set()
    state_order = (
        "warm_warm", "cold_warm", "warm_cold", "cold_cold"
    )
    for state in state_order:
        positive_records = [
            [compound_id, protein_id, 1.0]
            for compound_id, protein_id in sorted(
                state_positive_edges[state]
            )
        ]
        negative_records = _rectangle_negatives(
            len(positive_records),
            state_endpoints[state][0],
            state_endpoints[state][1],
            all_positive_pairs,
            seed,
            "%s|%s" % (inner_key, state),
            forbidden_pairs=inner_pairs | seen_validation_pairs,
        )
        records = positive_records + negative_records
        pairs = {(row[0], row[1]) for row in records}
        if inner_pairs & pairs:
            raise ValueError(
                "Four-state inner %s validation overlaps training." % state
            )
        if seen_validation_pairs & pairs:
            raise ValueError(
                "Four-state inner validation pair appears in multiple states."
            )
        seen_validation_pairs.update(pairs)
        state_records[state] = records
        state_metadata[state] = {
            "positive_count": len(positive_records),
            "negative_count": len(negative_records),
            "records_sha256": records_sha256(records),
        }

    assigned_positive_edges = (
        inner_positive_edges
        | set().union(*state_positive_edges.values())
    )
    discarded_positive_count = len(
        outer_positive_edges - assigned_positive_edges
    )
    assignment_lines = [
        "%s\t%s\t%d\t%s" % (
            row[0], row[1], int(float(row[2]) > 0), partition
        )
        for partition, records in (
            [("train", inner_records)]
            + [
                ("validation_%s" % state, state_records[state])
                for state in state_order
            ]
        )
        for row in records
    ]
    metadata = {
        "strategy": "support_complete_four_state_inner",
        "unit_key": str(unit_key),
        "seed": int(seed),
        "ratio": ratio,
        "heldout_compounds": len(heldout_compounds),
        "heldout_proteins": len(heldout_proteins),
        "warm_compounds": len(warm_compounds),
        "warm_proteins": len(warm_proteins),
        "inner_train_positive_count": len(inner_positive_records),
        "inner_train_negative_count": len(inner_negative_records),
        "inner_train_records_sha256": records_sha256(inner_records),
        "discarded_buffer_positive_count": discarded_positive_count,
        "states": state_metadata,
        "assignments_sha256": hashlib.sha256(
            ("\n".join(sorted(assignment_lines)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return inner_records, state_records, metadata


def _support_inner_metadata(
        mode, ratio, seed, inner_positive_edges, inner_negative_records,
        validation_positive_edges, validation_negative_records,
        heldout_compounds, heldout_proteins, discarded_positive_count):
    inner_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in inner_positive_edges
    ]
    validation_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in validation_positive_edges
    ]
    inner_records = inner_positive_records + inner_negative_records
    validation_records = (
        validation_positive_records + validation_negative_records
    )
    train_pairs = {(row[0], row[1]) for row in inner_records}
    validation_pairs = {(row[0], row[1]) for row in validation_records}
    if train_pairs & validation_pairs:
        raise ValueError(
            "Support-state inner train and validation pairs overlap."
        )
    assignment_lines = [
        "%s\t%s\t%d\t%s"
        % (row[0], row[1], int(float(row[2]) > 0), partition)
        for partition, records in (
            ("train", inner_records),
            ("validation", validation_records),
        )
        for row in records
    ]
    assignment_hash = hashlib.sha256(
        ("\n".join(sorted(assignment_lines)) + "\n").encode("utf-8")
    ).hexdigest()
    metadata = {
        "strategy": "support_complete_%s_inner" % mode,
        "mode": mode,
        "seed": int(seed),
        "ratio": float(ratio),
        "inner_train_records": len(inner_records),
        "inner_train_positive_count": len(inner_positive_records),
        "inner_train_negative_count": len(inner_negative_records),
        "validation_records": len(validation_records),
        "validation_positive_count": len(validation_positive_records),
        "validation_negative_count": len(validation_negative_records),
        "heldout_compounds": len(heldout_compounds),
        "heldout_proteins": len(heldout_proteins),
        "discarded_buffer_positive_count": int(discarded_positive_count),
        "assignments_sha256": assignment_hash,
        "inner_train_records_sha256": records_sha256(inner_records),
        "validation_records_sha256": records_sha256(validation_records),
    }
    return inner_records, validation_records, metadata


def build_support_state_inner_validation(
        outer_training_records, all_positive_pairs, mode, ratio, seed,
        unit_key):
    """Build deterministic inner validation with the outer support state."""
    ratio = float(ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            "Support-state inner validation ratio must be between 0 and 1."
        )
    all_positive_pairs = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id in all_positive_pairs
    }
    outer_positive_edges = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id, label in outer_training_records
        if float(label) > 0
    }
    if len(outer_positive_edges) < 2:
        raise ValueError(
            "Support-state inner validation requires at least two positives."
        )
    outer_compounds = {edge[0] for edge in outer_positive_edges}
    outer_proteins = {edge[1] for edge in outer_positive_edges}
    inner_key = "%s|inner|%s" % (unit_key, mode)

    if mode == "target_cold":
        heldout_proteins = _heldout_entities(
            outer_proteins, ratio, seed, "protein"
        )
        heldout_compounds = set()
        inner_positive_edges = {
            edge for edge in outer_positive_edges
            if edge[1] not in heldout_proteins
        }
        inner_compounds = {edge[0] for edge in inner_positive_edges}
        validation_positive_edges = {
            edge for edge in outer_positive_edges
            if edge[1] in heldout_proteins and edge[0] in inner_compounds
        }
        discarded_positive_count = (
            len(outer_positive_edges)
            - len(inner_positive_edges)
            - len(validation_positive_edges)
        )
        validation_negative_records = (
            _protein_matched_validation_negatives(
                validation_positive_edges,
                inner_compounds,
                all_positive_pairs,
                seed,
                inner_key,
            )
        )
    elif mode == "double_cold":
        heldout_compounds = _heldout_entities(
            outer_compounds, ratio, seed, "compound"
        )
        heldout_proteins = _heldout_entities(
            outer_proteins, ratio, seed, "protein"
        )
        inner_positive_edges = {
            edge for edge in outer_positive_edges
            if edge[0] not in heldout_compounds
            and edge[1] not in heldout_proteins
        }
        validation_positive_edges = {
            edge for edge in outer_positive_edges
            if edge[0] in heldout_compounds
            and edge[1] in heldout_proteins
        }
        discarded_positive_count = (
            len(outer_positive_edges)
            - len(inner_positive_edges)
            - len(validation_positive_edges)
        )
        validation_negative_records = _block_validation_negatives(
            len(validation_positive_edges),
            heldout_compounds,
            heldout_proteins,
            all_positive_pairs,
            seed,
            inner_key,
        )
    else:
        raise ValueError(
            "Support-state inner mode must be target_cold or double_cold."
        )

    if not inner_positive_edges:
        raise ValueError(
            "Support-state inner split contains no training positives."
        )
    if not validation_positive_edges:
        raise ValueError(
            "Support-state inner split contains no validation positives. "
            "Use a larger validation.ratio or a different validation.seed."
        )
    inner_allowed_proteins = (
        outer_proteins - heldout_proteins
    )
    inner_negative_records = build_compound_matched_training_negatives(
        inner_positive_edges,
        inner_allowed_proteins,
        all_positive_pairs,
        seed,
        inner_key,
    )
    return _support_inner_metadata(
        mode,
        ratio,
        seed,
        inner_positive_edges,
        inner_negative_records,
        validation_positive_edges,
        validation_negative_records,
        heldout_compounds,
        heldout_proteins,
        discarded_positive_count,
    )


def _verify_manifest_files(manifest_path, manifest):
    root = Path(manifest_path).resolve().parent
    for relation, metadata in manifest["sources"].items():
        path = Path(metadata["path"])
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(
                "Support-complete source file changed: %s." % relation
            )
    for name, metadata in manifest["artifacts"].items():
        path = root / metadata["path"]
        if not path.exists() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(
                "Support-complete artifact changed: %s." % name
            )


def _unit_entry(manifest, mode, fold, compound_group, protein_group):
    if mode == "target_cold":
        if fold is None:
            raise ValueError("target_cold requires fold.")
        matches = [
            row for row in manifest["target_cold"]["folds"]
            if int(row["fold"]) == int(fold)
        ]
    elif mode == "double_cold":
        if compound_group is None or protein_group is None:
            raise ValueError(
                "double_cold requires compound_group and protein_group."
            )
        matches = [
            row for row in manifest["double_cold"]["cells"]
            if int(row["compound_group"]) == int(compound_group)
            and int(row["protein_group"]) == int(protein_group)
        ]
    else:
        raise ValueError(
            "Unsupported support-complete mode %r." % mode
        )
    if len(matches) != 1:
        raise ValueError(
            "Expected one manifest entry for %s; found %d."
            % (mode, len(matches))
        )
    return matches[0]


def load_support_complete_unit(
        manifest_path, mode, fold=None,
        compound_group=None, protein_group=None):
    """Reconstruct one explicit train/test unit and verify all frozen hashes."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError(
            "Support-complete loader requires manifest version 2."
        )
    if manifest.get("protocol") != "support_complete_cold_start":
        raise ValueError("Unexpected support-complete protocol.")
    _verify_manifest_files(manifest_path, manifest)
    root = manifest_path.parent
    entry = _unit_entry(
        manifest, mode, fold, compound_group, protein_group
    )
    cp_edges = read_pairs(manifest["sources"]["C_P"]["path"])
    cp_compounds = {compound for compound, _ in cp_edges}
    cp_proteins = {protein for _, protein in cp_edges}
    protein_groups = read_groups(
        root / manifest["artifacts"]["cold_target_groups"]["path"],
        "protein",
    )

    if mode == "target_cold":
        heldout_proteins = protein_groups[int(fold)]
        heldout_compounds = set()
        training_positive_edges = {
            edge for edge in cp_edges
            if edge[1] not in heldout_proteins
        }
        unit_key = "target_fold_%d" % int(fold)
    else:
        compound_groups = read_groups(
            root / manifest["artifacts"][
                "double_cold_compound_groups"
            ]["path"],
            "compound",
        )
        heldout_compounds = compound_groups[int(compound_group)]
        heldout_proteins = protein_groups[int(protein_group)]
        training_positive_edges = {
            edge for edge in cp_edges
            if edge[0] not in heldout_compounds
            and edge[1] not in heldout_proteins
        }
        unit_key = "double_c%d_p%d" % (
            int(compound_group), int(protein_group)
        )

    training_positive_records = [
        [compound_id, protein_id, 1.0]
        for compound_id, protein_id in training_positive_edges
    ]
    if (
        len(training_positive_records) != entry["training_positive_count"]
        or records_sha256(training_positive_records)
        != entry["training_positives_sha256"]
    ):
        raise ValueError(
            "Reconstructed training positives do not match manifest for %s."
            % unit_key
        )
    allowed_proteins = cp_proteins - heldout_proteins
    training_negative_records = build_compound_matched_training_negatives(
        training_positive_edges,
        allowed_proteins,
        cp_edges,
        manifest["seed"],
        unit_key,
    )
    if (
        len(training_negative_records) != entry["training_negative_count"]
        or records_sha256(training_negative_records)
        != entry["training_negatives_sha256"]
    ):
        raise ValueError(
            "Reconstructed training negatives do not match manifest for %s."
            % unit_key
        )

    test_path = root / entry["test_path"]
    test_records = read_records(test_path)
    if records_sha256(test_records) != entry["test_records_sha256"]:
        raise ValueError(
            "Loaded test records do not match manifest for %s." % unit_key
        )
    test_positive_count = sum(float(row[2]) > 0 for row in test_records)
    test_negative_count = len(test_records) - test_positive_count
    if (
        test_positive_count != entry["test_positive_count"]
        or test_negative_count != entry["test_negative_count"]
    ):
        raise ValueError("Test label counts do not match manifest.")

    training_records = training_positive_records + training_negative_records
    train_pairs = {(row[0], row[1]) for row in training_records}
    test_pairs = {(row[0], row[1]) for row in test_records}
    if train_pairs & test_pairs:
        raise ValueError("Support-complete unit contains train/test pair overlap.")
    if any(
            float(row[2]) <= 0 and (row[0], row[1]) in cp_edges
            for row in training_records + test_records):
        raise ValueError("A sampled negative is a known C-P positive.")

    train_compounds = {row[0] for row in training_records}
    train_proteins = {row[1] for row in training_records}
    if train_proteins & heldout_proteins:
        raise ValueError("Held-out protein leaked into training records.")
    if mode == "target_cold":
        if any(row[1] not in heldout_proteins for row in test_records):
            raise ValueError("Target-cold test contains a non-held-out protein.")
        positive_test_compounds = {
            row[0] for row in test_records if float(row[2]) > 0
        }
        if not positive_test_compounds <= train_compounds:
            raise ValueError(
                "Target-cold test contains a cold compound positive."
            )
    else:
        if train_compounds & heldout_compounds:
            raise ValueError("Held-out compound leaked into training records.")
        if any(
                row[0] not in heldout_compounds
                or row[1] not in heldout_proteins
                for row in test_records):
            raise ValueError(
                "Double-cold test contains an endpoint outside its cell."
            )

    metadata = {
        "mode": mode,
        "unit_key": unit_key,
        "seed": int(manifest["seed"]),
        "training_positive_count": len(training_positive_records),
        "training_negative_count": len(training_negative_records),
        "test_positive_count": test_positive_count,
        "test_negative_count": test_negative_count,
        "heldout_compounds": len(heldout_compounds),
        "heldout_proteins": len(heldout_proteins),
        "train_test_pair_overlap": 0,
        "train_test_compound_overlap": len(
            train_compounds & heldout_compounds
        ),
        "train_test_protein_overlap": len(
            train_proteins & heldout_proteins
        ),
        "training_positives_sha256": records_sha256(
            training_positive_records
        ),
        "training_negatives_sha256": records_sha256(
            training_negative_records
        ),
        "test_records_sha256": records_sha256(test_records),
    }
    return training_records, test_records, metadata


def load_four_state_support_artifact(manifest_path):
    """Load and verify one frozen shared four-state support artifact."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Four-state artifact loader requires version 1.")
    if manifest.get("protocol") != "support_complete_four_state":
        raise ValueError("Unexpected four-state artifact protocol.")

    source_manifest = Path(
        manifest["source_manifest"]["path"]
    ).expanduser().resolve()
    if (
        not source_manifest.exists()
        or sha256_file(source_manifest)
        != manifest["source_manifest"]["sha256"]
    ):
        raise ValueError("Four-state source manifest changed.")

    root = manifest_path.parent
    loaded = {}
    for name, artifact in manifest["artifacts"].items():
        path = root / artifact["path"]
        if not path.exists() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(
                "Four-state artifact changed: %s." % name
            )
        records = read_records(path)
        if len(records) != int(artifact["records"]):
            raise ValueError(
                "Four-state artifact count changed: %s." % name
            )
        if records_sha256(records) != artifact["records_sha256"]:
            raise ValueError(
                "Four-state record hash changed: %s." % name
            )
        loaded[name] = records

    expected = {
        "training",
        "test_warm_warm",
        "test_cold_warm",
        "test_warm_cold",
        "test_cold_cold",
    }
    if set(loaded) != expected:
        raise ValueError(
            "Four-state artifact set is incomplete: %s." %
            sorted(loaded)
        )
    states = {
        state: loaded["test_%s" % state]
        for state in (
            "warm_warm", "cold_warm", "warm_cold", "cold_cold"
        )
    }
    return loaded["training"], states, manifest["metadata"]

import hashlib
from collections import defaultdict

import numpy as np


def _stable_seed(base_seed, compound_id):
    digest = hashlib.sha256(str(compound_id).encode('utf-8')).digest()
    offset = int.from_bytes(digest[:8], byteorder='big', signed=False)
    return (int(base_seed) + offset) % (2 ** 32)


def _allocate_nested_quotas(positive_counts, capacities, ratios, seed):
    compounds = sorted(positive_counts)
    total_positives = sum(positive_counts.values())
    if sum(capacities.values()) < max(ratios) * total_positives:
        raise ValueError(
            'The global unlabeled pool cannot satisfy the maximum ratio.')
    previous = {compound: 0 for compound in compounds}
    allocations = {}
    for ratio in ratios:
        target = ratio * total_positives
        quotas = {
            compound: min(
                capacities[compound],
                max(previous[compound], ratio * positive_counts[compound]))
            for compound in compounds
        }
        deficit = target - sum(quotas.values())
        if deficit < 0:
            raise ValueError('Nested quota allocation exceeded its target.')
        remaining = {
            compound: capacities[compound] - quotas[compound]
            for compound in compounds
        }
        while deficit:
            available = {
                compound: value for compound, value in remaining.items()
                if value > 0
            }
            total_available = sum(available.values())
            if total_available < deficit:
                raise ValueError('Insufficient capacity during quota redistribution.')
            additions = {}
            fractions = []
            assigned = 0
            for compound, capacity in available.items():
                exact = deficit * capacity / float(total_available)
                addition = min(capacity, int(np.floor(exact)))
                additions[compound] = addition
                assigned += addition
                fractions.append((
                    exact - addition,
                    _stable_seed(seed + ratio, compound),
                    compound,
                ))
            remainder = deficit - assigned
            for _, _, compound in sorted(fractions, reverse=True):
                if remainder <= 0:
                    break
                if additions[compound] < available[compound]:
                    additions[compound] += 1
                    remainder -= 1
            if not any(additions.values()):
                raise ValueError('Quota redistribution made no progress.')
            for compound, addition in additions.items():
                quotas[compound] += addition
                remaining[compound] -= addition
                deficit -= addition
        if sum(quotas.values()) != target:
            raise ValueError('Quota allocation does not match the global target.')
        if any(quotas[value] < previous[value] for value in compounds):
            raise ValueError('Quota allocation is not nested.')
        allocations[ratio] = quotas
        previous = quotas
    return allocations


def build_nested_negative_ratio_pairs(
        protein_ids, train_records, test_records, ratios=(1, 5, 10), seed=0):
    """Build deterministic nested unlabeled sets for each test compound."""
    protein_ids = sorted({str(value) for value in protein_ids})
    ratios = tuple(sorted({int(value) for value in ratios}))
    if not ratios or any(value <= 0 for value in ratios):
        raise ValueError('Negative ratios must be positive integers.')
    train_positives = _positive_targets_by_compound(train_records)
    test_positives = _positive_targets_by_compound(test_records)
    if not test_positives:
        raise ValueError('No held-out positive pairs are available.')
    pools = {}
    positive_counts = {}
    capacities = {}
    for compound_id in sorted(test_positives):
        positives = sorted(test_positives[compound_id])
        excluded = train_positives.get(compound_id, set()) | set(positives)
        pool = [value for value in protein_ids if value not in excluded]
        pools[compound_id] = pool
        positive_counts[compound_id] = len(positives)
        capacities[compound_id] = len(pool)
    allocations = _allocate_nested_quotas(
        positive_counts, capacities, ratios, seed)
    selections = {ratio: [] for ratio in ratios}
    for compound_id in sorted(test_positives):
        pool = pools[compound_id]
        required = allocations[max(ratios)][compound_id]
        rng = np.random.RandomState(_stable_seed(seed, compound_id))
        order = rng.permutation(len(pool))[:required]
        sampled = [pool[index] for index in order]
        for ratio in ratios:
            selections[ratio].extend(
                (compound_id, protein_id)
                for protein_id in sampled[:allocations[ratio][compound_id]])
    return {
        'ratios': ratios,
        'positive_pairs': sorted(positive_pairs(test_records)),
        'negative_pairs': selections,
        'allocation': {
            'mode': 'positive_proportional_with_capacity_redistribution',
            'quotas': allocations,
        },
        'nested': all(
            set(selections[left]).issubset(set(selections[right]))
            for left, right in zip(ratios, ratios[1:])
        ),
    }


def evaluate_negative_ratio_sensitivity(
        protein_ids, train_records, test_records, score_pairs,
        ratios=(1, 5, 10), seed=0):
    """Score fixed positives against deterministic nested unlabeled samples."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    samples = build_nested_negative_ratio_pairs(
        protein_ids, train_records, test_records, ratios=ratios, seed=seed)
    positives = samples['positive_pairs']
    positive_compounds = [pair[0] for pair in positives]
    positive_proteins = [pair[1] for pair in positives]
    positive_scores = np.asarray(
        score_pairs(positive_compounds, positive_proteins),
        dtype=np.float64).reshape(-1)
    if positive_scores.size != len(positives):
        raise ValueError('Positive scorer output size mismatch.')
    if not np.all(np.isfinite(positive_scores)):
        raise ValueError('Positive scores contain NaN or infinity.')
    rows = []
    previous_negatives = set()
    for ratio in samples['ratios']:
        negatives = samples['negative_pairs'][ratio]
        compounds = [pair[0] for pair in negatives]
        proteins = [pair[1] for pair in negatives]
        negative_scores = np.asarray(
            score_pairs(compounds, proteins), dtype=np.float64).reshape(-1)
        if negative_scores.size != len(negatives):
            raise ValueError('Negative scorer output size mismatch.')
        if not np.all(np.isfinite(negative_scores)):
            raise ValueError('Negative scores contain NaN or infinity.')
        logits = np.concatenate([positive_scores, negative_scores])
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        labels = np.concatenate([
            np.ones(len(positives), dtype=np.int8),
            np.zeros(len(negatives), dtype=np.int8),
        ])
        prevalence = len(positives) / float(len(labels))
        aupr = float(average_precision_score(labels, scores))
        negative_lines = [
            '%s\t%s' % pair for pair in sorted(negatives)
        ]
        current_negatives = set(negatives)
        rows.append({
            'negative_ratio': ratio,
            'records': len(labels),
            'positives': len(positives),
            'negatives': len(negatives),
            'prevalence': prevalence,
            'AUC': float(roc_auc_score(labels, scores)),
            'AUPR': aupr,
            'AUPR_minus_prevalence': aupr - prevalence,
            'AUPR_lift_over_prevalence': aupr / prevalence,
            'negative_pairs_sha256': hashlib.sha256(
                ('\n'.join(negative_lines) + '\n').encode('utf-8')
            ).hexdigest(),
            'contains_previous_ratio_negatives': (
                previous_negatives.issubset(current_negatives)),
        })
        previous_negatives = current_negatives
    return {
        'protocol': 'deterministic_nested_unlabeled_ratio_sensitivity_v2',
        'sample_seed': int(seed),
        'nested_negative_sets': samples['nested'],
        'positive_pairs': len(positives),
        'positive_pairs_sha256': hashlib.sha256(
            ('\n'.join('%s\t%s' % pair for pair in positives) + '\n').encode('utf-8')
        ).hexdigest(),
        'candidate_proteins_sha256': hashlib.sha256(
            ('\n'.join(sorted({str(value) for value in protein_ids})) + '\n').encode('utf-8')
        ).hexdigest(),
        'rows': rows,
    }


def positive_pairs(records):
    return {
        (str(left_id), str(right_id))
        for left_id, right_id, label in records
        if float(label) > 0
    }


def _positive_targets_by_compound(records):
    targets = defaultdict(set)
    for compound_id, protein_id, label in records:
        if float(label) > 0:
            targets[str(compound_id)].add(str(protein_id))
    return targets


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {'min': 0.0, 'median': 0.0, 'mean': 0.0, 'max': 0.0}
    return {
        'min': float(np.min(array)),
        'median': float(np.median(array)),
        'mean': float(np.mean(array)),
        'max': float(np.max(array)),
    }


def evaluate_fixed_candidate_ranking(
        protein_ids,
        outer_train_records,
        outer_test_records,
        score_pairs,
        ks=(10, 20, 50),
        export_top=20,
        export_compound_ids=None):
    """Evaluate per-compound ranking over the complete fixed protein universe.

    Known positives from the outer training fold are filtered. Held-out outer-test
    positives remain in the candidate set. Every other pair is treated as
    unlabeled, not as a verified biological negative.
    """
    protein_ids = [str(protein_id) for protein_id in protein_ids]
    if len(protein_ids) != len(set(protein_ids)):
        raise ValueError('The protein candidate universe contains duplicate IDs.')
    if not protein_ids:
        raise ValueError('The protein candidate universe is empty.')

    ks = tuple(sorted({int(k) for k in ks}))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError('Ranking cutoffs must be positive integers.')
    export_top = int(export_top)
    if export_top < 0:
        raise ValueError('export_top must be non-negative.')
    export_compound_ids = (
        None if export_compound_ids is None else
        {str(value) for value in export_compound_ids}
    )

    train_positives = _positive_targets_by_compound(outer_train_records)
    test_positives = _positive_targets_by_compound(outer_test_records)
    if export_compound_ids is not None:
        missing_export_compounds = export_compound_ids - set(test_positives)
        if missing_export_compounds:
            raise ValueError(
                'Requested export compounds are absent from the evaluation '
                'positive set: %s.' % sorted(missing_export_compounds)
            )
    sampled_test_negatives = {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id, label in outer_test_records
        if float(label) <= 0
    }
    overlap = positive_pairs(outer_train_records) & positive_pairs(outer_test_records)
    if overlap:
        raise ValueError('Outer train/test positive pairs overlap: %d.' % len(overlap))

    protein_universe = set(protein_ids)
    missing_test_proteins = {
        protein_id
        for targets in test_positives.values()
        for protein_id in targets
        if protein_id not in protein_universe
    }
    if missing_test_proteins:
        raise ValueError(
            'Held-out positives contain %d proteins outside the candidate universe.' %
            len(missing_test_proteins)
        )

    per_compound = []
    top_candidates = []
    macro = {
        'precision': {k: [] for k in ks},
        'recall': {k: [] for k in ks},
        'hits': {k: [] for k in ks},
    }
    micro_hits = {k: 0 for k in ks}
    micro_slots = {k: 0 for k in ks}
    reciprocal_ranks = []
    first_positive_ranks = []
    candidate_counts = []
    total_test_positives = 0

    for compound_id in sorted(test_positives):
        held_out = test_positives[compound_id]
        filtered = train_positives.get(compound_id, set())
        candidates = [
            protein_id for protein_id in protein_ids
            if protein_id not in filtered
        ]
        missing = held_out - set(candidates)
        if missing:
            raise ValueError(
                'Filtering removed %d held-out positives for compound %s.' %
                (len(missing), compound_id)
            )

        scores = np.asarray(
            score_pairs([compound_id] * len(candidates), candidates),
            dtype=np.float64,
        ).reshape(-1)
        if scores.size != len(candidates):
            raise ValueError(
                'Scorer returned %d values for %d candidates.' %
                (scores.size, len(candidates))
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError('Candidate scores contain NaN or infinity.')

        # Protein ID is the deterministic secondary key for tied scores.
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index]),
        )
        ranked_proteins = [candidates[index] for index in order]
        ranked_scores = [float(scores[index]) for index in order]
        positive_ranks = [
            rank for rank, protein_id in enumerate(ranked_proteins, start=1)
            if protein_id in held_out
        ]
        if len(positive_ranks) != len(held_out):
            raise ValueError(
                'Only %d of %d held-out positives were ranked for compound %s.' %
                (len(positive_ranks), len(held_out), compound_id)
            )

        first_rank = min(positive_ranks)
        reciprocal_rank = 1.0 / first_rank
        reciprocal_ranks.append(reciprocal_rank)
        first_positive_ranks.append(first_rank)
        candidate_counts.append(len(candidates))
        total_test_positives += len(held_out)

        row = {
            'compound_id': compound_id,
            'candidate_count': len(candidates),
            'filtered_train_positives': len(filtered),
            'test_positive_count': len(held_out),
            'first_positive_rank': first_rank,
            'reciprocal_rank': reciprocal_rank,
        }
        for k in ks:
            cutoff = min(k, len(ranked_proteins))
            hits = sum(
                protein_id in held_out for protein_id in ranked_proteins[:cutoff]
            )
            precision = hits / float(cutoff) if cutoff else 0.0
            recall = hits / float(len(held_out))
            hit_rate = 1.0 if hits else 0.0
            row['precision@%d' % k] = precision
            row['recall@%d' % k] = recall
            row['hits@%d' % k] = hit_rate
            macro['precision'][k].append(precision)
            macro['recall'][k].append(recall)
            macro['hits'][k].append(hit_rate)
            micro_hits[k] += hits
            micro_slots[k] += cutoff
        per_compound.append(row)

        should_export = (
            export_compound_ids is None or compound_id in export_compound_ids
        )
        export_limit = (
            len(ranked_proteins)
            if should_export and export_top == 0 and export_compound_ids is not None
            else export_top
        )
        for rank, (protein_id, score) in enumerate(
                zip(
                    ranked_proteins[:export_limit],
                    ranked_scores[:export_limit],
                ), start=1):
            if not should_export:
                break
            top_candidates.append({
                'compound_id': compound_id,
                'rank': rank,
                'protein_id': protein_id,
                'score': score,
                'label_status': (
                    'held_out_positive' if protein_id in held_out else 'unlabeled'
                ),
                'in_sampled_test_negatives': (
                    (compound_id, protein_id) in sampled_test_negatives
                ),
            })

    if not per_compound:
        raise ValueError('The outer test fold contains no positive compounds.')

    metrics = {
        'aggregation': 'macro_over_compounds',
        'MRR': float(np.mean(reciprocal_ranks)),
        'evaluated_compounds': len(per_compound),
        'held_out_positive_pairs': total_test_positives,
        'candidate_pairs': int(sum(candidate_counts)),
        'held_out_positive_prevalence': (
            total_test_positives / float(sum(candidate_counts))
        ),
        'candidate_count_per_compound': _summary(candidate_counts),
        'first_positive_rank': _summary(first_positive_ranks),
    }
    for k in ks:
        metrics['Precision@%d' % k] = float(np.mean(macro['precision'][k]))
        metrics['Recall@%d' % k] = float(np.mean(macro['recall'][k]))
        metrics['Hits@%d' % k] = float(np.mean(macro['hits'][k]))
        metrics['Enrichment@%d' % k] = (
            metrics['Precision@%d' % k] /
            metrics['held_out_positive_prevalence']
        )
        metrics['MicroPrecision@%d' % k] = (
            micro_hits[k] / float(micro_slots[k]) if micro_slots[k] else 0.0
        )
        metrics['MicroRecall@%d' % k] = (
            micro_hits[k] / float(total_test_positives)
        )

    return {
        'protocol': {
            'candidate_scope': 'all_model_proteins',
            'filter': 'outer_train_positive_pairs',
            'targets': 'outer_test_positive_pairs',
            'non_positive_status': 'unlabeled',
            'tie_break': 'protein_id_ascending',
            'precision_denominator': 'min(K, candidate_count)',
            'aggregation': 'macro_over_test_compounds',
            'export_compound_ids': (
                sorted(export_compound_ids)
                if export_compound_ids is not None else None
            ),
            'export_top': export_top,
        },
        'metrics': metrics,
        'per_compound': per_compound,
        'top_candidates': top_candidates,
    }


def assess_pu_evidence(ranking_metrics, sampled_metrics=None, largest_k=None):
    """Describe what checkpoint-only ranking can and cannot establish about PU."""
    recall_keys = sorted(
        (
            (int(key.split('@', 1)[1]), key)
            for key in ranking_metrics
            if key.startswith('Recall@')
        ),
        key=lambda item: item[0],
    )
    if not recall_keys:
        raise ValueError('At least one Recall@K metric is required for PU assessment.')
    if largest_k is None:
        largest_k, recall_key = recall_keys[-1]
    else:
        largest_k = int(largest_k)
        recall_key = 'Recall@%d' % largest_k
        if recall_key not in ranking_metrics:
            raise ValueError('%s is missing from ranking metrics.' % recall_key)

    recall_value = float(ranking_metrics[recall_key])
    mrr = float(ranking_metrics['MRR'])
    sampled_aupr = None
    if sampled_metrics is not None and 'AUPR' in sampled_metrics:
        sampled_aupr = float(sampled_metrics['AUPR'])

    if sampled_aupr is not None and sampled_aupr >= 0.90 and recall_value < 0.50:
        priority = 'external_validation_before_pu_pilot'
        rationale = (
            'Sampled-negative AUPR is high but fixed-candidate Recall@%d is below 0.50. '
            'This shows an evaluation/ranking gap, but does not isolate false-negative bias. '
            'Validate top unlabeled predictions first; test PU only if they contain credible positives.'
        ) % largest_k
    elif recall_value >= 0.80 and mrr >= 0.50:
        priority = 'pu_not_a_current_priority'
        rationale = (
            'Fixed-candidate ranking already retrieves most held-out positives near the top. '
            'Checkpoint-only evidence does not currently justify adding PU complexity.'
        )
    else:
        priority = 'evidence_inconclusive_validate_top_unlabeled'
        rationale = (
            'Ranking quality is neither clearly saturated nor diagnostic of label noise. '
            'External/database validation of top unlabeled pairs is required before attributing '
            'the remaining error to positive-unlabeled bias.'
        )

    return {
        'necessity_conclusion': 'not_identifiable_from_internal_labels_alone',
        'trial_priority': priority,
        'rationale': rationale,
        'decision_inputs': {
            'sampled_test_AUPR': sampled_aupr,
            recall_key: recall_value,
            'MRR': mrr,
            'held_out_positive_prevalence': float(
                ranking_metrics['held_out_positive_prevalence']
            ),
        },
        'interpretation_limits': [
            'Unrecorded compound-protein pairs are unlabeled, not verified negatives.',
            'Strong or weak ranking alone cannot estimate the hidden-positive rate.',
            'A PU pilot is justified only after external validation suggests that high-scoring '
            'unlabeled pairs contain a meaningful number of credible positives.',
        ],
    }

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score


COUNTERFACTUAL_DONOR_STRATEGIES = (
    'random',
    'exact_degree',
    'exact_degree_disjoint',
)


def build_counterfactual_donors(
        compound_ids,
        compound_to_herbs,
        draws=20,
        seed=42026,
        strategy='exact_degree_disjoint',
        donor_compound_ids=None,
        assignment_compound_ids=None):
    """Build deterministic per-compound donor assignments.

    ``random`` only excludes the source compound. ``exact_degree`` also
    requires equal H-C degree but permits herb overlap. The strict
    ``exact_degree_disjoint`` strategy additionally rejects every donor that
    shares an herb with the source.
    """
    if int(draws) <= 0:
        raise ValueError('draws must be positive.')
    strategy = str(strategy).strip().lower()
    if strategy not in COUNTERFACTUAL_DONOR_STRATEGIES:
        raise ValueError(
            'Unknown donor strategy %s; expected one of %s.' % (
                strategy, ', '.join(COUNTERFACTUAL_DONOR_STRATEGIES)
            )
        )

    normalized = {
        str(compound_id): frozenset(str(herb_id) for herb_id in herbs)
        for compound_id, herbs in compound_to_herbs.items()
        if herbs
    }
    compounds = sorted({str(compound_id) for compound_id in compound_ids})
    donor_compounds = sorted({
        str(compound_id) for compound_id in (
            compounds if donor_compound_ids is None else donor_compound_ids
        )
    })
    assignment_compounds = set(
        compounds if assignment_compound_ids is None else (
            str(compound_id) for compound_id in assignment_compound_ids
        )
    )
    eligible = {
        compound_id for compound_id in donor_compounds
        if compound_id in normalized
    }
    degree_buckets = defaultdict(set)
    herb_to_compounds = defaultdict(set)
    for compound_id in eligible:
        herbs = normalized[compound_id]
        degree_buckets[len(herbs)].add(compound_id)
        for herb_id in herbs:
            herb_to_compounds[herb_id].add(compound_id)

    rng = np.random.RandomState(int(seed))
    assignments = {}
    pool_sizes = {}
    unique_donors = {}
    for compound_id in compounds:
        herbs = normalized.get(compound_id, frozenset())
        if not herbs:
            continue
        if strategy == 'random':
            pool_size = len(eligible) - int(compound_id in eligible)
            pool = None
            if compound_id in assignment_compounds:
                pool = set(eligible)
                pool.discard(compound_id)
        else:
            degree_pool = degree_buckets[len(herbs)]
            if strategy == 'exact_degree_disjoint':
                overlapping = set()
                for herb_id in herbs:
                    overlapping.update(herb_to_compounds[herb_id])
                pool_size = len(degree_pool - overlapping)
                pool = None
                if compound_id in assignment_compounds:
                    pool = set(degree_pool)
                    pool.difference_update(overlapping)
            else:
                pool_size = len(degree_pool) - int(compound_id in degree_pool)
                pool = None
                if compound_id in assignment_compounds:
                    pool = set(degree_pool)
                    pool.discard(compound_id)
        if pool_size <= 0:
            continue
        replace = pool_size < int(draws)
        if compound_id not in assignment_compounds:
            rng.choice(pool_size, size=int(draws), replace=replace)
            continue
        pool = sorted(pool)
        pool_sizes[compound_id] = len(pool)
        selected = rng.choice(pool, size=int(draws), replace=replace).tolist()
        assignments[compound_id] = [str(value) for value in selected]
        unique_donors[compound_id] = len(set(assignments[compound_id]))

    return {
        'strategy': strategy,
        'assignments': assignments,
        'pool_sizes': pool_sizes,
        'unique_donors': unique_donors,
        'draws': int(draws),
        'seed': int(seed),
        'eligible_compounds': len(assignments),
        'requested_compounds': len(
            set(compounds).intersection(assignment_compounds)
        ),
    }


def build_exact_degree_counterfactuals(
        compound_ids, compound_to_herbs, draws=20, seed=42026):
    """Build deterministic, disjoint H-C counterfactual assignments.

    A valid donor has the same H-C degree as the source compound and shares no
    herb with it. Assignments are made per compound, not per C-P pair, so a
    compound receives the same donor in one draw for all candidate proteins.
    """
    return build_counterfactual_donors(
        compound_ids,
        compound_to_herbs,
        draws=draws,
        seed=seed,
        strategy='exact_degree_disjoint',
    )


def summarize_donor_assignments(assignments, compound_to_herbs):
    """Summarize degree matching and herb overlap in donor assignments."""
    normalized = {
        str(compound_id): frozenset(str(herb_id) for herb_id in herbs)
        for compound_id, herbs in compound_to_herbs.items()
    }
    rows = []
    for source_id, donor_ids in assignments.items():
        source_herbs = normalized.get(str(source_id), frozenset())
        for donor_id in donor_ids:
            donor_herbs = normalized.get(str(donor_id), frozenset())
            intersection = len(source_herbs & donor_herbs)
            union = len(source_herbs | donor_herbs)
            rows.append({
                'degree_matched': len(source_herbs) == len(donor_herbs),
                'shared_herbs': intersection,
                'jaccard': float(intersection / float(union)) if union else 0.0,
            })
    if not rows:
        return {
            'assignments': 0,
            'degree_matched_fraction': None,
            'overlap_fraction': None,
            'mean_shared_herbs': None,
            'mean_jaccard': None,
        }
    return {
        'assignments': len(rows),
        'degree_matched_fraction': float(np.mean([
            row['degree_matched'] for row in rows
        ])),
        'overlap_fraction': float(np.mean([
            row['shared_herbs'] > 0 for row in rows
        ])),
        'mean_shared_herbs': float(np.mean([
            row['shared_herbs'] for row in rows
        ])),
        'mean_jaccard': float(np.mean([row['jaccard'] for row in rows])),
    }


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return (None, None)
    probability = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z * np.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        ) / denominator
    )
    return float(center - radius), float(center + radius)


def _margin_summary(margins, compound_ids):
    margins = np.asarray(margins, dtype=np.float64)
    compound_ids = np.asarray(compound_ids, dtype=object)
    if margins.size == 0:
        return {
            'pairs': 0,
            'compounds': 0,
            'mean_margin': None,
            'median_margin': None,
            'pair_win_rate': None,
            'pair_win_rate_ci95': [None, None],
            'compound_win_rate': None,
        }

    pair_wins = int(np.sum(margins > 0.0))
    lower, upper = wilson_interval(pair_wins, len(margins))
    compound_means = []
    for compound_id in sorted(set(compound_ids.tolist())):
        compound_means.append(float(np.mean(margins[compound_ids == compound_id])))
    compound_means = np.asarray(compound_means, dtype=np.float64)
    return {
        'pairs': int(len(margins)),
        'compounds': int(len(compound_means)),
        'mean_margin': float(np.mean(margins)),
        'median_margin': float(np.median(margins)),
        'margin_std': float(np.std(margins)),
        'standardized_mean_margin': (
            float(np.mean(margins) / np.std(margins))
            if np.std(margins) > 0 else None
        ),
        'pair_win_rate': float(pair_wins / float(len(margins))),
        'pair_win_rate_ci95': [lower, upper],
        'compound_win_rate': float(np.mean(compound_means > 0.0)),
        'compound_mean_margin': float(np.mean(compound_means)),
    }


def summarize_counterfactual_audit(
        labels,
        compound_ids,
        factual_logits,
        counterfactual_logits,
        subgroup_values=None,
        requested_records=None,
        minimum_group_pairs=30,
        pair_win_threshold=0.60,
        aupr_drop_threshold=0.001,
        consistency_threshold=0.75,
        coverage_threshold=0.90):
    labels = np.asarray(labels, dtype=np.int32)
    compound_ids = np.asarray(compound_ids, dtype=object)
    factual_logits = np.asarray(factual_logits, dtype=np.float64)
    counterfactual_logits = np.asarray(counterfactual_logits, dtype=np.float64)
    if counterfactual_logits.ndim != 2:
        raise ValueError('counterfactual_logits must have shape [draws, records].')
    if not (
            len(labels) == len(compound_ids) == len(factual_logits)
            == counterfactual_logits.shape[1]):
        raise ValueError('All record-level inputs must have the same length.')
    if len(np.unique(labels)) != 2:
        raise ValueError('The audit split must contain both positive and negative labels.')

    requested_records = int(requested_records or len(labels))
    coverage = float(len(labels) / float(requested_records)) if requested_records else 0.0
    factual_scores = 1.0 / (1.0 + np.exp(-np.clip(factual_logits, -50.0, 50.0)))
    factual_aupr = float(average_precision_score(labels, factual_scores))
    draw_rows = []
    for draw_index, logits in enumerate(counterfactual_logits):
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        counterfactual_aupr = float(average_precision_score(labels, scores))
        draw_rows.append({
            'draw': int(draw_index + 1),
            'counterfactual_AUPR': counterfactual_aupr,
            'factual_minus_counterfactual_AUPR': (
                factual_aupr - counterfactual_aupr
            ),
        })

    mean_counterfactual_logits = np.mean(counterfactual_logits, axis=0)
    mean_margins = factual_logits - mean_counterfactual_logits
    positive_mask = labels == 1
    positive_summary = _margin_summary(
        mean_margins[positive_mask], compound_ids[positive_mask]
    )
    negative_summary = _margin_summary(
        mean_margins[~positive_mask], compound_ids[~positive_mask]
    )

    subgroup_report = {}
    analyzable_directions = []
    for subgroup_name, raw_groups in (subgroup_values or {}).items():
        raw_groups = np.asarray(raw_groups, dtype=object)
        if len(raw_groups) != len(labels):
            raise ValueError('Subgroup %s does not match record count.' % subgroup_name)
        groups = {}
        for group in sorted(set(raw_groups.tolist())):
            mask = positive_mask & (raw_groups == group)
            summary = _margin_summary(mean_margins[mask], compound_ids[mask])
            summary['analyzable'] = summary['pairs'] >= int(minimum_group_pairs)
            groups[str(group)] = summary
            if summary['analyzable']:
                analyzable_directions.append(summary['mean_margin'] > 0.0)
        subgroup_report[str(subgroup_name)] = groups

    mean_aupr_drop = float(np.mean([
        row['factual_minus_counterfactual_AUPR'] for row in draw_rows
    ]))
    positive_strata_fraction = (
        float(np.mean(analyzable_directions)) if analyzable_directions else None
    )
    criteria = {
        'coverage': coverage >= float(coverage_threshold),
        'positive_pair_win_rate': (
            positive_summary['pair_win_rate'] is not None
            and positive_summary['pair_win_rate'] >= float(pair_win_threshold)
        ),
        'positive_mean_margin': (
            positive_summary['mean_margin'] is not None
            and positive_summary['mean_margin'] > 0.0
        ),
        'AUPR_drop': mean_aupr_drop >= float(aupr_drop_threshold),
        'degree_strata_consistency': (
            positive_strata_fraction is not None
            and positive_strata_fraction >= float(consistency_threshold)
        ),
    }
    if not criteria['coverage']:
        decision = 'inconclusive_counterfactual_coverage'
    elif all(criteria.values()):
        decision = 'supports_CHCR_training_pilot'
    else:
        decision = 'does_not_support_CHCR_training_pilot'

    return {
        'coverage': {
            'requested_records': requested_records,
            'audited_records': int(len(labels)),
            'fraction': coverage,
        },
        'factual_AUPR': factual_aupr,
        'counterfactual_AUPR': {
            'mean': float(np.mean([
                row['counterfactual_AUPR'] for row in draw_rows
            ])),
            'std': float(np.std([
                row['counterfactual_AUPR'] for row in draw_rows
            ])),
            'mean_factual_minus_counterfactual': mean_aupr_drop,
        },
        'positive_pairs': positive_summary,
        'negative_pairs': negative_summary,
        'degree_strata_positive_fraction': positive_strata_fraction,
        'subgroups': subgroup_report,
        'draws': draw_rows,
        'pre_registered_thresholds': {
            'coverage': float(coverage_threshold),
            'positive_pair_win_rate': float(pair_win_threshold),
            'AUPR_drop': float(aupr_drop_threshold),
            'degree_strata_consistency': float(consistency_threshold),
            'minimum_group_pairs': int(minimum_group_pairs),
        },
        'criteria': criteria,
        'decision': decision,
        'mean_counterfactual_logits': mean_counterfactual_logits,
        'mean_margins': mean_margins,
    }

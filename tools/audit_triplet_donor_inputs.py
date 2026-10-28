#!/usr/bin/env python3
"""Sampled four-library input diagnostics; never score validation outcomes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse

from tools.run_donor_reweighting_pilot import load_dataset, dump
from tools.evaluate_checkpoint_ranking import sha256_file
from tools.evaluate_non_neural_cold_start_baselines import read_relation_pairs
from util.config import ModelConf
from util.dataSplit import SIDE_RELATION_FILE_CANDIDATES
from util.io import resolve_optional_dataset_file
from util.triplet_donor_context import TripletDonorContext


SETTINGS = dict(seed=112026, queries=32, targets=256, target_block=64,
                numerical_tolerance=1e-6)


def mapped_pd(protein_ids, pairs):
    proteins = {str(value): i for i, value in enumerate(protein_ids)}
    if len(proteins) != len(protein_ids):
        raise ValueError('Duplicate target IDs.')
    diseases = sorted({str(d) for _, d in pairs})
    disease_index = {d: i for i, d in enumerate(diseases)}
    rows, cols = [], []
    for protein, disease in pairs:
        if str(protein) not in proteins:
            raise ValueError('P-D contains a target outside the frozen universe.')
        rows.append(proteins[str(protein)])
        cols.append(disease_index[str(disease)])
    result = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(len(protein_ids), len(diseases)))
    result.data[:] = 1
    return result, diseases


def diagnose_block(kappa, labels, weights, tolerance=1e-6):
    """Only TRAINING donor labels enter these covariances, never query labels."""
    kappa, labels, weights = map(np.asarray, (kappa, labels, weights))
    if (kappa.shape != labels.shape or kappa.ndim != 2
            or weights.shape != (kappa.shape[0],) or not len(weights)
            or not np.all(np.isfinite(kappa)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1)
            or not np.all((labels == 0) | (labels == 1))):
        raise ValueError('Invalid training-only diagnostic inputs.')
    base = weights @ labels
    mean = weights @ kappa
    # d(sum softmax(log w + theta*kappa)*Y)/dtheta at theta=0.
    derivative = weights @ (kappa * labels) - base * mean
    spread = np.ptp(kappa, axis=0)
    mixed = (labels.sum(axis=0) > 0) & (labels.sum(axis=0) < len(weights))
    return dict(variable=spread > tolerance, mixed=mixed,
                sensitive=np.abs(derivative) > tolerance, derivative=derivative,
                spread=spread, mean=mean)


def audit_dataset(dataset, out):
    matrices, _, _, metadata = load_dataset(dataset)
    conf = ModelConf(str(ROOT / dataset['candidate_config']))
    path = resolve_optional_dataset_file(str((ROOT / conf['datapath']).parent),
                                        SIDE_RELATION_FILE_CANDIDATES['P_D'])
    if sha256_file(path) != metadata['relation_hashes']['P_D']:
        raise ValueError('P-D changed during preflight.')
    pd, disease_ids = mapped_pd(matrices['protein_ids'], read_relation_pairs(path))
    rng = np.random.RandomState(SETTINGS['seed'])
    queries = np.sort(rng.choice(matrices['test_compounds'],
                      min(SETTINGS['queries'], len(matrices['test_compounds'])), replace=False))
    targets = np.sort(rng.choice(len(matrices['protein_ids']),
                      min(SETTINGS['targets'], len(matrices['protein_ids'])), replace=False))
    # Freeze selections before constructing any condition or reading its values.
    dump(out / 'input_seal.json', dict(metadata=metadata,
        query_ids=[matrices['compound_ids'][int(c)] for c in queries],
        target_ids=[matrices['protein_ids'][int(p)] for p in targets],
        disease_order_sha256=hashlib.sha256(json.dumps(disease_ids).encode()).hexdigest()))
    start = time.monotonic()
    model = TripletDonorContext(matrices['hc'], matrices['cp'], pd)
    if model.evidence.cp[queries].nnz:
        raise ValueError('Sampled cold query has training labels.')
    records = []
    for c in queries:
        query = model.evidence.query(int(c))
        count = len(query.donors)
        record = dict(compound_id=matrices['compound_ids'][int(c)], donors=count,
                      candidate_count=len(targets), evaluated_candidates=0,
                      variable_candidates=0, mixed_support_candidates=0,
                      sensitive_candidates=0, changed_by_candidate_exclusion=0)
        if not count:
            records.append(record)
            continue
        weights = query.jaccard / query.jaccard.sum()
        derivative_sum = spread_sum = mean_sum = 0.0
        for offset in range(0, len(targets), SETTINGS['target_block']):
            block = targets[offset:offset + SETTINGS['target_block']]
            kappa = model.compatibility(query.donors, block)
            labels = query.labels[:, block].toarray()
            stats = diagnose_block(kappa, labels, weights, SETTINGS['numerical_tolerance'])
            product = (model.profiles[query.donors] @ model.pd[block].T).toarray()
            denom = np.sqrt(model.profile_norm2[query.donors, None] * model.target_norm2[block][None, :])
            included = np.zeros_like(product)
            np.divide(product, denom, out=included, where=denom > 1e-12)
            changed = np.any(np.abs(included-kappa) > SETTINGS['numerical_tolerance'], axis=0)
            record['evaluated_candidates'] += len(block)
            record['variable_candidates'] += int(stats['variable'].sum())
            record['mixed_support_candidates'] += int(stats['mixed'].sum())
            record['sensitive_candidates'] += int(stats['sensitive'].sum())
            record['changed_by_candidate_exclusion'] += int(changed.sum())
            derivative_sum += float(np.abs(stats['derivative']).sum())
            spread_sum += float(stats['spread'].sum())
            mean_sum += float(stats['mean'].sum())
        record.update(mean_abs_score_derivative=derivative_sum / len(targets),
                      mean_donor_range=spread_sum / len(targets),
                      mean_compatibility=mean_sum / len(targets))
        records.append(record)
    totals = {field: sum(r[field] for r in records) for field in (
        'evaluated_candidates', 'variable_candidates', 'mixed_support_candidates',
        'sensitive_candidates', 'changed_by_candidate_exclusion')}
    denominator = max(totals['evaluated_candidates'], 1)
    max_donors = max(r['donors'] for r in records)
    report = dict(metadata=metadata, sampled_queries=len(queries), sampled_targets=len(targets),
        no_donor_queries=sum(r['donors']==0 for r in records),
        disease_count=len(disease_ids), pd_edges=int(pd.nnz),
        targets_with_pd=int(np.count_nonzero(np.diff(pd.indptr))),
        training_donors=int(model.evidence.supported.sum()),
        donors_with_nonempty_profile=int(np.count_nonzero(model.profile_norm2)),
        profile_nnz=int(model.profiles.nnz),
        profile_csr_bytes=int(model.profiles.data.nbytes+model.profiles.indices.nbytes+model.profiles.indptr.nbytes),
        one_sampled_dense_block_bytes=int(max_donors * min(SETTINGS['target_block'],len(targets)) * 8),
        totals=totals, fractions={k: v/denominator for k,v in totals.items() if k!='evaluated_candidates'},
        per_query=records, seconds=time.monotonic()-start,
        training_optimizer_steps=0, validation_metrics_computed=False, outer_test_scored=False)
    dump(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/triplet_donor_input_audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'configs/sacti_inner_ranking_audit.json'
    datasets = json.loads(source.read_text())['datasets']
    files = [source, Path(__file__).resolve(), ROOT/'util/triplet_donor_context.py',
             ROOT/'util/donor_reweighting.py', ROOT/'tools/run_donor_reweighting_pilot.py',
             ROOT/'tools/evaluate_non_neural_cold_start_baselines.py', ROOT/'util/dataSplit.py']
    dump(args.output/'seal.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        settings=SETTINGS, source_hashes={str(p.relative_to(ROOT)): sha256_file(p) for p in files}))
    reports=[]
    for dataset in datasets:
        print('Input preflight: '+dataset['label'], flush=True)
        out=args.output/dataset['key']
        out.mkdir()
        reports.append(audit_dataset(dataset,out))
    lines=['# Triplet Donor Input Preflight', '',
           'Fixed ID-based sample, no validation metrics, no training, no outer scoring.', '',
           '| Dataset | Queries | No donors | Variable condition | Mixed donor labels | Nonzero score derivative |',
           '|---|---:|---:|---:|---:|---:|']
    for r in reports:
        f=r['fractions']
        lines.append('| %s | %d | %d | %.4f | %.4f | %.4f |' % (
            r['metadata']['dataset'],r['sampled_queries'],r['no_donor_queries'],
            f['variable_candidates'],f['mixed_support_candidates'],f['sensitive_candidates']))
    lines+=['', 'Fractions use sampled query-target pairs WITH at least one donor as denominator.',
            'Derivative uses only training donor labels, not held-out outcomes. Nonzero is not useful prediction.',
            'Memory fields describe CSR storage and ONE dense block, not process or GPU peak memory.',
            'A nondegenerate input does not establish novelty, ranking gain, or authorize full training.']
    dump(args.output/'summary.json', dict(settings=SETTINGS,reports=reports,
         validation_metrics_computed=False,outer_test_scored=False,training_optimizer_steps=0))
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Summary: '+str(args.output/'summary.md'))


if __name__=='__main__':
    main()

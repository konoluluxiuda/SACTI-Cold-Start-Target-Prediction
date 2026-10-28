#!/usr/bin/env python3
"""Frozen-budget four-library inner-only donor weighting pilot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.evaluate_checkpoint_ranking import records_sha256, sha256_file
from tools.evaluate_non_neural_cold_start_baselines import (
    build_matrices_from_records, read_relation_pairs, herb_prototype_profiles,
    hc_jaccard_label_propagation)
from tools.audit_full_candidate_support_strata import write_tsv
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking
from util.config import ModelConf
from util.dataSplit import DataSplit, SIDE_RELATION_FILE_CANDIDATES
from util.io import resolve_optional_dataset_file
from util.model_components import resolve_early_stopping
from util.donor_reweighting import DonorEvidence
from util.donor_weight_learner import DonorWeightLearner


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_dataset(dataset):
    config = ROOT / dataset['candidate_config']
    if sha256_file(config) != dataset['candidate_config_sha256']:
        raise ValueError('Frozen source config changed.')
    conf = ModelConf(str(config))
    folds, _ = DataSplit.prepareStrictFolds(conf, conf['datapath'], 5)
    train, validation, _ = DataSplit.innerValidationSplitForConfig(
        conf, folds[0][0], resolve_early_stopping(conf)['ratio'], int(conf['validation.seed']))
    old_path = ROOT / 'results/sacti_retrieval_error_audit' / dataset['key'] / 'report.json'
    old = json.loads(old_path.read_text())
    if (records_sha256(train) != old['split']['ranking_train_sha256'] or
            records_sha256(validation) != old['split']['ranking_test_sha256']):
        raise ValueError('Frozen inner partition changed.')
    if {str(r[0]) for r in train} & {str(r[0]) for r in validation}:
        raise ValueError('Expected compound-disjoint inner validation.')
    data_dir = (ROOT / conf['datapath']).parent
    paths = {key: resolve_optional_dataset_file(str(data_dir), SIDE_RELATION_FILE_CANDIDATES[key])
             for key in ('H_C', 'P_D', 'C_P')}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    if (hashes['H_C'] != old['score_complementarity']['hc_sha256'] or
            hashes['P_D'] != old['score_complementarity']['pd_sha256']):
        raise ValueError('Frozen side information changed.')
    # Full C-P and outer records provide entity IDs only, never donor evidence.
    protein_ids = ([p for p, _ in read_relation_pairs(paths['P_D'])]
                   + [p for _, p in read_relation_pairs(paths['C_P'])]
                   + [str(r[1]) for r in folds[0][1]])
    matrices = build_matrices_from_records(read_relation_pairs(paths['H_C']), train, validation,
                                          extra_protein_ids=protein_ids)
    if len(matrices['protein_ids']) != old['entity_universe']['proteins']:
        raise ValueError('Candidate universe changed.')
    metadata = dict(dataset=dataset['label'], key=dataset['key'], config_sha256=sha256_file(config),
        relation_hashes=hashes, train_sha256=records_sha256(train),
        validation_sha256=records_sha256(validation), source_report_sha256=sha256_file(old_path),
        candidate_ids_sha256=hashlib.sha256(
            json.dumps(matrices['protein_ids']).encode()).hexdigest(),
        proteins=len(matrices['protein_ids']), compounds=len(matrices['compound_ids']))
    return matrices, train, validation, metadata


def ranking(matrices, train, validation, scorer):
    cindex = {str(c): i for i, c in enumerate(matrices['compound_ids'])}
    pindex = {str(p): i for i, p in enumerate(matrices['protein_ids'])}

    def score_pairs(cs, ps):
        if len(set(cs)) != 1:
            raise ValueError('Expected one query per full-candidate scoring call.')
        scores = scorer(cindex[str(cs[0])])
        return scores[[pindex[str(p)] for p in ps]]

    return evaluate_fixed_candidate_ranking(matrices['protein_ids'], train, validation,
        score_pairs, ks=(20,), export_top=20)


def make_blocks(cp, settings):
    supported = np.flatnonzero(np.diff(cp.indptr) > 0)
    if len(supported) < 2:
        raise ValueError('At least two supported training compounds are required.')
    rng = np.random.RandomState(settings['seed'])
    return np.array_split(rng.permutation(supported), min(settings['masked_blocks'], len(supported)))


def selected_epoch(current, best, reference, settings):
    return (current['MRR'] > best['MRR'] + settings['selection_min_delta'] and
            current['Recall@20'] >= reference['Recall@20'] - settings['selection_recall20_tolerance'])


def run_dataset(data, settings, out):
    matrices, train, validation, metadata = data
    started = time.monotonic()
    hc, cp = matrices['hc'], matrices['cp']
    evidence = DonorEvidence(hc, cp, [])
    baseline = ranking(matrices, train, validation, lambda c: evidence.query(c).score()[0])
    references = {'fixed_donor': baseline['metrics']}
    for name, function in (('prototype', herb_prototype_profiles), ('jaccard', hc_jaccard_label_propagation)):
        profiles, _ = function(hc, cp, matrices['test_compounds'])
        row = {int(c): i for i, c in enumerate(matrices['test_compounds'])}
        references[name] = ranking(matrices, train, validation, lambda c: profiles[row[c]])['metrics']
    blocks = make_blocks(cp, settings)
    dump(out / 'masked_blocks.json', [[matrices['compound_ids'][int(c)] for c in b] for b in blocks])
    masked = [DonorEvidence(hc, cp, b) for b in blocks]
    block_of = {int(c): i for i, block in enumerate(blocks) for c in block}
    schedule = np.array(sorted(block_of))
    rng = np.random.RandomState(settings['seed'])
    schedule = rng.permutation(schedule)
    cursor = 0
    learner = DonorWeightLearner(settings)
    steps, no_donor, unreachable = 0, 0, 0
    best_epoch, stale = 0, 0
    best_report = baseline
    history = [dict(epoch=0, **baseline['metrics'])]
    try:
        best_values = learner.values()
        # Independently check the shared network leaves the initial score unchanged.
        for c in matrices['test_compounds']:
            q = evidence.query(int(c))
            np.testing.assert_allclose(learner.predict(q, best_values), q.score()[0], atol=1e-12, rtol=0)
        for epoch in range(1, settings['max_epochs'] + 1):
            losses = []
            for _ in range(min(settings['queries_per_epoch'], len(schedule))):
                if cursor == len(schedule):
                    schedule = rng.permutation(schedule)
                    cursor = 0
                c = int(schedule[cursor])
                cursor += 1
                q = masked[block_of[c]].query(c)
                positives = cp[c].indices
                reachable = np.asarray(q.labels.getnnz(axis=0)).reshape(-1) > 0
                unreachable += int(np.sum(~reachable[positives]))
                loss = learner.step(q, positives)
                if loss is None:
                    no_donor += 1
                else:
                    losses.append(loss)
                    steps += 1
            values = learner.values()
            report = ranking(matrices, train, validation,
                             lambda c: learner.predict(evidence.query(c), values))
            metrics = report['metrics']
            improved = selected_epoch(metrics, best_report['metrics'], baseline['metrics'], settings)
            if improved:
                best_epoch, best_values, best_report, stale = epoch, values, report, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None,
                selected=bool(improved), optimizer_steps=steps, **metrics))
            dump(out / 'history.json', history)
            print('%s epoch %d MRR=%.6f R@20=%.6f best=%d stale=%d/%d' % (
                metadata['dataset'], epoch, metrics['MRR'], metrics['Recall@20'],
                best_epoch, stale, settings['patience']), flush=True)
            if stale >= settings['patience']:
                break
        learner.restore(best_values)
        np.savez(out / 'weights.npz', kernel=best_values[0], bias=best_values[1], output=best_values[2])
    finally:
        learner.close()
    for key in ('per_compound', 'top_candidates'):
        rows = best_report[key]
        if rows:
            write_tsv(out / (key + '.tsv'), rows, list(rows[0]))
    result = dict(metadata=metadata, outer_test_scored=False, evaluation_split='validation',
        best_epoch=best_epoch, optimizer_steps=steps, no_donor_query_visits=no_donor,
        unreachable_positive_visits=unreachable, references=references, candidate=best_report['metrics'],
        seconds=time.monotonic() - started, weights_sha256=sha256_file(out / 'weights.npz'),
        masked_blocks_sha256=sha256_file(out / 'masked_blocks.json'))
    dump(out / 'report.json', result)
    dump(out / 'completed.json', {name: sha256_file(out / name) for name in (
        'report.json', 'weights.npz', 'masked_blocks.json', 'history.json',
        'per_compound.tsv', 'top_candidates.tsv')})
    return result


def summarize(reports, settings, out):
    # The stronger rule per metric is an evaluation envelope, not a deployed selector.
    rows = []
    for r in reports:
        refs, candidate = r['references'], r['candidate']
        mrr = max(x['MRR'] for x in refs.values())
        recall = max(x['Recall@20'] for x in refs.values())
        rows.append(dict(dataset=r['metadata']['dataset'], fixed=refs['fixed_donor']['MRR'],
            prototype=refs['prototype']['MRR'], jaccard=refs['jaccard']['MRR'],
            candidate=candidate['MRR'], delta_mrr=candidate['MRR'] - mrr,
            candidate_recall=candidate['Recall@20'], delta_recall=candidate['Recall@20'] - recall,
            best_epoch=r['best_epoch']))
    gate = settings['gate']
    checks = dict(expected_datasets=len(rows) == 4,
        macro_mrr=np.mean([r['delta_mrr'] for r in rows]) >= gate['macro_mrr_delta'],
        macro_recall20=np.mean([r['delta_recall'] for r in rows]) >= gate['macro_recall20_delta'],
        mrr_wins=sum(r['delta_mrr'] > 0 for r in rows) >= gate['minimum_mrr_winning_datasets'],
        maximum_mrr_drop=min(r['delta_mrr'] for r in rows) >= -gate['maximum_mrr_drop'],
        maximum_recall_drop=min(r['delta_recall'] for r in rows) >= -gate['maximum_recall20_drop'])
    checks = {k: bool(v) for k, v in checks.items()}
    dump(out / 'summary.json', dict(rows=rows, checks=checks, passed=all(checks.values()), outer_test_scored=False))
    lines = ['# Donor Reweighting Inner Pilot', '',
        '| Dataset | Fixed MRR | Prototype MRR | Jaccard MRR | Learned MRR | Delta vs strongest | Learned R@20 | R@20 delta | Best epoch |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {dataset} | {fixed:.6f} | {prototype:.6f} | {jaccard:.6f} | {candidate:.6f} | {delta_mrr:+.6f} | {candidate_recall:.6f} | {delta_recall:+.6f} | {best_epoch} |'.format(**r))
    lines += ['', 'Gate: ' + ('PASS' if all(checks.values()) else 'FAIL'), '',
        'Deltas use the strongest reference for each dataset and metric, not a deployable oracle mixture.',
        'Development validation only. PASS is not evidence of novelty or an unbiased outer confirmation.',
        'FAIL stops this configuration without new outer scoring.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed datasets, not interrupted training.')
    args = parser.parse_args()
    config_path = ROOT / 'configs/donor_reweighting_pilot.json'
    settings = json.loads(config_path.read_text())
    if settings['outer_test_scored'] or settings['evaluation_split'] != 'validation':
        raise ValueError('Only inner validation is allowed.')
    source_path = ROOT / settings['source_manifest']
    datasets = json.loads(source_path.read_text())['datasets']
    if len(datasets) != 4 or len({d['key'] for d in datasets}) != 4:
        raise ValueError('Expected four unique datasets.')
    data = [load_dataset(d) for d in datasets]
    files = [config_path, source_path, Path(__file__).resolve(),
        ROOT / 'util/donor_reweighting.py', ROOT / 'util/donor_weight_learner.py',
        ROOT / 'util/checkpoint_ranking.py', ROOT / 'tools/evaluate_non_neural_cold_start_baselines.py',
        ROOT / 'util/dataSplit.py']
    signature = dict(settings=settings, file_hashes={str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        datasets=[d[3] for d in data])
    if args.dry_run:
        print(json.dumps(signature, indent=2))
        print('Preflight passed. No training or ranking performed.')
        return
    out = (args.output_dir or ROOT / 'results/batch_runs' /
           ('donor_reweighting_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    seal = out / 'seal.json'
    if args.resume:
        if not args.output_dir or not seal.exists() or json.loads(seal.read_text())['signature'] != signature:
            raise ValueError('Resume requires an unchanged sealed run directory.')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Use a new output directory or --resume.')
        out.mkdir(parents=True, exist_ok=True)
        dump(seal, dict(created_at=datetime.now(timezone.utc).isoformat(), signature=signature))
    reports = []
    for dataset, item in zip(datasets, data):
        directory = out / dataset['key']
        report_path = directory / 'report.json'
        if args.resume and report_path.exists():
            completion_path = directory / 'completed.json'
            if not completion_path.exists():
                raise ValueError('Incomplete report without completion marker; archive it before retrying.')
            for name, expected in json.loads(completion_path.read_text()).items():
                if sha256_file(directory / name) != expected:
                    raise ValueError('Completed artifact changed: ' + name)
            report = json.loads(report_path.read_text())
            if report['metadata'] != item[3] or report['weights_sha256'] != sha256_file(directory / 'weights.npz'):
                raise ValueError('Completed dataset metadata or weights changed.')
            reports.append(report)
            print('Reusing completed ' + dataset['label'], flush=True)
            continue
        if directory.exists():
            directory.rename(out / (dataset['key'] + '_incomplete_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        directory.mkdir()
        reports.append(run_dataset(item, settings, directory))
    summarize(reports, settings, out)
    print('Summary: ' + str(out / 'summary.md'))


if __name__ == '__main__':
    main()

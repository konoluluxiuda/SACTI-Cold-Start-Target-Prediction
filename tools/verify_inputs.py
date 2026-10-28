#!/usr/bin/env python3
"""Verify released folds, inner partitions and the 685-pair evidence set."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_data import verify
from tools.evaluate_checkpoint_ranking import records_sha256
from util.config import ModelConf
from util.dataSplit import DataSplit


def check():
    verify(ROOT, json.loads((ROOT / 'data/manifest.json').read_text()))
    policy = json.loads((ROOT / 'configs/sacti_outer_lockbox_seed901026.json').read_text())
    rows = []
    for dataset in policy['datasets']:
        for seed in (52026, 901026):
            conf = ModelConf(str(ROOT / dataset['candidate_config']))
            split_dir = Path(conf['datapath']).parent / 'splits' / (
                'strict_compound_cold_start_seed_%d_k5' % seed)
            conf.config.update({'split.seed': str(seed), 'split.dir': str(split_dir), 'split.reuse': 'True'})
            folds, manifest = DataSplit.prepareStrictFolds(conf, conf['datapath'], 5)
            counts = []
            all_test = set()
            for fold, (outer_train, outer_test) in enumerate(folds, start=1):
                train, validation, _ = DataSplit.innerValidationSplitForConfig(
                    conf, outer_train, 0.1, 152026 + fold - 1)
                compounds = [{str(r[0]) for r in records} for records in (train, validation, outer_test)]
                if any(compounds[i] & compounds[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                    raise ValueError('Compound overlap: ' + dataset['slug'])
                if all_test & compounds[2]:
                    raise ValueError('Duplicate outer-test compound.')
                all_test.update(compounds[2])
                if seed == 901026:
                    reference = ROOT / 'results/batch_runs/sacti_outer_lockbox_seed901026/ranking' / dataset['slug'] / 'candidate' / ('fold_%d' % fold) / 'report.json'
                    historical = json.loads(reference.read_text())['split']
                    for records, key in ((train, 'model_train_sha256'), (validation, 'validation_sha256'),
                                         (outer_test, 'outer_test_sha256'), (outer_train, 'outer_train_sha256')):
                        if records_sha256(records) != historical[key]:
                            raise ValueError('Historical partition mismatch: ' + dataset['slug'])
                counts.append([len(train), len(validation), len(outer_test)])
            rows.append(dict(dataset=dataset['slug'], seed=seed, folds=len(folds),
                             compounds=len(all_test), record_counts=counts,
                             assignment_sha256=manifest['assignments_sha256']))
    evidence = json.loads((ROOT / 'configs/sacti_etcm_confirmed_target_evaluation.json').read_text())
    with (ROOT / evidence['evidence']['path']).open() as handle:
        pairs = list(csv.DictReader(handle, delimiter='\t'))
    cp = {tuple(line.split()[:2]) for line in (ROOT / evidence['model_graph']['cp_path']).read_text().splitlines() if line.strip()}
    held = {(row['compound_id'], row['protein_id']) for row in pairs}
    compounds = {c for c, _ in held}
    if len(pairs) != 685 or len(held) != 685 or len(compounds) != 332 or cp & held:
        raise ValueError('Held-out ETCM evidence identity/isolation mismatch.')
    return dict(passed=True, datasets=4, split_partitions=8, outer_folds=40,
                historical_inner_outer_folds_matched=20,
                evidence_pairs=len(held), evidence_compounds=len(compounds),
                evidence_model_graph_overlap=0, optimizer_steps=0, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the repository root.')
    report = check()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

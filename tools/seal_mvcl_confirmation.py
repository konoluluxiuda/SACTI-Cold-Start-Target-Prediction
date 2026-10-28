#!/usr/bin/env python3
"""Seal a matched outer comparison plan only; no training, inference or new outer metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_mvcl_sacti_matched import digest, read, require


def validate_policy(p, reference):
    require(p['folds'] == [1, 2, 3, 4, 5], 'All five folds required.')
    require(p['split_seed'] == reference['split_seed'] == 901026, 'Wrong matched split seed.')
    require(p['training_seed'] == reference['training_seed'] == 52026, 'Wrong training seed.')
    require(p['validation_seed'] == reference['validation_seed'] == 152026, 'Wrong validation seed.')
    require(p['datasets'] == [d['slug'] for d in reference['datasets']], 'Dataset scope changed.')
    require(not p['new_lockbox_claim'] and p['historical_outer_results_already_seen'],
            'Cannot rebrand a consumed evaluation as an unseen lockbox.')
    for key in ('dataset_specific_selection', 'model_selection_reopened', 'reuse_development_checkpoint',
                'start_training_automatically', 'read_new_outer_metrics_at_sealing'):
        require(p[key] is False, 'Disallowed protocol setting: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new seal output directory.')
    protocol_path = ROOT/'configs/mvcl_formal_confirmation.json'
    p = read(protocol_path)
    reference_path = ROOT/p['matched_sacti_protocol']
    reference = read(reference_path)
    validate_policy(p, reference)
    train_path = ROOT/p['reference_training_seal']
    train = read(train_path)
    require(not train['dry_run'] and not train['outer_test_scored'], 'Invalid development training reference.')
    require(train['config']['model']['reconstruction_scaling'] == 'cp_anchored_mean', 'Wrong variant.')
    comparison_path = ROOT/p['development_comparison']
    comparison = read(comparison_path)
    require(comparison['complete_sacti_four_library_comparison'], 'Development matching incomplete.')
    hashes = {str(f): digest(f) for f in (protocol_path, reference_path, train_path, comparison_path, Path(__file__))}
    for filename, expected in train['code_hashes'].items():
        require(digest(ROOT/filename) == expected, 'Frozen model implementation changed: ' + filename)
        hashes[str(ROOT/filename)] = expected
    jobs = []
    for d in reference['datasets']:
        directory = ROOT/d['split_dir']
        manifest_path, assignments = directory/'manifest.json', directory/'fold_assignments.tsv'
        m = read(manifest_path)
        require(m['seed'] == p['split_seed'], 'Manifest seed mismatch.')
        require(assignments.is_file(), 'Existing assignments required; no new splits created.')
        config = ROOT/d['candidate_config']
        require(digest(config) == d['candidate_config_sha256'], 'SACTI configuration changed.')
        for f in (manifest_path, assignments, config):
            hashes[str(f)] = digest(f)
        jobs.extend(dict(dataset=d['slug'], fold=fold, split_dir=d['split_dir'],
                         state='RUNNER_PENDING', reuse_checkpoint=False) for fold in p['folds'])
    require(len(jobs) == 20, 'Expected 20 new training jobs.')
    result = dict(protocol=p, training_config=train['config'], source_hashes=hashes, jobs=jobs,
                  optimizer_steps=0, model_forward_calls=0, new_outer_metrics_read=False,
                  ready_to_train=False, blocked_by='multi-fold data/training/outer-evaluation entry points pending')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'seal.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Frozen MVCL Matched Confirmation Scope', '',
             'Protocol only. Not a completed confirmation or executable training batch.',
             'Four datasets x five folds = 20 new training jobs on existing split seed 901026.',
             'Scale policy and training budget frozen from the completed ETCM development seal.',
             'All datasets use the same variant. Original adaptation remains an inner sensitivity analysis.',
             'Historical outer results were already seen; this is not a fresh lockbox.',
             'No new training, model inference or outer metric reading.', '',
             'Next engineering step: fold-local data preparation with train/validation/test isolation;',
             'validate fold identity before implementing the training/evaluation batch.',
             'Existing first-fold seed52026 weights must not be reused for seed901026.',
             'All outcomes and resource failures must be reported; no result-dependent retuning.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Protocol sealed; runner not yet ready. Summary:', args.output/'summary.md')


if __name__ == '__main__':
    main()

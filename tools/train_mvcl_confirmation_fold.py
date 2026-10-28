#!/usr/bin/env python3
"""Train one frozen confirmation fold; selects by inner AP and never scores outer records."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tools.mvcl_confirmation_inputs import load_confirmation_bundle
from tools.run_mvcl_cti_pilot import train
from tools.summarize_mvcl_sacti_matched import read, digest, require
from util.mvcl_relation_scaled import RelationScaledMVCL


def run(args):
    require(Path.cwd().resolve() == ROOT, 'Run from repository root.')
    require(not args.output.exists(), 'Use a new training output directory.')
    protocol_path = ROOT/'results/mvcl_formal_confirmation_protocol_v1/seal.json'
    protocol = read(protocol_path)
    for filename, expected in protocol['source_hashes'].items():
        require(digest(filename) == expected, 'Frozen protocol implementation/input changed: '+filename)
    require(args.dataset in protocol['protocol']['datasets'] and args.fold in protocol['protocol']['folds'],
            'Dataset/fold outside frozen scope.')
    data, pairs, labels = load_confirmation_bundle(args.data, args.dataset, args.fold)
    report = data['report']
    require(report['source_hashes'].get(str(protocol_path)) == digest(protocol_path), 'Bundle uses another protocol seal.')
    historical_path = ROOT/'results/batch_runs/sacti_outer_lockbox_seed901026/ranking'/args.dataset/'candidate'/('fold_%d'%args.fold)/'report.json'
    require(report['source_hashes'].get(str(historical_path)) == digest(historical_path),
            'Missing matching historical SACTI partition source.')
    historical = read(historical_path)['split']
    for name, old in [('train_sha256','model_train_sha256'),('validation_sha256','validation_sha256'),
                      ('outer_test_sha256','outer_test_sha256')]:
        require(report[name] == historical[old], 'SACTI partition mismatch.')
    config = copy.deepcopy(protocol['training_config'])
    config['model']['seed'] = protocol['protocol']['training_seed'] + args.fold - 1
    config['scope'] = 'frozen MVCL scaled relation-only matched confirmation; inner selection only'
    require(config['model']['seed'] == report['training_seed'], 'Training seed mismatch.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback.')
    torch.manual_seed(config['model']['seed'])
    model = RelationScaledMVCL(data, **config['model']).to(args.device)
    bundle_files = [p for p in args.data.iterdir() if p.is_file()]
    code = [Path(__file__), ROOT/'tools/mvcl_confirmation_inputs.py', ROOT/'tools/prepare_mvcl_confirmation_data.py',
            ROOT/'tools/run_mvcl_cti_pilot.py', *sorted((ROOT/'util').glob('mvcl*.py'))]
    hashes = {str(p.resolve()): digest(p) for p in bundle_files}
    seal = dict(protocol='mvcl_scaled_confirmation_fold_v1', dataset=args.dataset, fold=args.fold,
                config=config, data=str(args.data.resolve()), data_fingerprint=model.checkpoint_fingerprint,
                confirmation_seal_sha256=digest(protocol_path), bundle_source_hashes=hashes,
                code_hashes={str(p.relative_to(ROOT)):digest(p) for p in code},
                device=args.device, torch_version=str(torch.__version__), dry_run=args.dry_run,
                outer_test_scored=False, reconstruction_weights=model.reconstruction_weights,
                train_sha256=report['train_sha256'], validation_sha256=report['validation_sha256'],
                outer_test_sha256=report['outer_test_sha256'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'seal.json').write_text(json.dumps(seal, indent=2)+'\n')
    print('Fold:', args.fold, 'seed:', config['model']['seed'], 'weights:', model.reconstruction_weights, flush=True)
    if args.dry_run:
        print('Multi-fold preflight PASS; no encoder forward or optimizer steps.')
        return
    result = train(model, pairs, labels, config, args.output)
    require(all(digest(p)==h for p,h in hashes.items()), 'Bundle changed during training; do not evaluate.')
    require(all(digest(ROOT/p)==h for p,h in seal['code_hashes'].items()), 'Code changed during training.')
    (args.output/'confirmation_state.json').write_text(json.dumps(dict(training_complete=True,
        fold=args.fold, dataset=args.dataset, best_epoch=result['best_epoch'], outer_test_scored=False))+'\n')
    # The optimization loop is shared; label the result as a confirmation fold, not a pilot.
    (args.output/'summary.md').write_text('# MVCL Confirmation Fold Training\n\n'
        + 'Dataset: %s; fold: %d; selected epoch: %d.\n\n' % (args.dataset,args.fold,result['best_epoch'])
        + 'Inner AP: %.6f; inner AUC: %.6f.\n\n' % (result['validation']['AUPR'],result['validation']['AUC'])
        + 'Outer test not scored. This is training completion, not full confirmation completion.\n')
    print('Summary:', args.output/'summary.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=['tcmsuite','tcmsp','symmap','etcm_mention10'])
    parser.add_argument('--fold', required=True, type=int, choices=range(1,6))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Single-factor ETCM inner pilot; fixed relation-size correction, no outer scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tools.run_mvcl_cti_pilot import load_bundle, train
from util.mvcl_relation_scaled import RelationScaledMVCL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    baseline = ROOT/'results/batch_runs/mvcl_remaining_inner_v1/etcm_mention10'
    config = json.loads((ROOT/'configs/mvcl_etcm_relation_scaled_pilot.json').read_text())
    old = json.loads((baseline/'training/seal.json').read_text())
    comparable = json.loads(json.dumps(config))
    comparable['scope'] = old['config']['scope']
    comparable['model'].pop('reconstruction_scaling')
    if comparable != old['config'] or old['outer_test_scored'] or old['dry_run']:
        raise ValueError('Expected single-factor change from completed frozen inner baseline.')
    for filename, expected in old['code_hashes'].items():
        if hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Historical implementation changed: ' + filename)
    data, pairs, labels = load_bundle(baseline/'data')
    reference = json.loads(json.dumps(old['config']))
    from util.mvcl_cti_model import MVCLCTIReference
    if MVCLCTIReference(data, **reference['model']).checkpoint_fingerprint != old['data_fingerprint']:
        raise ValueError('Historical data identity mismatch.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback.')
    torch.manual_seed(config['model']['seed'])
    model = RelationScaledMVCL(data, **config['model']).to(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'tools/run_mvcl_cti_pilot.py', *sorted((ROOT/'util').glob('mvcl*.py'))]
    seal = dict(config=config, data_fingerprint=model.checkpoint_fingerprint,
                data=str(baseline/'data'), device=args.device, torch_version=str(torch.__version__),
                outer_test_scored=False, dry_run=args.dry_run,
                reconstruction_weights=model.reconstruction_weights,
                baseline_seal_sha256=hashlib.sha256((baseline/'training/seal.json').read_bytes()).hexdigest(),
                code_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'seal.json').write_text(json.dumps(seal, indent=2)+'\n')
    print('Reconstruction weights: ' + json.dumps(model.reconstruction_weights), flush=True)
    if args.dry_run:
        print('Single-factor preflight PASS; no forward or optimizer steps.')
        return
    train(model, pairs, labels, config, output)
    print('Summary: ' + str(output/'summary.md'))


if __name__ == '__main__':
    main()

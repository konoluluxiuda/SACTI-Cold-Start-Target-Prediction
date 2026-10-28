#!/usr/bin/env python3
"""Replay the published fixed comparison; no claim of a new unseen lockbox."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_data import verify
from tools.run_sacti_outer_lockbox import build_jobs, run_training, run_ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', choices=['tcmsuite', 'tcmsp', 'symmap', 'etcm_mention10'])
    parser.add_argument('--execute', action='store_true', help='Train five folds per model, then evaluate ranking.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the repository root.')
    verify(ROOT, json.loads((ROOT / 'data/manifest.json').read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'configs/sacti_outer_lockbox_seed901026.json').read_text())
    if args.dataset:
        manifest['datasets'] = [x for x in manifest['datasets'] if x['slug'] == args.dataset]
    jobs = build_jobs(manifest, output)
    protocol = dict(purpose='Reproduction of the already reported comparison',
                    fresh_unseen_lockbox=False, model_selection_reopened=False,
                    frozen_protocol='configs/sacti_outer_lockbox_seed901026.json', jobs=jobs)
    protocol_path = output / 'replay_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Replay scope differs from the existing output directory.')
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
    if not args.execute:
        print(json.dumps(dict(plan_only=True, jobs=jobs), indent=2))
        return
    run_training(manifest, jobs, sys.executable, args.resume)
    run_ranking(manifest, jobs, output, sys.executable, args.resume)
    print('Replay finished. Five-fold sampled metrics are in logs/; complete-candidate metrics are in ranking/.')


if __name__ == '__main__':
    main()

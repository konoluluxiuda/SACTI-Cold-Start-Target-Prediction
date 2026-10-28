#!/usr/bin/env python3
"""Replay ETCM annotation recovery using regenerated five-fold checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.prepare_data import digest, verify
from tools.run_sacti_outer_lockbox import checkpoint_prefixes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sacti-log', required=True, type=Path)
    parser.add_argument('--no-context-log', type=Path)
    parser.add_argument('--hctx-log', type=Path)
    parser.add_argument('--final-log', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the repository root.')
    verify(ROOT, json.loads((ROOT / 'data/manifest.json').read_text()))
    manifest = copy.deepcopy(json.loads((ROOT / 'configs/sacti_etcm_confirmed_target_evaluation.json').read_text()))
    logs = {'SACTI': args.sacti_log, 'NoContext': args.no_context_log,
            'Hctx-P': args.hctx_log, 'Final': args.final_log}
    methods = []
    for method in manifest['neural_methods']:
        log = logs[method['name']]
        if log is None:
            continue
        checkpoints = checkpoint_prefixes(log, 5)
        method['checkpoints'] = [dict(fold=fold, prefix=str(prefix),
                                      index_sha256=digest(str(prefix) + '.index'),
                                      data_sha256=digest(str(prefix) + '.data-00000-of-00001'))
                                 for fold, prefix in enumerate(checkpoints, start=1)]
        methods.append(method)
    manifest['neural_methods'] = methods
    available = {m['name'] for m in methods} | set(manifest['non_neural_methods']['names'])
    manifest['bootstrap']['comparisons'] = [x for x in manifest['bootstrap']['comparisons']
                                            if x['method'] in available]
    manifest['release_replay'] = dict(new_unseen_lockbox=False, model_selection_reopened=False,
                                     source_protocol='configs/sacti_etcm_confirmed_target_evaluation.json')
    manifest['reporting_boundaries'] = manifest['reporting_boundaries'][:5] + [
        'This replay uses regenerated checkpoints and does not establish a new unseen validation cohort.']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / 'replay_protocol.json'
    if protocol.exists() and json.loads(protocol.read_text()) != manifest:
        raise ValueError('Replay checkpoint list differs from the existing output directory.')
    protocol.write_text(json.dumps(manifest, indent=2) + '\n')
    command = [sys.executable, 'tools/run_etcm_confirmed_target_evaluation.py',
               '--manifest', str(protocol), '--output-dir', str(output / 'evaluation')]
    if not args.execute:
        command.append('--dry-run')
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

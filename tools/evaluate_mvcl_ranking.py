#!/usr/bin/env python3
"""Frozen best MVCL weights, inner full-candidate ranking; never train/select."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tools.run_mvcl_cti_pilot import load_bundle
from util.mvcl_relation_scaled import build_model as MVCLCTIReference
from util.mvcl_checkpoint import load_checkpoint
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking
from tools.evaluate_non_neural_cold_start_baselines import (
    herb_prototype_profiles, hc_jaccard_label_propagation)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def pair_scores(left, right, pairs, block=4096):
    if block < 1:
        raise ValueError('Positive block size required.')
    values = []
    for start in range(0, len(pairs), block):
        indices = torch.as_tensor(pairs[start:start + block], device=left.device)
        values.append((left[indices[:, 0]] * right[indices[:, 1]]).sum(1).cpu().numpy())
    scores = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores.')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/batch_runs/mvcl_cti_tcmsuite_convergence_400_v1')
    parser.add_argument('--data', default='results/mvcl_cti_data_tcmsuite_pilot_v2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    run, out = Path(args.run), Path(args.output)
    seal = json.loads((run / 'seal.json').read_text())
    selected = json.loads((run / 'summary.json').read_text())
    checkpoint = run / 'best_weights.pt'
    if seal['dry_run'] or seal['outer_test_scored'] or selected['outer_test_scored']:
        raise ValueError('Expected completed inner-only training.')
    if Path(selected['checkpoint']).resolve() != checkpoint.resolve():
        raise ValueError('Only the previously selected best checkpoint is permitted.')
    for filename, expected in seal['code_hashes'].items():
        if digest(ROOT / filename) != expected:
            raise ValueError('Training implementation changed: ' + filename)
    data, vp, vy = load_bundle(args.data)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback.')
    model = MVCLCTIReference(data, **seal['config']['model']).to(args.device)
    if model.checkpoint_fingerprint != seal['data_fingerprint']:
        raise ValueError('Training/input identity changed.')
    load_checkpoint(model, checkpoint)
    model.eval().requires_grad_(False)
    sources = {str(p): digest(p) for p in (checkpoint, run/'summary.json', run/'seal.json')}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(source_hashes=sources, best_epoch=selected['best_epoch'],
                    data_fingerprint=model.checkpoint_fingerprint,
                    train_sha256=data['report']['train_sha256'],
                    validation_sha256=data['report']['validation_sha256'],
                    candidate_ids_sha256=data['report']['candidate_ids_sha256'],
                    training_optimizer_steps=0, outer_test_scored=False,
                    evaluation_split='inner_validation', dry_run=args.dry_run,
                    device=args.device, torch_version=str(torch.__version__),
                    code_hashes={str(p.relative_to(ROOT)): digest(p) for p in (
                        Path(__file__), ROOT/'util/checkpoint_ranking.py',
                        ROOT/'tools/evaluate_non_neural_cold_start_baselines.py')},
                    decision='Descriptive only; no ranking-based checkpoint selection.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if args.dry_run:
        print('Checkpoint/input preflight PASS; zero forward/optimizer steps.')
        return
    with torch.no_grad():
        _, embeddings = model.encode()
        left = embeddings['compound'] @ model.decoder['C_P']
        right = embeddings['protein']
    sampled = pair_scores(left, right, vp)
    sampled_metrics = dict(AUPR=float(average_precision_score(vy, sampled)),
                           AUC=float(roc_auc_score(vy, sampled)))
    # This is an integrity tolerance, not an opportunity to choose a new epoch.
    if any(abs(sampled_metrics[k] - selected['validation'][k]) > 1e-5
           for k in sampled_metrics):
        (out/'sampled_mismatch.json').write_text(json.dumps(sampled_metrics, indent=2)+'\n')
        raise ValueError('Sampled metrics differ from frozen summary; no ranking issued.')
    ids = data['ids']
    ci = {x: i for i, x in enumerate(ids['compound'])}
    pi = {x: i for i, x in enumerate(ids['protein'])}
    cp, hc = data['relations']['C_P'], data['relations']['H_C']
    rows, cols = data['train_mask'].nonzero()
    train = [(ids['compound'][c], ids['protein'][p], int(cp[c, p])) for c, p in zip(rows, cols)]
    valid = [(ids['compound'][c], ids['protein'][p], int(y)) for (c, p), y in zip(vp, vy)]
    queries = np.unique(vp[:, 0])
    qi = {int(c): i for i, c in enumerate(queries)}
    prototype, _ = herb_prototype_profiles(hc, cp, queries, prior_strength=1.)
    jaccard, _ = hc_jaccard_label_propagation(hc, cp, queries)
    profiles = dict(prototype=prototype, jaccard=jaccard, raw_equal=(prototype+jaccard)*.5)
    reports = {}
    for arm in ('mvcl', *profiles):
        def score(cs, ps):
            pairs = np.array([(ci[c], pi[p]) for c, p in zip(cs, ps)], dtype=np.int64)
            if arm == 'mvcl':
                return pair_scores(left, right, pairs)
            return profiles[arm][[qi[int(c)] for c in pairs[:, 0]], pairs[:, 1]]
        ranking = evaluate_fixed_candidate_ranking(ids['protein'], train, valid, score,
                                                    ks=(20,), export_top=0)
        ranking['protocol'].update(evaluation_split='inner_validation',
                                  filter='inner_train_positive_pairs',
                                  targets='inner_validation_positive_pairs')
        values = score([r[0] for r in valid], [r[1] for r in valid])
        metrics = dict(AUPR=float(average_precision_score(vy, values)),
                       AUC=float(roc_auc_score(vy, values)), **ranking['metrics'])
        reports[arm] = metrics
        (out/(arm+'.json')).write_text(json.dumps(ranking, indent=2)+'\n')
    if any(digest(p) != h for p, h in sources.items()):
        raise ValueError('Source files changed during inference.')
    result = dict(protocol=protocol, metrics=reports, source_hashes_unchanged=True,
                  elapsed_seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Frozen MVCL Inner Ranking', '',
             '| Method | Sampled AP | Sampled AUC | MRR | Macro R@20 |',
             '|---|---:|---:|---:|---:|']
    for name, m in reports.items():
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f |' % (
            name, m['AUPR'], m['AUC'], m['MRR'], m['Recall@20']))
    lines += ['', 'Frozen best checkpoint; zero optimizer steps; no outer-test evaluation.',
              'Relation-only adaptation, not official reproduction. No superiority Gate.',
              'SACTI historical results are not imported without a matched ranking-protocol check.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Summary:', out/'summary.md')


if __name__ == '__main__':
    main()

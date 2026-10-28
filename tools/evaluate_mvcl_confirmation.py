#!/usr/bin/env python3
"""Evaluate a frozen confirmation checkpoint, without training or reselection."""
import argparse
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
from tools.mvcl_confirmation_inputs import load_confirmation_bundle
from tools.evaluate_mvcl_ranking import pair_scores
from tools.summarize_mvcl_sacti_matched import read, digest, require
from tools.evaluate_non_neural_cold_start_baselines import herb_prototype_profiles, hc_jaccard_label_propagation
from util.mvcl_checkpoint import load_checkpoint
from util.mvcl_relation_scaled import RelationScaledMVCL
from util.checkpoint_ranking import evaluate_fixed_candidate_ranking


def check_completion(seal, selected, state, run):
    require(seal['protocol'] == 'mvcl_scaled_confirmation_fold_v1', 'Not a formal fold.')
    require(not seal['dry_run'] and not seal['outer_test_scored'] and not selected['outer_test_scored'],
            'Completed inner-only training required.')
    require(state.get('training_complete') is True and not state['outer_test_scored'], 'Incomplete training.')
    require(all(state[k] == seal[k] for k in ('dataset', 'fold')), 'Completion identity mismatch.')
    require(state['best_epoch'] == selected['best_epoch'], 'Selected epoch mismatch.')
    require(Path(selected['checkpoint']).resolve() == (run/'best_weights.pt').resolve(),
            'Only the inner-selected checkpoint is permitted.')


def run(args):
    require(Path.cwd().resolve() == ROOT, 'Run from repository root.')
    require(not args.output.exists(), 'Use a new evaluation directory.')
    start = time.monotonic()
    seal, selected, state = [read(args.run/name) for name in ('seal.json', 'summary.json', 'confirmation_state.json')]
    check_completion(seal, selected, state, args.run)
    formal_path = ROOT/'results/mvcl_formal_confirmation_protocol_v1/seal.json'
    require(digest(formal_path) == seal['confirmation_seal_sha256'], 'Formal protocol changed.')
    formal = read(formal_path)
    sources = dict(formal['source_hashes'], **seal['bundle_source_hashes'])
    sources.update({str(ROOT/p): h for p,h in seal['code_hashes'].items()})
    for p,h in sources.items():
        require(digest(p) == h, 'Frozen source changed: '+p)
    for p in (formal_path, args.run/'best_weights.pt', args.run/'seal.json', args.run/'summary.json',
              args.run/'confirmation_state.json', Path(__file__), ROOT/'tools/evaluate_mvcl_ranking.py',
              ROOT/'util/checkpoint_ranking.py', ROOT/'tools/evaluate_non_neural_cold_start_baselines.py'):
        sources[str(p.resolve())] = digest(p)
    data, vp, vy = load_confirmation_bundle(Path(seal['data']), seal['dataset'], seal['fold'])
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no automatic CPU fallback.')
    model = RelationScaledMVCL(data, **seal['config']['model']).to(args.device)
    require(model.checkpoint_fingerprint == seal['data_fingerprint'], 'Checkpoint/input mismatch.')
    load_checkpoint(model, args.run/'best_weights.pt')
    model.eval().requires_grad_(False)
    with torch.no_grad():
        _, emb = model.encode()
        left, right = emb['compound'] @ model.decoder['C_P'], emb['protein']
    prediction = pair_scores(left, right, vp)
    inner = dict(AUPR=float(average_precision_score(vy, prediction)), AUC=float(roc_auc_score(vy, prediction)))
    require(all(abs(inner[k]-selected['validation'][k]) <= 1e-5 for k in inner),
            'Inner checkpoint replay mismatch; no outer scoring.')
    ids = data['ids']
    ci, pi = ({x:i for i,x in enumerate(ids[k])} for k in ('compound','protein'))
    outer = read(Path(seal['data'])/'outer_test.json')
    cp, hc = data['relations']['C_P'], data['relations']['H_C']
    rr, cc = data['train_mask'].nonzero()
    train = [(ids['compound'][c],ids['protein'][p],int(cp[c,p])) for c,p in zip(rr,cc)]
    queries = np.asarray(sorted({ci[c] for c,_,_ in outer}))
    require(cp[queries].nnz == 0, 'Outer query has training support.')
    qi = {int(c):i for i,c in enumerate(queries)}
    proto, _ = herb_prototype_profiles(hc,cp,queries,prior_strength=1.)
    jac, _ = hc_jaccard_label_propagation(hc,cp,queries)
    profiles = dict(prototype=proto,jaccard=jac,raw_equal=.5*(proto+jac))
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(dataset=seal['dataset'], fold=seal['fold'], best_epoch=selected['best_epoch'],
        evaluation_split='outer_test', outer_test_scored=True, training_optimizer_steps=0,
        new_lockbox_claim=False, model_selection_reopened=False, source_hashes=sources,
        inner_replay=inner, candidate_ids_sha256=data['report']['candidate_ids_sha256'],
        filter='inner_train_positive_pairs', data_fingerprint=model.checkpoint_fingerprint,
        **{k:seal[k] for k in ('train_sha256','validation_sha256','outer_test_sha256')})
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    reports = {}
    for arm in ('mvcl', *profiles):
        def score(cs, ps):
            pairs = np.asarray([(ci[c],pi[p]) for c,p in zip(cs,ps)],dtype=np.int64)
            if arm == 'mvcl':
                return pair_scores(left,right,pairs)
            return profiles[arm][[qi[int(c)] for c in pairs[:,0]],pairs[:,1]]
        ranking = evaluate_fixed_candidate_ranking(ids['protein'],train,outer,score,
            ks=tuple(formal['protocol']['ranking_ks']),export_top=0)
        values = score([r[0] for r in outer],[r[1] for r in outer])
        labels = [int(r[2]) for r in outer]
        reports[arm] = dict(AUPR=float(average_precision_score(labels,values)),
            AUC=float(roc_auc_score(labels,values)),**ranking['metrics'])
        ranking['protocol'].update(evaluation_split='outer_test',filter='inner_train_positive_pairs')
        (args.output/(arm+'.json')).write_text(json.dumps(ranking,indent=2)+'\n')
    require(all(digest(p)==h for p,h in sources.items()), 'Sources changed during evaluation.')
    result = dict(protocol=protocol,metrics=reports,source_hashes_unchanged=True,elapsed_seconds=time.monotonic()-start)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Frozen MVCL Outer Evaluation','', '| Method | Sampled AP | AUC | MRR | Recall@20 |',
             '|---|---:|---:|---:|---:|']
    lines += ['| %s | %.6f | %.6f | %.6f | %.6f |'%(k,m['AUPR'],m['AUC'],m['MRR'],m['Recall@20']) for k,m in reports.items()]
    lines += ['', 'Zero optimizer steps. All results reported; no checkpoint reselection.',
              'Relation-only adaptation; previously used outer split, not a new lockbox.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Summary:',args.output/'summary.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    torch.set_num_threads(4)
    run(p.parse_args())

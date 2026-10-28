#!/usr/bin/env python3
"""First-fold inner-validation MVCL-CTI pilot. Never scores outer records."""
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
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from util.mvcl_cti_data import matrix_hash
from util.mvcl_cti_model import MVCLCTIReference, RELATIONS
from util.mvcl_features import fixed_node_features
from util.mvcl_checkpoint import save_checkpoint
from util.mvcl_training_state import weights, restore_weights, save_training_state, load_training_state


def record_hash(records):
    lines = sorted('%s\t%s\t%d' % (c, p, y) for c, p, y in records)
    return hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()


def load_bundle(directory):
    directory = Path(directory)
    report = json.loads((directory / 'summary.json').read_text())
    ids = json.loads((directory / 'ids.json').read_text())
    records = json.loads((directory / 'validation.json').read_text())
    relations = {r: sparse.load_npz(directory / (r + '.npz')).tocsr() for r in RELATIONS}
    mask = sparse.load_npz(directory / 'train_mask.npz').tocsr()
    for r, m in dict(relations, train_mask=mask).items():
        if not np.all(m.data == 1) or matrix_hash(m) != report['matrices'][r]['sha256']:
            raise ValueError('Matrix integrity failure: ' + r)
    for r, (left, right) in RELATIONS.items():
        if relations[r].shape != (len(ids[left]), len(ids[right])):
            raise ValueError('Relation shape mismatch: ' + r)
    for a, b in [('H_C', 'C_H'), ('C_P', 'P_C'), ('P_D', 'D_P')]:
        if (relations[a] != relations[b].T).nnz:
            raise ValueError('Reverse relation mismatch.')
    if mask.shape != relations['C_P'].shape or (relations['C_P'] - relations['C_P'].multiply(mask)).nnz:
        raise ValueError('Training mask does not contain positives.')
    features, meta = fixed_node_features(ids)
    if meta['feature_sha256'] != report['features']['feature_sha256']:
        raise ValueError('Entity/features mismatch.')
    with np.load(directory / 'features.npz', allow_pickle=False) as stored:
        if any(not np.array_equal(stored[k], v) for k, v in features.items()):
            raise ValueError('Stored features changed.')
    if not records or any(y not in (0, 1) for _, _, y in records):
        raise ValueError('Nonempty binary validation required.')
    if record_hash(records) != report['validation_sha256']:
        raise ValueError('Validation hash mismatch.')
    if len({(c, p) for c, p, _ in records}) != len(records):
        raise ValueError('Duplicate validation pair.')
    ci = {x: i for i, x in enumerate(ids['compound'])}
    pi = {x: i for i, x in enumerate(ids['protein'])}
    pairs = np.array([(ci[c], pi[p]) for c, p, _ in records], dtype=np.int64)
    labels = np.array([y for _, _, y in records])
    if len(np.unique(labels)) != 2 or mask[np.unique(pairs[:, 0])].nnz:
        raise ValueError('Validation must be two-class and compound-disjoint.')
    rows, cols = mask.nonzero()
    training = [(ids['compound'][c], ids['protein'][p], int(relations['C_P'][c, p]))
                for c, p in zip(rows, cols)]
    if record_hash(training) != report['train_sha256'] or report['fold'] != 1:
        raise ValueError('Training identity/fold mismatch.')
    return dict(ids=ids, relations=relations, train_mask=mask, report=report), pairs, labels


@torch.no_grad()
def evaluate(model, pairs, labels):
    model.eval()
    _, embeddings = model.encode()
    device = next(model.parameters()).device
    scores = []
    for start in range(0, len(pairs), 4096):
        indices = torch.as_tensor(pairs[start:start + 4096], device=device)
        left = embeddings['compound'][indices[:, 0]] @ model.decoder['C_P']
        scores.append((left * embeddings['protein'][indices[:, 1]]).sum(1).cpu().numpy())
    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite validation scores.')
    return dict(AUPR=float(average_precision_score(labels, scores)),
                AUC=float(roc_auc_score(labels, scores)))


def train(model, pairs, labels, config, output, resume=None):
    output = Path(output)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    best, stale, best_epoch, checkpoint = -float('inf'), 0, 0, None
    best_weights, first_epoch, prior_elapsed = None, 1, 0.
    if resume:
        progress = load_training_state(model, optimizer, resume, config)
        best, stale, best_epoch = progress['best'], progress['stale'], progress['best_epoch']
        best_weights = progress['best_weights']
        first_epoch, prior_elapsed = progress['epoch'] + 1, progress['elapsed_seconds']
    start = time.monotonic()
    for epoch in range(first_epoch, config['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(contrast_weight=config['contrast_weight'], reg_lambda=config['reg_lambda'],
                       return_scores=False)
        loss = result['loss']
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss.')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Non-finite gradient.')
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), elapsed_seconds=time.monotonic()-start)
        row.update({k: float(result[k].detach()) for k in ('reconstruction', 'contrast', 'regularization')})
        row['weighted_contrast'] = config['contrast_weight'] * row['contrast']
        row['weighted_regularization'] = config['reg_lambda'] * row['regularization']
        del result, loss
        if epoch % config['eval_every'] == 0 or epoch == config['epochs']:
            row.update(evaluate(model, pairs, labels))
            if row['AUPR'] > best + config['min_delta']:
                best, best_epoch, stale = row['AUPR'], epoch, 0
                checkpoint = output / ('epoch_%03d.pt' % epoch)
                save_checkpoint(model, checkpoint)
                best_weights = weights(model)
            else:
                stale += 1
            row.update(best_AUPR=best, best_epoch=best_epoch, stale=stale)
        save_training_state(model, optimizer, output / 'last_training.pt', config,
            dict(epoch=epoch, best=best, stale=stale, best_epoch=best_epoch,
                 best_weights=best_weights, elapsed_seconds=prior_elapsed+time.monotonic()-start))
        with (output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if stale >= config['patience']:
            break
    restore_weights(model, best_weights)
    checkpoint = output / 'best_weights.pt'
    save_checkpoint(model, checkpoint)
    summary = dict(validation=evaluate(model, pairs, labels), best_epoch=best_epoch,
                   optimizer_steps=epoch, checkpoint=str(checkpoint), outer_test_scored=False,
                   elapsed_seconds=prior_elapsed+time.monotonic()-start, scope=config['scope'],
                   resumed_from=str(resume) if resume else None,
                   new_optimizer_steps=epoch-first_epoch+1)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'summary.md').write_text(
        '# MVCL-DTI-CTI Inner Pilot\n\n' + config['scope'] + '\n\n'
        + 'Best epoch: %d; validation AUPR: %.6f; AUC: %.6f.\n\n' %
        (best_epoch, summary['validation']['AUPR'], summary['validation']['AUC'])
        + 'Outer test scored: False. No formal superiority claim.\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs/mvcl_cti_tcmsuite_pilot.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', help='Full last_training.pt; output must be a NEW directory.')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if any(config[k] < 1 for k in ('epochs', 'eval_every', 'patience')):
        raise ValueError('Positive epoch/evaluation budgets required.')
    data, pairs, labels = load_bundle(args.data)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['model']['seed'])
    model = MVCLCTIReference(data, **config['model']).to(args.device)
    seal = dict(config=config, data_fingerprint=model.checkpoint_fingerprint,
                data=str(Path(args.data).resolve()), device=args.device, torch_version=str(torch.__version__),
                outer_test_scored=False, dry_run=args.dry_run,
                code_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), *sorted((ROOT / 'util').glob('mvcl*.py'))]})
    if args.resume:
        seal['resume_source'] = str(Path(args.resume).resolve())
        seal['resume_sha256'] = hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
    (output / 'seal.json').write_text(json.dumps(seal, indent=2) + '\n')
    if args.dry_run:
        print('Input/model preflight PASS; optimizer steps=0; no forward or outer scoring.')
        return
    train(model, pairs, labels, config, output, resume=args.resume)
    print('Summary: ' + str(output / 'summary.md'))


if __name__ == '__main__':
    main()

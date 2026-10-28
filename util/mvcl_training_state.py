"""Atomic training snapshots; distinct from weights-only MVCL checkpoints."""
import os
import hashlib
from pathlib import Path
import random
import numpy as np
import torch


def implementation_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('mvcl*.py'))}


def weights(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def restore_weights(model, saved):
    params = dict(model.named_parameters())
    if params.keys() != saved.keys():
        raise ValueError('Training parameter keys mismatch.')
    for n, p in params.items():
        if saved[n].shape != p.shape or saved[n].dtype != p.dtype or not torch.isfinite(saved[n]).all():
            raise ValueError('Invalid training parameter: ' + n)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n])


def save_training_state(model, optimizer, path, config, progress):
    state = np.random.get_state()
    payload = dict(format='mvcl_training_v1', fingerprint=model.checkpoint_fingerprint,
        config=config, parameters=weights(model), optimizer=optimizer.state_dict(), progress=progress,
        device=str(next(model.parameters()).device), torch_version=str(torch.__version__),
        implementation=implementation_hashes(),
        rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()
                 if next(model.parameters()).is_cuda else [], python=random.getstate(),
                 numpy=(state[0], state[1].tolist(), state[2], state[3], state[4])))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_state(model, optimizer, path, config):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'mvcl_training_v1':
        raise ValueError('Full training state required; weights-only checkpoints cannot resume.')
    if payload['fingerprint'] != model.checkpoint_fingerprint:
        raise ValueError('Training graph/model identity mismatch.')
    if payload['implementation'] != implementation_hashes():
        raise ValueError('Resume implementation changed.')
    for key in set(config) | set(payload['config']):
        if key != 'epochs' and config.get(key) != payload['config'].get(key):
            raise ValueError('Resume configuration changed: ' + key)
    progress = payload['progress']
    if config['epochs'] <= progress['epoch'] or progress['stale'] >= config['patience']:
        raise ValueError('Resume requires remaining budget and an untriggered early stop.')
    if (payload['device'] != str(next(model.parameters()).device)
            or payload['torch_version'] != str(torch.__version__)):
        raise ValueError('Resume requires the same device and Torch version.')
    restore_weights(model, payload['parameters'])
    optimizer.load_state_dict(payload['optimizer'])
    rng = payload['rng']
    torch.set_rng_state(rng['torch'])
    if rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])
    random.setstate(rng['python'])
    state = rng['numpy']
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    return progress

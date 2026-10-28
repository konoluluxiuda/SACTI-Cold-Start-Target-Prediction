"""Graph-bound model-weight checkpoints, not optimizer/RNG training resumes."""

from pathlib import Path
import torch


def save_checkpoint(model, path):
    path = Path(path)
    payload = dict(format='mvcl_cti_weights_v1', identity=model.checkpoint_identity,
                   fingerprint=model.checkpoint_fingerprint,
                   parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters()})
    # Graph buffers/operators are rebuilt from verified inputs, never overwritten.
    with path.open('xb') as handle:
        torch.save(payload, handle)


def load_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('format') != 'mvcl_cti_weights_v1'
            or payload.get('fingerprint') != model.checkpoint_fingerprint
            or payload.get('identity') != model.checkpoint_identity):
        raise ValueError('Checkpoint graph/data/model identity mismatch.')
    parameters = dict(model.named_parameters())
    saved = payload.get('parameters', {})
    if saved.keys() != parameters.keys():
        raise ValueError('Checkpoint parameter keys mismatch.')
    for name, value in saved.items():
        if (not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape
                or value.dtype != parameters[name].dtype or not torch.isfinite(value).all()):
            raise ValueError('Invalid checkpoint parameter: ' + name)
    with torch.no_grad():
        for name, value in saved.items():
            parameters[name].copy_(value)

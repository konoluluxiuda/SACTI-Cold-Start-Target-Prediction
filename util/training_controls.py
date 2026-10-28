"""Strict, opt-in negative controls for training-only audit runs."""

import hashlib

import numpy as np


def _config_bool(config, key, default=False):
    if not config.contains(key):
        return bool(default)
    value = str(config[key]).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError('Invalid boolean value for %s: %s' % (key, config[key]))


def resolve_label_permutation(config):
    enabled = _config_bool(config, 'training.label.permutation', False)
    seed = (
        int(config['training.label.permutation.seed'])
        if config.contains('training.label.permutation.seed') else 92026
    )
    if enabled:
        protocol = (
            str(config['experiment.protocol']).strip().lower()
            if config.contains('experiment.protocol') else 'legacy'
        )
        if protocol != 'strict':
            raise ValueError(
                'training.label.permutation requires experiment.protocol=strict.'
            )
        if not _config_bool(config, 'early.stopping', False):
            raise ValueError(
                'training.label.permutation requires inner validation.'
            )
        if _config_bool(config, 'evaluation.outer.test', True):
            raise ValueError(
                'training.label.permutation is a validation-only negative '
                'control; set evaluation.outer.test=False.'
            )
    return {'enabled': enabled, 'seed': seed}


def _records_sha256(records):
    text = '\n'.join(
        '%s\t%s\t%d' % (
            str(left), str(right), int(float(label) > 0)
        )
        for left, right, label in records
    ) + '\n'
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def permute_training_labels(records, seed):
    """Shuffle model-train labels while preserving pairs and class counts."""
    copied = [[str(row[0]), str(row[1]), float(row[2])] for row in records]
    if not copied:
        raise ValueError('Cannot permute an empty training set.')
    labels = np.asarray([
        1.0 if float(row[2]) > 0 else 0.0 for row in copied
    ], dtype=np.float32)
    if len(np.unique(labels)) != 2:
        raise ValueError('Label permutation requires both classes.')
    original_labels = labels.copy()
    rng = np.random.RandomState(int(seed))
    rng.shuffle(labels)
    fallback_rotation = False
    if np.array_equal(labels, original_labels):
        labels = np.roll(original_labels, 1)
        fallback_rotation = True
    for row, label in zip(copied, labels):
        row[2] = float(label)
    unchanged = int(np.sum(labels == original_labels))
    return copied, {
        'strategy': 'model_train_label_permutation',
        'seed': int(seed),
        'records': len(copied),
        'positive_count': int(np.sum(labels > 0)),
        'negative_count': int(np.sum(labels <= 0)),
        'unchanged_label_count': unchanged,
        'changed_label_count': int(len(labels) - unchanged),
        'fallback_rotation_used': fallback_rotation,
        'source_records_sha256': _records_sha256(records),
        'permuted_records_sha256': _records_sha256(copied),
    }

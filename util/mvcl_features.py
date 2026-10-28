"""Frozen, label-independent inputs for the relation-only MVCL adaptation."""

import hashlib
import json

import numpy as np

from util.mvcl_cti_data import NODE_TYPES

FEATURE_DIMENSION = 128
FEATURE_SEED = 52026


def fixed_node_features(ids, dimension=FEATURE_DIMENSION, seed=FEATURE_SEED):
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError('Feature dimension must be a positive integer.')
    arrays, hashes = {}, {}
    for kind in NODE_TYPES:
        values = [str(x) for x in ids[kind]]
        if len(values) != len(set(values)):
            raise ValueError('Duplicate typed node IDs.')
        array = np.empty((len(values), dimension), dtype=np.float32)
        for row, value in enumerate(values):
            key = json.dumps([int(seed), kind, value], ensure_ascii=True).encode()
            # Per-entity streams keep features stable under ordering/universe changes.
            entropy = np.frombuffer(hashlib.sha256(key).digest(), dtype='<u4').tolist()
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
            array[row] = rng.standard_normal(dimension).astype(np.float32)
        arrays[kind] = array
        hashes[kind] = hashlib.sha256(array.astype('<f4').tobytes()).hexdigest()
    return arrays, dict(policy='typed_id_fixed_gaussian_v1', dimension=dimension,
                        seed=seed, trainable=False, labels_used=False,
                        numpy_version=np.__version__, feature_sha256=hashes)

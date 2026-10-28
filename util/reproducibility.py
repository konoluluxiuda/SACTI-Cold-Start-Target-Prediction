import os
import random

import numpy as np


def seed_python_numpy(seed):
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_tensorflow(seed, reset_graph=False):
    import tensorflow.compat.v1 as tf

    if reset_graph:
        tf.reset_default_graph()
    tf.set_random_seed(int(seed))


def set_global_seed(seed, reset_tensorflow_graph=False):
    seed_python_numpy(seed)
    seed_tensorflow(seed, reset_graph=reset_tensorflow_graph)

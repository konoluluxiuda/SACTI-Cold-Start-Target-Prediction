import os


def _conf_get(conf, key, default=None):
    if conf is not None and hasattr(conf, 'contains') and conf.contains(key):
        return conf[key]
    return default


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def configure_cuda_environment(conf=None):
    """Set CUDA visibility before TensorFlow is imported."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
    if _as_bool(os.environ.get('HDCTI_FORCE_CPU'), False):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return

    gpu_id = _conf_get(conf, 'gpu.id', os.environ.get('HDCTI_GPU', '0'))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(gpu_id))
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')


def make_session_config(tf, conf=None):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = _as_bool(_conf_get(conf, 'gpu.log_device_placement'), False)

    if _as_bool(_conf_get(conf, 'gpu.allow_growth'), True):
        config.gpu_options.allow_growth = True
    elif _conf_get(conf, 'gpu.memory_fraction') is not None:
        config.gpu_options.per_process_gpu_memory_fraction = float(_conf_get(conf, 'gpu.memory_fraction'))

    return config


def configure_tensorflow_runtime(tf, conf=None):
    """Best-effort TensorFlow 2.x GPU runtime configuration."""
    if not _as_bool(_conf_get(conf, 'gpu.allow_growth'), True):
        return
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

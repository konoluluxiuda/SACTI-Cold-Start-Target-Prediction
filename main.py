# Derived from https://github.com/tong87-bio/HDCTI (MIT License).
# Modified to support explicit experiment configuration paths and fixed seeds.
from util.config import ModelConf
from util.gpu import configure_cuda_environment
from util.reproducibility import seed_python_numpy
import argparse
import os


if __name__ == '__main__':
    import time
    s = time.time()
    parser = argparse.ArgumentParser(description='Run an HDCTI experiment.')
    parser.add_argument(
        '--config',
        default=os.environ.get('HDCTI_CONFIG', 'HDCTI.conf'),
        help='Path to the experiment configuration file.',
    )
    args = parser.parse_args()

    try:
        conf = ModelConf(args.config)
    except KeyError:
        print('wrong num!')
        exit(-1)
    print('Experiment config:', os.path.abspath(args.config))
    seed_python_numpy(int(conf['random.seed']) if conf.contains('random.seed') else 2026)
    configure_cuda_environment(conf)

    from HDR import HDR

    recSys = HDR(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))

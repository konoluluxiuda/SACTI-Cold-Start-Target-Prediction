# SACTI: Herb-Context Transfer for Compound Cold-Start Target Prediction

Implementation, processed numerical inputs, fixed compound-disjoint folds and
frozen result summaries for the SACTI manuscript. The included data package
supports training from the processed relations; raw database pages are not
needed for this workflow. Trained checkpoint binaries are not included.

## Install and check the data

```bash
git clone https://github.com/konoluluxiuda/SACTI-Cold-Start-Target-Prediction.git
cd SACTI-Cold-Start-Target-Prediction
python tools/verify_release.py
python tools/prepare_data.py
python tools/prepare_data.py --verify
```

The installer uses the bundled `data/processed_inputs.tar.gz` (about 35 MB),
checks every archived file against `data/manifest.json`, and installs the
four datasets under `dataset/`. It updates machine-specific paths in the
runtime split metadata and MVCL protocol while preserving all numerical input
files. `data/installed_manifest.json` records original and installed hashes.
Historical result metadata retain their original paths as provenance only.

## SACTI environment and training

The TensorFlow experiments used Python 3.10. Create a dedicated environment
and install `requirements.txt` before running the following checks:

```bash
python -m pip install -r requirements.txt
python tools/verify_inputs.py
python -m unittest discover -s tests -p test_release_data.py -v
python -m unittest discover -s tests -p test_sacti_model_smoke.py -v
```

`verify_inputs.py` checks all 40 released outer folds, compound isolation,
the exact 20 historical confirmation train/validation/test partitions, and
the 685 ETCM2.0 evidence pairs. The model smoke test runs short synthetic
training and checkpoint checks on CPU.

For a first-fold, inner-validation training run on the processed TCM-Suite data:

```bash
python main.py --config configs/SACTI_tcmsuite_inner_pilot.conf
```

To replay the full five-fold SACTI versus predecessor comparison:

```bash
python tools/reproduce_sacti.py --output results/replay_sacti
python tools/reproduce_sacti.py --output results/replay_sacti --execute
```

The first command prepares and prints the plan. The second trains all five
folds for each configuration and evaluates full-candidate ranking after
training. Use `--dataset tcmsuite` to select one database, and `--resume` to
reuse completed jobs. This is reproduction of the reported comparison,
not a new unseen lockbox.

## Component ablation and neural controls

```bash
python tools/run_sacti_component_ablation.py --run-dir results/replay_ablation --dry-run
python tools/run_sacti_component_ablation.py --run-dir results/replay_ablation
bash run_cold_start_external_baselines_full.sh --dry-run
bash run_cold_start_external_baselines_full.sh
```

The controls include Dual-HGNN-CTI, LightGCN-CTI, R-GCN-CTI and HGT-CTI.
The fixed inputs and configurations match the reported relation-only task.

After component-ablation training, evaluate the regenerated full ETCM model
against all held-out annotations:

```bash
python tools/reproduce_etcm_evidence.py --sacti-log results/replay_ablation/logs/etcm_mention10_full.log --output results/replay_etcm
python tools/reproduce_etcm_evidence.py --sacti-log results/replay_ablation/logs/etcm_mention10_full.log --output results/replay_etcm --execute
```

Optional `--no-context-log`, `--hctx-log` and `--final-log` arguments add
regenerated reference models. The archived original per-pair results are
included for direct inspection without retraining.

For the reported positive-to-unlabelled ratio sensitivity, the shared
`tools/evaluate_checkpoint_ranking.py` entry point accepts
`--negative-ratios 1 5 10 --negative-ratio-universe cp_positive
--negative-ratio-seed <252026 + fold - 1> --skip-full-candidate`, together with
the replay configuration, checkpoint prefix and one-based fold. This evaluates
the same trained model; it does not retrain or select a checkpoint by the ratio.

## MVCL comparison

MVCL uses a separate Python 3.9/PyTorch environment. For the CUDA 12.8 build
used in the original experiment:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-mvcl.txt
python tools/run_mvcl_confirmation.py --output results/replay_mvcl
python tools/run_mvcl_confirmation.py --output results/replay_mvcl --execute
```

The runner reconstructs fold-local graph bundles from the released processed
relations and fixed folds, trains all 20 models with the archived training
configuration, then evaluates them. The full batch expects CUDA. Single-fold
workers also accept `--device cpu`. See `util/MVCL_ADAPTATION_NOTICE.md` for
the distinction between this relation-only adaptation and the original model.

## Results and provenance

`paper_artifacts/results/` contains archived numerical results and a checksum
manifest. `figures/manuscript_main/*_source.tsv` contains plot source tables.
To regenerate the four frozen result tables without model training:

```bash
python tools/build_manuscript_tables.py --output results/replay_tables.md
```

[DATA_ACCESS.md](DATA_ACCESS.md) describes the processed files and ETCM2.0
annotation-recovery inputs. [UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md)
documents code reused from HDCTI under its MIT license. The code license does
not replace source databases' terms or attribution requirements.

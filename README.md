# SACTI: Herb-Context Transfer for Compound Cold-Start Target Prediction

This repository contains a **code and frozen-result snapshot** for the SACTI
manuscript. It is not yet a complete, independently executable reproduction
package: the four training datasets, frozen split files, held-out ETCM2.0
evidence, and trained checkpoints have not been released here. See
[DATA_ACCESS.md](DATA_ACCESS.md) before attempting to reproduce the results.

## What is included

| Location | Contents |
|---|---|
| `SACTI.py`, `HDCTI.py`, `HDR.py`, `main.py`, `base/`, `util/` | Model and shared training code |
| `configs/` | Frozen SACTI, matched predecessor, ablation, and evaluation protocols |
| `tools/` | Dataset construction and evaluation scripts used in the reported workflow |
| `paper_artifacts/results/`, `results/` | Selected frozen numerical summaries |
| `figures/manuscript_main/*_source.tsv` | Numerical source tables for manuscript plots |
| `tests/` | Focused code and protocol tests |
| `FILE_MANIFEST.json` | SHA-256 hashes for copied source files; `source_sha256` records the local source snapshot before path sanitization |

The code's HDCTI ancestry is documented in
[UPSTREAM_PROVENANCE.md](UPSTREAM_PROVENANCE.md). The repository's MIT license
applies to code; it does not grant rights to redistribute third-party data.

## Environment

The SACTI training environment is specified by `requirements.txt` and
`environment.yml`. MVCL comparison scripts require a separate PyTorch
environment and are included for method transparency; they have not been
validated as a standalone install from this snapshot. Once the exact input
datasets are available, inspect the frozen protocols and run a dry run before
training:

```bash
python tools/run_sacti_outer_lockbox.py --dry-run
python tools/run_sacti_component_ablation.py --dry-run
```

These commands do not supply the missing data or checkpoints. No claim of
end-to-end reproduction is made for the current release.

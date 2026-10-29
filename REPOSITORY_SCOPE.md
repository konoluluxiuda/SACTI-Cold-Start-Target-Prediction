# Paper release scope

This repository contains the implementation and reproducibility materials for
the current SACTI manuscript.

The project title is *Herb-Context Transfer for Compound Cold-Start Target
Prediction*. Its runtime environment, command-line entry point and launcher
use SACTI naming. Names of published comparator models and upstream copyright
notices retain their original identities; they do not name the SACTI project.

| Location | Role in the paper |
|---|---|
| `SACTI.py`, `HDR.py`, `rating.py`, `base/`, `util/` | Final model, shared data loading and required model dependencies |
| `HDCTI.py`, `LightGCNCTI.py`, `RGCNCTI.py`, `HGTCTI.py` | Matched predecessor and neural controls, including the Dual-HGNN profile |
| `configs/` | Frozen final-model, ablation, comparator and annotation-recovery configurations |
| `data/` | Processed four-dataset input archive, file inventory and acquisition-independent installation |
| `tools/reproduce_*.py`, `tools/run_sacti_*.py`, MVCL workers | Training and evaluation for the reported experiments |
| `tools/build_etcm2_*.py`, `tools/create_etcm2_*.py` | Provenance of the ETCM processing used to build the released inputs |
| `tools/experiment_io.py` | Shared helpers extracted from historical experiment launchers |
| `tools/summarize_mvcl_sacti_matched.py` | Small compatibility import used by the archived MVCL protocol |
| `paper_artifacts/results/` | Nine current-paper result/evidence files and their checksum manifest |
| `figures/manuscript_main/` | Current result-figure source tables and plotting code |
| `tests/`, `reproducibility/` | Focused correctness checks and release verification outputs |

Early random-edge, CHCR, router, preliminary SCHPT/DG-SCHPT, donor-reweighting
and query-path experiment launchers/results are excluded from the current
release tree. Some internal helper modules are still imported by the frozen
predecessor and shared model code; retaining them preserves executable model
dependencies without adding those historical experiments to the paper.

See [data/README.md](data/README.md) for the exact uploaded data inventory.

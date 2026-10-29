# Upstream Provenance

## Original Work

This codebase was initialized from the public HDCTI repository:

- Repository: https://github.com/tong87-bio/HDCTI
- Article: https://doi.org/10.1093/bib/bbaf399
- Original copyright: `Copyright (c) 2024 tong87-bio`
- License: MIT

The upstream Git history was not preserved when this research repository was
created. This document restores explicit provenance and must remain with all
public releases of the derived code.

## Inherited Backbone

The following areas retain substantial upstream structure or implementation:

- `HDCTI.py`: H-C/P-D hypergraph propagation, embedding backbone, legacy
  PageRank/attention-compatible code paths, and training loop foundations;
- `HDR.py`, `rating.py`: experiment orchestration and dataset loading;
- `base/`: recommender abstractions and evaluation flow;
- `util/config.py`, `util/io.py`, `util/log.py`: configuration and I/O helpers;
- `main.py`: command-line training entry point.

These components are not claimed as new contributions of the derived work.

## Current Paper Implementation

`SACTI.py` defines a separate model class with sparse residual membership
encoders, protein identity--disease context fusion, support-gated identity
scoring, herb-context matching and two fold-local LOCO transfer scores.
`HDCTI.py` remains necessary for the matched predecessor and Dual-HGNN control.
The runtime entry point, data loader and recommender abstractions retain
upstream code and are covered by the original MIT notice.

The current implementation also includes compound-disjoint partitions,
inner-validation checkpoint selection, controlled neural comparisons,
annotation-recovery evaluation and checksum-verified processed inputs.
See `REPOSITORY_SCOPE.md` and the frozen `configs/` files for the current
paper workflow. Historical results outside that scope are omitted from the
current release tree.

## Citation and Reuse

Any publication using this repository should cite the original HDCTI article
for the inherited backbone and separately cite the derived paper when its
bibliographic record becomes available. Reuse of figures, data, or text from
third-party sources remains subject to their respective licenses and terms.


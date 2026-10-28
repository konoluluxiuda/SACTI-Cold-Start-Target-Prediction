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

## Derived Research Contributions

The current paper pipeline adds or materially changes:

1. strict, reusable pair-stratified and compound-cold split manifests;
2. fold-local graph/statistic construction and type-safe bipartite PageRank;
3. fixed random seeds, inner-validation early stopping, checkpoint restoration,
   and five-fold metric aggregation;
4. Hctx-P candidate-level herb-context interaction;
5. SDIS support-conditioned suppression of an unreliable compound-ID branch;
6. SCHPT leave-one-compound-out, prior-smoothed herb-target prototype transfer;
7. same-input external baselines, data preparation utilities, result manifests,
   and hash-verified paper artifacts.

The exact final method is defined in
`docs/FINAL_METHOD_SPECIFICATION.md`. Historical experiments and failed
candidates remain available through Git history but are intentionally omitted
from the paper release tree.

## Citation and Reuse

Any publication using this repository should cite the original HDCTI article
for the inherited backbone and separately cite the derived paper when its
bibliographic record becomes available. Reuse of figures, data, or text from
third-party sources remains subject to their respective licenses and terms.


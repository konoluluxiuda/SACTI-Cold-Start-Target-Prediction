# Data and checkpoint availability

## Current release boundary

This snapshot intentionally excludes the four processed training datasets,
their frozen compound-disjoint split files, the ETCM2.0 held-out annotation
set and its complete pair-level rank table, and model checkpoint binaries. The existing local preparation notes
identify source-specific redistribution conditions that have not been cleared.
Do not interpret the presence of preprocessing scripts or result summaries as
availability of the exact inputs needed to rerun the paper.

The training datasets are TCM-Suite, TCMSP, SymMap2.0, and the
ETCM2.0-mention10 construction. The original HDCTI data/source release for
some of these inputs is at <https://github.com/tong87-bio/HDCTI>. ETCM2.0
records must be obtained from their database provider under its terms.
The exact relation inputs, filters, entity mappings, sampled evaluation pairs,
and split assignments still require a tested, lawful acquisition route.

## Minimum missing material for numerical reproduction

- Processed `H_C.txt`, `C_P.txt`, `P_D.txt`, label-pair files, and entity maps
  for all four datasets, or tested scripts to regenerate identical files from
  publicly accessible sources.
- Four datasets' frozen split assignments for the development and sealed
  partitions, with checksums and fixed negative-pair sampling.
- The ETCM2.0 `unseen_confirmed.tsv` evidence file or a lawful deterministic
  reconstruction route; the protocol and expected hashes are in
  `configs/sacti_etcm_confirmed_target_evaluation.json`.
- Trained checkpoint files if exact checkpoint-only re-evaluation is claimed;
  otherwise the released inputs and scripts must permit retraining.
- MVCL comparison seal and run artifacts needed by the formal comparison
  scripts. The current `results/` folder contains only a compact summary.

Until these items are accessible to editors and reviewers, **a repository URL
alone does not satisfy a complete Data and Software Availability requirement**.
The manuscript availability statement must describe the actual release state,
not promise materials that are absent here.

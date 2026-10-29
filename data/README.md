# What is in the processed data package?

`processed_inputs.tar.gz` is a 35,447,652-byte archive containing the exact
processed relation inputs, sampled-pair sources and fixed folds used by the
paper. SHA-256: `b21a6df1745cae7fa5a2842d01524837bf5c7b304b05c5a6a94fd34094b8a9a0`.
Install it with `python tools/prepare_data.py` from the repository root.

## Dataset contents

Counts below are unique relations and eligible evaluation entities, not raw
line counts. Original duplicate rows are retained in the frozen input files;
the loader applies the same deduplication as the reported experiments.

| Dataset | Evaluated compounds | Candidate proteins | H-C relations | Positive C-P relations | P-D relations |
|---|---:|---:|---:|---:|---:|
| TCM-Suite | 1,187 | 7,258 | 6,496 | 43,669 | 44,170 |
| TCMSP | 6,929 | 1,748 | 33,756 | 56,102 | 365 |
| SymMap2.0 | 1,618 | 18,192 | 85,172 | 37,991 | 194,558 |
| ETCM2.0-mention10 | 9,519 | 509 | 25,606 | 88,431 | 1,824,967 |

H-C records herb membership, C-P records the known compound-target relations,
and P-D records protein-disease membership. During training, C-P-derived
quantities are reconstructed from the current fold's training positives.
The complete C-P table also preserves the entity universe across folds.

## File groups

| Files | Count | Purpose |
|---|---:|---|
| Relation, positive-pair and unlabelled-candidate inputs | 19 | Train the four datasets and reconstruct the original sampled splits |
| Entity mapping tables | 12 | Link processed identifiers to the available source identifiers/names |
| Two five-fold partitions per dataset | 56 | Eight assignment tables, eight manifests and 40 test-fold files |
| ETCM held-out annotation table and source manifest | 2 | Evaluate 685 held-out pairs involving 332 compounds |
| Fixed partition reports and MVCL protocol metadata | 23 | Verify historical train/validation/test identities and reuse the frozen comparator settings |
| **Total** | **112** | |

The candidate files named `zero`/`ZERO` contain sampled or eligible unobserved
pairs, not experimentally verified non-interactions. TCMSP has no separate
negative-candidate file; its compound-matched unlabelled pairs are generated
by the frozen Cartesian sampling procedure. Fixed evaluation pairs are
already recorded in the released fold files.

ETCM `unseen_confirmed.tsv` is used only for annotation recovery, not model
training or early stopping. All 685 pairs are absent from the model C-P graph.
The package contains no raw ETCM page collection, unrelated downloaded papers,
manuscript drafts, training logs or trained checkpoint binaries.

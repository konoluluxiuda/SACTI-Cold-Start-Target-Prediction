# Frozen Manuscript Tables

> Generated from SHA-256-verified result artifacts and local relation files. No checkpoint is restored and no training or outer-test rescoring is performed.

## Table 1

**Table 1. Graph scale and compound cold-start evaluation coverage.** H and D are unique entities occurring in H-C and P-D, respectively; C is the set of compounds with at least one known C-P link and therefore eligible for compound-disjoint evaluation; P candidates is the complete mapped protein candidate set. Relation counts are unique pairs. H-C coverage is the fraction of evaluated compounds with at least one H-C link.

| Dataset | H | C | P candidates | D | H-C | C-P+ | P-D | H-C coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 1,009 | 1,187 | 7,258 | 11,071 | 6,496 | 43,669 | 44,170 | 100.00% |
| TCMSP | 501 | 6,929 | 1,748 | 321 | 33,756 | 56,102 | 365 | 99.68% |
| SymMap2.0 | 686 | 1,618 | 18,192 | 6,155 | 85,172 | 37,991 | 194,558 | 98.64% |
| ETCM2.0-mention10 | 1,757 | 9,519 | 509 | 7,693 | 25,606 | 88,431 | 1,824,967 | 100.00% |

## Table 2

**Table 2. Unseen seed-901026 lockbox comparison between SACTI and the matched HDCTI-based predecessor.** Values are means across five compound-disjoint outer folds. AUPR denotes average precision; MRR is macro-averaged over compounds within each fold. Macro is the unweighted mean of database means. Delta is SACTI minus predecessor. The lockbox was sealed before split generation and reported regardless of direction.

| Dataset | Pred. AUPR | SACTI AUPR | Delta | Pred. MRR | SACTI MRR | Delta | Pred. R@20 | SACTI R@20 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.736347 | 0.787528 | +0.051181 | 0.135344 | 0.207247 | +0.071903 | 0.072811 | 0.106102 | +0.033291 |
| TCMSP | 0.959680 | 0.968892 | +0.009212 | 0.561915 | 0.569710 | +0.007795 | 0.584569 | 0.557681 | -0.026889 |
| SymMap2.0 | 0.842451 | 0.847696 | +0.005245 | 0.337812 | 0.428082 | +0.090270 | 0.215197 | 0.319649 | +0.104452 |
| ETCM2.0-mention10 | 0.922747 | 0.942283 | +0.019535 | 0.607131 | 0.621065 | +0.013935 | 0.572521 | 0.588271 | +0.015750 |
| Macro | 0.865306 | 0.886600 | +0.021293 | 0.410550 | 0.456526 | +0.045976 | 0.361275 | 0.392926 | +0.031651 |

## Table 3

**Table 3. Frozen seed-52026 SACTI component ablation.** Values are mean AUPR (average precision) across five outer folds on fixed 1:1 positive-to-unlabelled pairs. Macro is the unweighted mean of database means. Prototype and Jaccard retain only the named transfer branch; No gate and No context remove the indicated native component from full SACTI.

| Dataset | ID + H-C context | + Prototype | + Jaccard | Full SACTI | No gate | No H-C context |
|---|---:|---:|---:|---:|---:|---:|
| TCM-Suite | 0.779791 | 0.784427 | 0.785223 | 0.787323 | 0.780354 | 0.674692 |
| TCMSP | 0.966857 | 0.968206 | 0.968041 | 0.968634 | 0.968179 | 0.863690 |
| SymMap2.0 | 0.841859 | 0.844317 | 0.843513 | 0.844403 | 0.841890 | 0.854464 |
| ETCM2.0-mention10 | 0.936401 | 0.940315 | 0.940767 | 0.942788 | 0.942744 | 0.839007 |
| Macro | 0.881227 | 0.884316 | 0.884386 | 0.885787 | 0.883292 | 0.807963 |

## Table 4

**Table 4. Compound cold-start comparison with same-input neural controls.** Values are mean ± sample SD (ddof=1) of AUPR (average precision) across five outer folds. Macro is the unweighted mean of database means. Controls are topology-budget adaptations rather than full reproductions of their original tasks or attribute inputs.

| Method | TCM-Suite | TCMSP | SymMap2.0 | ETCM2.0-mention10 | Macro |
|---|---:|---:|---:|---:|---:|
| Dual-HGNN-CTI | 0.729257 ± 0.026157 | 0.939348 ± 0.002513 | 0.830540 ± 0.014266 | 0.910730 ± 0.003993 | 0.852469 |
| LightGCN-CTI | 0.568966 ± 0.010352 | 0.662029 ± 0.016537 | 0.595482 ± 0.016050 | 0.608434 ± 0.010306 | 0.608728 |
| R-GCN-CTI | 0.670161 ± 0.028661 | 0.910556 ± 0.009702 | 0.812299 ± 0.026080 | 0.809541 ± 0.012075 | 0.800639 |
| HGT-CTI | 0.696228 ± 0.066732 | 0.934128 ± 0.006916 | 0.839690 ± 0.012176 | 0.846547 ± 0.010724 | 0.829148 |
| SACTI | **0.787323 ± 0.010796** | **0.968634 ± 0.002666** | **0.844403 ± 0.009661** | **0.942788 ± 0.003803** | **0.885787** |

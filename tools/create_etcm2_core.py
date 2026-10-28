#!/usr/bin/env python3
"""Create a strongly connected ETCM2.0 core dataset.

Core definition:
- Keep C_P edges whose compound appears in H_C and protein appears in P_D.
- Keep H_C edges for core compounds.
- Keep P_D edges for core proteins.
- Keep H_D edges connecting herbs/diseases retained by H_C/P_D.
- Generate 1:1 negative C_P samples as ZERO_indices.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


Edge = Tuple[int, int]


def read_edges(path: Path) -> Set[Edge]:
    edges: Set[Edge] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            edges.add((int(parts[0]), int(parts[1])))
    return edges


def write_edges(path: Path, edges: Iterable[Edge], with_label: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for left, right in sorted(edges):
            if with_label:
                f.write(f"{left}\t{right}\t1\n")
            else:
                f.write(f"{left}\t{right}\n")


def write_zero(path: Path, edges: Iterable[Edge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for left, right in sorted(edges):
            f.write(f"{left}\t{right}\t0\n")


def sample_negative_edges(
    compounds: Sequence[int],
    proteins: Sequence[int],
    positives: Set[Edge],
    count: int,
    seed: int,
) -> Set[Edge]:
    total_candidates = len(compounds) * len(proteins) - len(positives)
    if total_candidates < count:
        raise ValueError(
            f"Not enough negative candidates: requested {count}, available {total_candidates}"
        )

    rng = random.Random(seed)
    negatives: Set[Edge] = set()
    compounds = list(compounds)
    proteins = list(proteins)
    while len(negatives) < count:
        edge = (rng.choice(compounds), rng.choice(proteins))
        if edge in positives or edge in negatives:
            continue
        negatives.add(edge)
    return negatives


def filter_mapping_csv(src: Path, dst: Path, id_column: str, keep_ids: Set[int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8", newline="") as fin, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row[id_column]) in keep_ids:
                writer.writerow(row)


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100.0 / total, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/ETCM2.0_processed")
    parser.add_argument("--output", default="dataset/ETCM2.0_core")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Negative:positive ratio for ZERO_indices.txt",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    stats_dir = dst / "stats"
    mapping_dir = dst / "mappings"

    hc = read_edges(src / "H_C.txt")
    cp = read_edges(src / "C_P.txt")
    pd = read_edges(src / "P_D.txt")
    hd = read_edges(src / "H_D.txt")

    hc_compounds = {compound for _, compound in hc}
    pd_proteins = {protein for protein, _ in pd}
    core_cp = {(compound, protein) for compound, protein in cp if compound in hc_compounds and protein in pd_proteins}
    core_compounds = {compound for compound, _ in core_cp}
    core_proteins = {protein for _, protein in core_cp}

    core_hc = {(herb, compound) for herb, compound in hc if compound in core_compounds}
    core_pd = {(protein, disease) for protein, disease in pd if protein in core_proteins}
    core_herbs = {herb for herb, _ in core_hc}
    core_diseases = {disease for _, disease in core_pd}
    core_hd = {(herb, disease) for herb, disease in hd if herb in core_herbs and disease in core_diseases}

    zero_count = int(round(len(core_cp) * args.negative_ratio))
    zero_edges = sample_negative_edges(
        sorted(core_compounds),
        sorted(core_proteins),
        core_cp,
        zero_count,
        args.seed,
    )

    dst.mkdir(parents=True, exist_ok=True)
    write_edges(dst / "H_C.txt", core_hc, with_label=True)
    write_edges(dst / "C_P.txt", core_cp, with_label=False)
    write_edges(dst / "P_D.txt", core_pd, with_label=True)
    write_edges(dst / "H_D.txt", core_hd, with_label=True)
    write_edges(dst / "ONE_indices.txt", core_cp, with_label=True)
    write_zero(dst / "ZERO_indices.txt", zero_edges)

    filter_mapping_csv(
        src / "mappings" / "herb_id_map.csv",
        mapping_dir / "herb_id_map.csv",
        "herb_id",
        core_herbs | {herb for herb, _ in core_hd},
    )
    filter_mapping_csv(
        src / "mappings" / "compound_id_map.csv",
        mapping_dir / "compound_id_map.csv",
        "compound_id",
        core_compounds,
    )
    filter_mapping_csv(
        src / "mappings" / "protein_id_map.csv",
        mapping_dir / "protein_id_map.csv",
        "protein_id",
        core_proteins,
    )
    filter_mapping_csv(
        src / "mappings" / "disease_id_map.csv",
        mapping_dir / "disease_id_map.csv",
        "disease_id",
        core_diseases | {disease for _, disease in core_hd},
    )

    payload = {
        "source": str(src),
        "output": str(dst),
        "definition": {
            "C_P": "compound in H_C compounds and protein in P_D proteins",
            "H_C": "H_C edges whose compound appears in core C_P",
            "P_D": "P_D edges whose protein appears in core C_P",
            "H_D": "H_D edges whose herb appears in core H_C and disease appears in core P_D",
            "ZERO_indices": f"{args.negative_ratio}:1 sampled negatives, seed={args.seed}",
        },
        "relation_counts": {
            "H_C": len(core_hc),
            "C_P": len(core_cp),
            "P_D": len(core_pd),
            "H_D": len(core_hd),
            "ONE_indices": len(core_cp),
            "ZERO_indices": len(zero_edges),
        },
        "entity_usage": {
            "herbs": len(core_herbs),
            "compounds": len(core_compounds),
            "proteins": len(core_proteins),
            "diseases": len(core_diseases),
            "hd_herbs": len({herb for herb, _ in core_hd}),
            "hd_diseases": len({disease for _, disease in core_hd}),
        },
        "retention_from_full": {
            "H_C_edges": {"core": len(core_hc), "full": len(hc), "percent": pct(len(core_hc), len(hc))},
            "C_P_edges": {"core": len(core_cp), "full": len(cp), "percent": pct(len(core_cp), len(cp))},
            "P_D_edges": {"core": len(core_pd), "full": len(pd), "percent": pct(len(core_pd), len(pd))},
            "H_D_edges": {"core": len(core_hd), "full": len(hd), "percent": pct(len(core_hd), len(hd))},
        },
        "negative_sampling": {
            "seed": args.seed,
            "negative_ratio": args.negative_ratio,
            "candidate_universe": len(core_compounds) * len(core_proteins),
            "positive_edges": len(core_cp),
            "negative_edges": len(zero_edges),
        },
    }

    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "core_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ETCM2.0 core stats",
        "",
        "## Relation counts",
        "",
    ]
    for key, value in payload["relation_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Entity usage", ""])
    for key, value in payload["entity_usage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Retention from full", ""])
    for key, item in payload["retention_from_full"].items():
        lines.append(f"- {key}: {item['core']} / {item['full']} = {item['percent']}%")
    lines.extend(["", "## Negative sampling", ""])
    for key, value in payload["negative_sampling"].items():
        lines.append(f"- {key}: {value}")
    (stats_dir / "core_stats.md").write_text("\n".join(lines), encoding="utf-8")

    if src / "stats" / "entity_stats.json":
        for name in ("entity_stats.json", "entity_stats.md"):
            source = src / "stats" / name
            if source.exists():
                shutil.copy2(source, stats_dir / f"full_{name}")

    print("Done.")
    print(f"Core dataset: {dst}")
    print(
        "Relation counts: "
        f"H_C={len(core_hc)}, C_P={len(core_cp)}, P_D={len(core_pd)}, "
        f"H_D={len(core_hd)}, ZERO={len(zero_edges)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

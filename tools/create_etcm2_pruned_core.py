#!/usr/bin/env python3
"""Create pruned ETCM2.0 core datasets.

The input is expected to be an already connected ETCM2.0 core directory with:

    H_C.txt, C_P.txt, P_D.txt, H_D.txt, ONE_indices.txt, ZERO_indices.txt,
    mappings/*.csv

This script prunes compounds, then rebuilds all relation files and 1:1
negative samples. It keeps original entity ids instead of reindexing them; the
runtime loader maps ids to dense internal indices.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


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


def write_edges(path: Path, edges: Iterable[Edge], label: Optional[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for left, right in sorted(edges):
            if label is None:
                f.write(f"{left}\t{right}\n")
            else:
                f.write(f"{left}\t{right}\t{label}\n")


def load_compound_metadata(path: Path) -> Dict[int, Dict[str, object]]:
    metadata: Dict[int, Dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            compound_id = int(row["compound_id"])
            metadata[compound_id] = {
                "mention_count": int(float(row.get("mention_count") or 0)),
                "standards": row.get("standards") or "",
                "qed": parse_float(row.get("qed")),
                "fdamdd": parse_float(row.get("fdamdd")),
            }
    return metadata


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


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
    compounds = list(compounds)
    proteins = list(proteins)
    negatives: Set[Edge] = set()
    while len(negatives) < count:
        edge = (rng.choice(compounds), rng.choice(proteins))
        if edge in positives or edge in negatives:
            continue
        negatives.add(edge)
    return negatives


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100.0 / total, 4)


def degree_stats(edges: Set[Edge], side: int) -> Dict[str, int]:
    counter = Counter(edge[side] for edge in edges)
    values = sorted(counter.values())
    if not values:
        return {
            "nodes": 0,
            "min": 0,
            "median": 0,
            "p90": 0,
            "p99": 0,
            "max": 0,
            "singletons": 0,
        }

    def percentile(p: float) -> int:
        index = int(round((len(values) - 1) * p))
        return values[index]

    return {
        "nodes": len(values),
        "min": values[0],
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": values[-1],
        "singletons": sum(1 for value in values if value == 1),
    }


def build_keep_compounds(
    compounds: Set[int],
    cp_degree: Counter,
    hc_degree: Counter,
    metadata: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
) -> Set[int]:
    keep = set(compounds)

    if args.min_cp_degree is not None:
        keep = {compound for compound in keep if cp_degree[compound] >= args.min_cp_degree}
    if args.min_hc_degree is not None:
        keep = {compound for compound in keep if hc_degree[compound] >= args.min_hc_degree}
    if args.min_mention_count is not None:
        keep = {
            compound
            for compound in keep
            if int(metadata.get(compound, {}).get("mention_count", 0)) >= args.min_mention_count
        }
    if args.min_qed is not None:
        keep = {
            compound
            for compound in keep
            if metadata.get(compound, {}).get("qed") is not None
            and float(metadata[compound]["qed"]) >= args.min_qed
        }
    if args.standards_only:
        keep = {
            compound
            for compound in keep
            if str(metadata.get(compound, {}).get("standards", "")).lower() == "yes"
        }
    if args.max_compounds is not None and len(keep) > args.max_compounds:
        keep = set(
            sorted(
                keep,
                key=lambda compound: (
                    cp_degree[compound],
                    hc_degree[compound],
                    int(metadata.get(compound, {}).get("mention_count", 0)),
                    -compound,
                ),
                reverse=True,
            )[: args.max_compounds]
        )

    return keep


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/ETCM2.0_core")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-cp-degree", type=int)
    parser.add_argument("--min-hc-degree", type=int)
    parser.add_argument("--min-mention-count", type=int)
    parser.add_argument("--min-qed", type=float)
    parser.add_argument("--standards-only", action="store_true")
    parser.add_argument("--max-compounds", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to update it")

    hc = read_edges(src / "H_C.txt")
    cp = read_edges(src / "C_P.txt")
    pd = read_edges(src / "P_D.txt")
    hd = read_edges(src / "H_D.txt")

    metadata = load_compound_metadata(src / "mappings" / "compound_id_map.csv")
    cp_degree = Counter(compound for compound, _ in cp)
    hc_degree = Counter(compound for _, compound in hc)
    source_compounds = {compound for compound, _ in cp}
    keep_compounds = build_keep_compounds(source_compounds, cp_degree, hc_degree, metadata, args)

    pruned_cp = {(compound, protein) for compound, protein in cp if compound in keep_compounds}
    pruned_compounds = {compound for compound, _ in pruned_cp}
    pruned_proteins = {protein for _, protein in pruned_cp}
    pruned_hc = {(herb, compound) for herb, compound in hc if compound in pruned_compounds}
    pruned_pd = {(protein, disease) for protein, disease in pd if protein in pruned_proteins}
    pruned_herbs = {herb for herb, _ in pruned_hc}
    pruned_diseases = {disease for _, disease in pruned_pd}
    pruned_hd = {
        (herb, disease)
        for herb, disease in hd
        if herb in pruned_herbs and disease in pruned_diseases
    }

    if not pruned_cp:
        raise ValueError("Pruning removed all C_P edges; relax filters")

    zero_count = int(round(len(pruned_cp) * args.negative_ratio))
    zero_edges = sample_negative_edges(
        sorted(pruned_compounds),
        sorted(pruned_proteins),
        pruned_cp,
        zero_count,
        args.seed,
    )

    dst.mkdir(parents=True, exist_ok=True)
    write_edges(dst / "H_C.txt", pruned_hc, label=1)
    write_edges(dst / "C_P.txt", pruned_cp, label=None)
    write_edges(dst / "P_D.txt", pruned_pd, label=1)
    write_edges(dst / "H_D.txt", pruned_hd, label=1)
    write_edges(dst / "ONE_indices.txt", pruned_cp, label=1)
    write_edges(dst / "ZERO_indices.txt", zero_edges, label=0)

    mapping_dir = dst / "mappings"
    filter_mapping_csv(
        src / "mappings" / "herb_id_map.csv",
        mapping_dir / "herb_id_map.csv",
        "herb_id",
        pruned_herbs | {herb for herb, _ in pruned_hd},
    )
    filter_mapping_csv(
        src / "mappings" / "compound_id_map.csv",
        mapping_dir / "compound_id_map.csv",
        "compound_id",
        pruned_compounds,
    )
    filter_mapping_csv(
        src / "mappings" / "protein_id_map.csv",
        mapping_dir / "protein_id_map.csv",
        "protein_id",
        pruned_proteins,
    )
    filter_mapping_csv(
        src / "mappings" / "disease_id_map.csv",
        mapping_dir / "disease_id_map.csv",
        "disease_id",
        pruned_diseases | {disease for _, disease in pruned_hd},
    )

    stats = {
        "source": str(src),
        "output": str(dst),
        "filters": {
            "min_cp_degree": args.min_cp_degree,
            "min_hc_degree": args.min_hc_degree,
            "min_mention_count": args.min_mention_count,
            "min_qed": args.min_qed,
            "standards_only": args.standards_only,
            "max_compounds": args.max_compounds,
        },
        "relation_counts": {
            "H_C": len(pruned_hc),
            "C_P": len(pruned_cp),
            "P_D": len(pruned_pd),
            "H_D": len(pruned_hd),
            "ONE_indices": len(pruned_cp),
            "ZERO_indices": len(zero_edges),
        },
        "entity_usage": {
            "herbs": len(pruned_herbs),
            "compounds": len(pruned_compounds),
            "proteins": len(pruned_proteins),
            "diseases": len(pruned_diseases),
            "hd_herbs": len({herb for herb, _ in pruned_hd}),
            "hd_diseases": len({disease for _, disease in pruned_hd}),
        },
        "retention_from_input": {
            "H_C_edges": {"pruned": len(pruned_hc), "input": len(hc), "percent": pct(len(pruned_hc), len(hc))},
            "C_P_edges": {"pruned": len(pruned_cp), "input": len(cp), "percent": pct(len(pruned_cp), len(cp))},
            "P_D_edges": {"pruned": len(pruned_pd), "input": len(pd), "percent": pct(len(pruned_pd), len(pd))},
            "H_D_edges": {"pruned": len(pruned_hd), "input": len(hd), "percent": pct(len(pruned_hd), len(hd))},
            "compounds": {"pruned": len(pruned_compounds), "input": len(source_compounds), "percent": pct(len(pruned_compounds), len(source_compounds))},
        },
        "negative_sampling": {
            "seed": args.seed,
            "negative_ratio": args.negative_ratio,
            "candidate_universe": len(pruned_compounds) * len(pruned_proteins),
            "positive_edges": len(pruned_cp),
            "negative_edges": len(zero_edges),
            "positive_negative_overlap": len(pruned_cp & zero_edges),
        },
        "degree_stats": {
            "H_C_left": degree_stats(pruned_hc, 0),
            "H_C_right": degree_stats(pruned_hc, 1),
            "C_P_left": degree_stats(pruned_cp, 0),
            "C_P_right": degree_stats(pruned_cp, 1),
            "P_D_left": degree_stats(pruned_pd, 0),
            "P_D_right": degree_stats(pruned_pd, 1),
            "H_D_left": degree_stats(pruned_hd, 0),
            "H_D_right": degree_stats(pruned_hd, 1),
        },
    }

    stats_dir = dst / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "pruned_core_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ETCM2.0 pruned core stats",
        "",
        "## Filters",
        "",
    ]
    for key, value in stats["filters"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Relation counts", ""])
    for key, value in stats["relation_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Entity usage", ""])
    for key, value in stats["entity_usage"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Retention from input", ""])
    for key, item in stats["retention_from_input"].items():
        lines.append(f"- {key}: {item['pruned']} / {item['input']} = {item['percent']}%")
    lines.extend(["", "## Negative sampling", ""])
    for key, value in stats["negative_sampling"].items():
        lines.append(f"- {key}: {value}")
    (stats_dir / "pruned_core_stats.md").write_text("\n".join(lines), encoding="utf-8")

    source_stats = src / "stats"
    if source_stats.exists():
        for path in source_stats.iterdir():
            if path.is_file() and path.name.startswith("full_"):
                shutil.copy2(path, stats_dir / path.name)

    print(f"Done: {dst}")
    print(
        "Relation counts: "
        f"H_C={len(pruned_hc)}, C_P={len(pruned_cp)}, P_D={len(pruned_pd)}, "
        f"H_D={len(pruned_hd)}, ZERO={len(zero_edges)}"
    )
    print(
        "Entity counts: "
        f"herbs={len(pruned_herbs)}, compounds={len(pruned_compounds)}, "
        f"proteins={len(pruned_proteins)}, diseases={len(pruned_diseases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

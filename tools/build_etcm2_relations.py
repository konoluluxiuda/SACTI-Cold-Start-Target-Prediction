#!/usr/bin/env python3
"""Build positive ETCM2.0 relation edge tables and audit statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from build_etcm2_entity_mappings import (
    basic_info_map,
    clean_text,
    first_value,
    load_relevant_sections,
    protein_key,
    related_blocks,
)


Edge = Tuple[int, int]
Edge3 = Tuple[int, int, int]


class EdgeStore:
    def __init__(self, name: str):
        self.name = name
        self.edges: Set[Edge] = set()
        self.source_edges: Dict[str, Set[Edge]] = defaultdict(set)
        self.raw_rows = Counter()
        self.skipped_rows = Counter()
        self.skipped_examples: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def add(self, left_id: Optional[int], right_id: Optional[int], source: str, context: Dict[str, str]) -> None:
        self.raw_rows[source] += 1
        missing = []
        if left_id is None:
            missing.append("left")
        if right_id is None:
            missing.append("right")
        if missing:
            reason = f"{source}:missing_{'_'.join(missing)}"
            self.skipped_rows[reason] += 1
            if len(self.skipped_examples[reason]) < 20:
                self.skipped_examples[reason].append(context)
            return
        edge = (left_id, right_id)
        self.edges.add(edge)
        self.source_edges[source].add(edge)

    def write(self, path: Path, with_label: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for left, right in sorted(self.edges):
                if with_label:
                    f.write(f"{left}\t{right}\t1\n")
                else:
                    f.write(f"{left}\t{right}\n")

    def source_counts(self) -> Dict[str, int]:
        return {source: len(edges) for source, edges in sorted(self.source_edges.items())}

    def left_ids(self) -> Set[int]:
        return {left for left, _ in self.edges}

    def right_ids(self) -> Set[int]:
        return {right for _, right in self.edges}


def load_id_map(path: Path, id_col: str, key_col: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    key_to_id: Dict[str, int] = {}
    id_to_key: Dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = clean_text(row[key_col])
            idx = int(row[id_col])
            key_to_id[key] = idx
            id_to_key[idx] = key
    return key_to_id, id_to_key


def get_id(mapping: Dict[str, int], key: str) -> Optional[int]:
    key = clean_text(key)
    if not key:
        return None
    return mapping.get(key)


def relation_context(path: Path, **kwargs: str) -> Dict[str, str]:
    out = {"file": str(path)}
    out.update({key: clean_text(value) for key, value in kwargs.items()})
    return out


def process_herb_file(
    path: Path,
    maps: Dict[str, Dict[str, int]],
    relations: Dict[str, EdgeStore],
    stats: Counter,
    include_herb_enriched_diseases: bool,
) -> None:
    sections = load_relevant_sections(path)
    stats["herb_files_parsed"] += 1
    info = basic_info_map(sections.get("base_information"))
    herb_key = first_value(info, "Herb Name in Pinyin") or clean_text(path.stem)
    herb_id = get_id(maps["herbs"], herb_key)

    for block_type, rows in related_blocks(sections.get("related_table")):
        if block_type == "ingredient":
            for row in rows:
                compound_key = first_value(row, "Ingredient Name")
                compound_id = get_id(maps["compounds"], compound_key)
                relations["H_C"].add(
                    herb_id,
                    compound_id,
                    "herb_related_ingredient",
                    relation_context(path, herb=herb_key, compound=compound_key),
                )
        elif block_type == "disease":
            for row in rows:
                disease_key = first_value(row, "Disease Name")
                disease_id = get_id(maps["diseases"], disease_key)
                relations["H_D_enriched_audit"].add(
                    herb_id,
                    disease_id,
                    "herb_related_disease_enrichment",
                    relation_context(path, herb=herb_key, disease=disease_key),
                )
                if include_herb_enriched_diseases:
                    relations["H_D"].add(
                        herb_id,
                        disease_id,
                        "herb_related_disease_enrichment",
                        relation_context(path, herb=herb_key, disease=disease_key),
                    )

    component_target = sections.get("ingredient_target")
    if isinstance(component_target, dict):
        for row in component_target.get("data") or []:
            if not isinstance(row, dict):
                continue
            compound_key = first_value(row, "Ingredient Name")
            p_key = protein_key(row)
            compound_id = get_id(maps["compounds"], compound_key)
            protein_id = get_id(maps["proteins"], p_key)
            relations["C_P"].add(
                compound_id,
                protein_id,
                "herb_component_target",
                relation_context(path, compound=compound_key, protein=p_key),
            )

    network = sections.get("basic_network")
    if isinstance(network, dict):
        for node in network.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = clean_text(node.get("id"))
            if not node_id.startswith("TCMIP-I-"):
                continue
            compound_key = clean_text(node.get("label") or node.get("title"))
            compound_id = get_id(maps["compounds"], compound_key)
            relations["H_C"].add(
                herb_id,
                compound_id,
                "herb_network_ingredient",
                relation_context(path, herb=herb_key, compound=compound_key),
            )


def process_disease_file(
    path: Path,
    maps: Dict[str, Dict[str, int]],
    relations: Dict[str, EdgeStore],
    stats: Counter,
) -> None:
    sections = load_relevant_sections(path)
    stats["disease_files_parsed"] += 1
    info = basic_info_map(sections.get("base_information"))
    disease_key = first_value(info, "Disease Name") or clean_text(path.stem.replace("_", " "))
    disease_id = get_id(maps["diseases"], disease_key)

    for block_type, rows in related_blocks(sections.get("related_table")):
        if block_type == "herb":
            for row in rows:
                herb_key = first_value(row, "Herb Name in Pinyin")
                herb_id = get_id(maps["herbs"], herb_key)
                relations["H_D"].add(
                    herb_id,
                    disease_id,
                    "disease_related_herb",
                    relation_context(path, herb=herb_key, disease=disease_key),
                )
        elif block_type == "target":
            for row in rows:
                p_key = protein_key(row)
                protein_id = get_id(maps["proteins"], p_key)
                relations["P_D"].add(
                    protein_id,
                    disease_id,
                    "disease_related_target",
                    relation_context(path, protein=p_key, disease=disease_key),
                )


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part * 100.0 / total, 4)


def sample_ids(ids: Iterable[int], id_to_key: Dict[int, str], limit: int = 50) -> List[Dict[str, object]]:
    rows = []
    for idx in sorted(ids)[:limit]:
        rows.append({"id": idx, "key": id_to_key.get(idx, "")})
    return rows


def build_stats(
    maps_reverse: Dict[str, Dict[int, str]],
    relations: Dict[str, EdgeStore],
    stats: Counter,
    errors: List[Tuple[str, str]],
    include_herb_enriched_diseases: bool,
) -> Dict[str, object]:
    hc = relations["H_C"]
    cp = relations["C_P"]
    pd = relations["P_D"]
    hd = relations["H_D"]
    hd_enriched = relations["H_D_enriched_audit"]

    cp_compounds = cp.left_ids()
    cp_proteins = cp.right_ids()
    hc_herbs = hc.left_ids()
    hc_compounds = hc.right_ids()
    pd_proteins = pd.left_ids()
    pd_diseases = pd.right_ids()
    hd_herbs = hd.left_ids()
    hd_diseases = hd.right_ids()

    cp_compounds_with_hc = cp_compounds & hc_compounds
    cp_proteins_with_pd = cp_proteins & pd_proteins
    pd_diseases_with_hd = pd_diseases & hd_diseases
    hd_herbs_with_hc = hd_herbs & hc_herbs
    connected_cp_edges = {
        edge for edge in cp.edges if edge[0] in hc_compounds and edge[1] in pd_proteins
    }

    relation_counts = {
        "H_C": len(hc.edges),
        "C_P": len(cp.edges),
        "P_D": len(pd.edges),
        "H_D": len(hd.edges),
        "H_D_enriched_audit_not_included_by_default": len(hd_enriched.edges),
        "ONE_indices": len(cp.edges),
    }
    entity_usage = {
        "H_C": {"herbs": len(hc_herbs), "compounds": len(hc_compounds)},
        "C_P": {"compounds": len(cp_compounds), "proteins": len(cp_proteins)},
        "P_D": {"proteins": len(pd_proteins), "diseases": len(pd_diseases)},
        "H_D": {"herbs": len(hd_herbs), "diseases": len(hd_diseases)},
    }
    intersections = {
        "cp_compounds_with_hc_support": {
            "count": len(cp_compounds_with_hc),
            "total_cp_compounds": len(cp_compounds),
            "coverage_percent": pct(len(cp_compounds_with_hc), len(cp_compounds)),
            "missing_count": len(cp_compounds - hc_compounds),
            "missing_examples": sample_ids(cp_compounds - hc_compounds, maps_reverse["compounds"]),
        },
        "cp_proteins_with_pd_support": {
            "count": len(cp_proteins_with_pd),
            "total_cp_proteins": len(cp_proteins),
            "coverage_percent": pct(len(cp_proteins_with_pd), len(cp_proteins)),
            "missing_count": len(cp_proteins - pd_proteins),
            "missing_examples": sample_ids(cp_proteins - pd_proteins, maps_reverse["proteins"]),
        },
        "pd_proteins_with_cp_support": {
            "count": len(cp_proteins_with_pd),
            "total_pd_proteins": len(pd_proteins),
            "coverage_percent": pct(len(cp_proteins_with_pd), len(pd_proteins)),
            "missing_count": len(pd_proteins - cp_proteins),
            "missing_examples": sample_ids(pd_proteins - cp_proteins, maps_reverse["proteins"]),
        },
        "pd_diseases_with_hd_support": {
            "count": len(pd_diseases_with_hd),
            "total_pd_diseases": len(pd_diseases),
            "coverage_percent": pct(len(pd_diseases_with_hd), len(pd_diseases)),
            "missing_count": len(pd_diseases - hd_diseases),
            "missing_examples": sample_ids(pd_diseases - hd_diseases, maps_reverse["diseases"]),
        },
        "hd_herbs_with_hc_support": {
            "count": len(hd_herbs_with_hc),
            "total_hd_herbs": len(hd_herbs),
            "coverage_percent": pct(len(hd_herbs_with_hc), len(hd_herbs)),
            "missing_count": len(hd_herbs - hc_herbs),
            "missing_examples": sample_ids(hd_herbs - hc_herbs, maps_reverse["herbs"]),
        },
        "connected_cp_edges_with_hc_and_pd_support": {
            "count": len(connected_cp_edges),
            "total_cp_edges": len(cp.edges),
            "coverage_percent": pct(len(connected_cp_edges), len(cp.edges)),
        },
    }

    return {
        "relation_counts": relation_counts,
        "entity_usage": entity_usage,
        "source_counts": {name: store.source_counts() for name, store in relations.items()},
        "raw_rows": {name: dict(store.raw_rows) for name, store in relations.items()},
        "skipped_rows": {name: dict(store.skipped_rows) for name, store in relations.items()},
        "skipped_examples": {name: dict(store.skipped_examples) for name, store in relations.items()},
        "intersections": intersections,
        "parse_stats": dict(stats),
        "include_herb_enriched_diseases": include_herb_enriched_diseases,
        "error_count": len(errors),
        "errors": [{"file": file, "error": error} for file, error in errors[:200]],
    }


def write_stats_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = ["# ETCM2.0 relation stats", ""]
    lines.append("## Relation counts")
    lines.append("")
    for key, value in payload["relation_counts"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Entity usage", ""])
    for rel, counts in payload["entity_usage"].items():
        parts = ", ".join(f"{key}: {value}" for key, value in counts.items())
        lines.append(f"- {rel}: {parts}")

    lines.extend(["", "## Intersection review", ""])
    for key, item in payload["intersections"].items():
        lines.append(f"### {key}")
        for field in ("count", "total_cp_compounds", "total_cp_proteins", "total_pd_proteins", "total_pd_diseases", "total_hd_herbs", "total_cp_edges", "coverage_percent", "missing_count"):
            if field in item:
                lines.append(f"- {field}: {item[field]}")
        if item.get("missing_examples"):
            lines.append("- missing_examples:")
            for example in item["missing_examples"][:10]:
                lines.append(f"  - {example['id']}: {example['key']}")
        lines.append("")

    lines.append("## Source counts")
    lines.append("")
    for rel, counts in payload["source_counts"].items():
        lines.append(f"### {rel}")
        for source, count in counts.items():
            lines.append(f"- {source}: {count}")
        lines.append("")

    lines.append("## Skipped rows")
    lines.append("")
    for rel, counts in payload["skipped_rows"].items():
        lines.append(f"### {rel}")
        if counts:
            for reason, count in counts.items():
                lines.append(f"- {reason}: {count}")
        else:
            lines.append("- none")
        lines.append("")

    lines.append("## Parse stats")
    lines.append("")
    for key, value in sorted(payload["parse_stats"].items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(f"Errors: {payload['error_count']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def iter_files(root: Path, subdir: str, max_files: Optional[int]) -> List[Path]:
    files = sorted((root / subdir).glob("*.json"))
    if max_files is not None:
        return files[:max_files]
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/ETCM2.0", help="ETCM2.0 raw JSON root")
    parser.add_argument("--output", default="dataset/ETCM2.0_processed", help="Processed output root")
    parser.add_argument(
        "--mapping-dir",
        default=None,
        help="Directory containing *_id_map.csv files. Defaults to OUTPUT/mappings.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug limit per raw subdir")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--include-herb-enriched-diseases",
        action="store_true",
        help="Also include herb-page Enriched Diseases in H_D.txt",
    )
    args = parser.parse_args()

    raw_root = Path(args.input)
    out_root = Path(args.output)
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else out_root / "mappings"
    stats_dir = out_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    herbs, herbs_rev = load_id_map(mapping_dir / "herb_id_map.csv", "herb_id", "herb_key")
    compounds, compounds_rev = load_id_map(mapping_dir / "compound_id_map.csv", "compound_id", "compound_key")
    proteins, proteins_rev = load_id_map(mapping_dir / "protein_id_map.csv", "protein_id", "protein_key")
    diseases, diseases_rev = load_id_map(mapping_dir / "disease_id_map.csv", "disease_id", "disease_key")
    maps = {"herbs": herbs, "compounds": compounds, "proteins": proteins, "diseases": diseases}
    maps_reverse = {
        "herbs": herbs_rev,
        "compounds": compounds_rev,
        "proteins": proteins_rev,
        "diseases": diseases_rev,
    }

    relations = {
        "H_C": EdgeStore("H_C"),
        "C_P": EdgeStore("C_P"),
        "P_D": EdgeStore("P_D"),
        "H_D": EdgeStore("H_D"),
        "H_D_enriched_audit": EdgeStore("H_D_enriched_audit"),
    }
    stats: Counter = Counter()
    errors: List[Tuple[str, str]] = []

    herb_files = iter_files(raw_root, "etcm_herbs", args.max_files)
    for index, path in enumerate(herb_files, start=1):
        try:
            process_herb_file(path, maps, relations, stats, args.include_herb_enriched_diseases)
        except Exception as exc:
            errors.append((str(path), repr(exc)))
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"etcm_herbs: {index}/{len(herb_files)} files | "
                f"H_C={len(relations['H_C'].edges)} C_P={len(relations['C_P'].edges)} "
                f"H_D_audit={len(relations['H_D_enriched_audit'].edges)}",
                flush=True,
            )

    disease_files = iter_files(raw_root, "etcm_diseases", args.max_files)
    for index, path in enumerate(disease_files, start=1):
        try:
            process_disease_file(path, maps, relations, stats)
        except Exception as exc:
            errors.append((str(path), repr(exc)))
        if args.progress_every and index % args.progress_every == 0:
            print(
                f"etcm_diseases: {index}/{len(disease_files)} files | "
                f"P_D={len(relations['P_D'].edges)} H_D={len(relations['H_D'].edges)}",
                flush=True,
            )

    relations["H_C"].write(out_root / "H_C.txt", with_label=True)
    relations["C_P"].write(out_root / "C_P.txt", with_label=False)
    relations["P_D"].write(out_root / "P_D.txt", with_label=True)
    relations["H_D"].write(out_root / "H_D.txt", with_label=True)
    relations["C_P"].write(out_root / "ONE_indices.txt", with_label=True)

    payload = build_stats(
        maps_reverse,
        relations,
        stats,
        errors,
        args.include_herb_enriched_diseases,
    )
    (stats_dir / "relation_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_stats_markdown(stats_dir / "relation_stats.md", payload)

    print("Done.")
    print(f"Relations: {out_root}")
    print(f"Stats: {stats_dir}")
    print(
        "Relation counts: "
        f"H_C={len(relations['H_C'].edges)}, "
        f"C_P={len(relations['C_P'].edges)}, "
        f"P_D={len(relations['P_D'].edges)}, "
        f"H_D={len(relations['H_D'].edges)}"
    )
    if errors:
        print(f"Errors: {len(errors)}; see stats/relation_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

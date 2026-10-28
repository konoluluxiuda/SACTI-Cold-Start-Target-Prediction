#!/usr/bin/env python3
"""Build entity mapping tables from ETCM2.0 JSON pages.

This script only creates entity maps and audit statistics. It does not build
relation edge files yet.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SECTION_IDS = {"base_information", "related_table", "ingredient_target", "basic_network"}
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "; ".join(clean_text(item) for item in value if clean_text(item))
    value = html.unescape(str(value))
    value = TAG_RE.sub("", value)
    value = SPACE_RE.sub(" ", value).strip()
    if value.upper() == "NULL":
        return ""
    return value


def as_values(value) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def first_value(row: Dict, *keys: str) -> str:
    for key in keys:
        values = as_values(row.get(key))
        if values:
            return values[0]
    return ""


def all_values(row: Dict, *keys: str) -> List[str]:
    values: List[str] = []
    for key in keys:
        values.extend(as_values(row.get(key)))
    return values


def stable_join(values: Iterable[str]) -> str:
    return "|".join(sorted({clean_text(v) for v in values if clean_text(v)}))


def iter_top_level_sections(text: str) -> Iterator[str]:
    """Yield raw JSON strings for objects in the top-level data array."""
    data_pos = text.find('"data"')
    if data_pos < 0:
        return
    start = text.find("[", data_pos)
    if start < 0:
        return

    i = start + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \r\n\t,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            i += 1
            continue

        obj_start = i
        depth = 0
        in_string = False
        escape = False
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        yield text[obj_start:i]
                        break
            i += 1


def load_relevant_sections(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections: Dict[str, object] = {}
    for section_text in iter_top_level_sections(text):
        if not any(f'"id": "{section_id}"' in section_text for section_id in SECTION_IDS):
            continue
        section = json.loads(section_text)
        section_id = section.get("id")
        if section_id in SECTION_IDS:
            sections[section_id] = section.get("value")
    return sections


class EntityStore:
    def __init__(self, key_name: str, fields: Sequence[str]):
        self.key_name = key_name
        self.fields = list(fields)
        self.rows: Dict[str, Dict[str, object]] = {}

    def add(self, key: str, source: str, **fields) -> None:
        key = clean_text(key)
        if not key:
            return
        row = self.rows.setdefault(
            key,
            {
                self.key_name: key,
                "source_types": set(),
                "mention_count": 0,
                **{field: set() for field in self.fields},
            },
        )
        row["source_types"].add(source)
        row["mention_count"] += 1
        for field, value in fields.items():
            if field not in row:
                continue
            for item in as_values(value):
                row[field].add(item)

    def __len__(self) -> int:
        return len(self.rows)

    def source_counter(self) -> Counter:
        counter = Counter()
        for row in self.rows.values():
            for source in row["source_types"]:
                counter[source] += 1
        return counter

    def write_csv(self, path: Path, id_column: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [id_column, self.key_name] + self.fields + ["source_types", "mention_count"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for idx, key in enumerate(sorted(self.rows), start=1):
                row = self.rows[key]
                out = {id_column: idx, self.key_name: row[self.key_name]}
                for field in self.fields:
                    out[field] = stable_join(row[field])
                out["source_types"] = stable_join(row["source_types"])
                out["mention_count"] = row["mention_count"]
                writer.writerow(out)


def basic_info_map(value) -> Dict[str, object]:
    info: Dict[str, object] = {}
    if not isinstance(value, list):
        return info
    for item in value:
        if not isinstance(item, dict):
            continue
        key = clean_text(item.get("key"))
        if key:
            info[key] = item.get("value")
    return info


def related_blocks(value) -> Iterator[Tuple[str, List[Dict]]]:
    if not isinstance(value, list):
        return
    for block in value:
        if not isinstance(block, dict):
            continue
        block_type = clean_text(block.get("type") or block.get("label")).lower()
        rows = block.get("value") or []
        if isinstance(rows, list):
            yield block_type, [row for row in rows if isinstance(row, dict)]


def protein_key(row: Dict) -> str:
    gene = first_value(row, "Gene Symbol")
    if gene:
        return gene
    uniprot = first_value(row, "UniProt Accession Number")
    if uniprot:
        return f"UNIPROT:{uniprot}"
    target = first_value(row, "Target Name")
    if target:
        return f"TARGET:{target}"
    return ""


def add_protein(proteins: EntityStore, key: str, source: str, row: Dict) -> None:
    proteins.add(
        key,
        source,
        gene_symbols=all_values(row, "Gene Symbol"),
        uniprot_accessions=all_values(row, "UniProt Accession Number"),
        target_names=all_values(row, "Target Name"),
        proteins=all_values(row, "Protein"),
        organisms=all_values(row, "Organism"),
    )


def add_herb(herbs: EntityStore, key: str, source: str, row: Dict) -> None:
    herbs.add(
        key,
        source,
        herb_name_pinyin=all_values(row, "Herb Name in Pinyin"),
        herb_name_latin=all_values(row, "Herb Name in Latin"),
        herb_name_english=all_values(row, "Herb Name in English"),
        property=all_values(row, "Property"),
        flavor=all_values(row, "Flavor"),
        meridian_tropism=all_values(row, "Meridian Tropism"),
    )


def add_compound(compounds: EntityStore, key: str, source: str, row: Dict) -> None:
    compounds.add(
        key,
        source,
        ingredient_names=all_values(row, "Ingredient Name") or [key],
        tcmip_ids=all_values(row, "TCMIP ID"),
        molecular_formula=all_values(row, "Molecular Formula"),
        molecular_weight=all_values(row, "Molecular Weight"),
        qed=all_values(row, "Quantitative Estimate of Drug-likeness（QED）"),
        fdamdd=all_values(row, "FDA Maximum Daily Dose (FDAMDD)"),
        standards=all_values(row, "Standards"),
    )


def add_disease(diseases: EntityStore, key: str, source: str, row: Dict) -> None:
    diseases.add(
        key,
        source,
        disease_names=all_values(row, "Disease Name") or [key],
        global_categories=all_values(row, "Global Category", "Global category"),
        anatomical_categories=all_values(row, "Anatomical Category", "Anatomical category"),
    )


def process_herb_file(path: Path, stores: Dict[str, EntityStore], stats: Counter) -> None:
    sections = load_relevant_sections(path)
    stats["herb_files_parsed"] += 1
    info = basic_info_map(sections.get("base_information"))
    herb_key = first_value(info, "Herb Name in Pinyin") or clean_text(path.stem)
    add_herb(stores["herbs"], herb_key, "herb_page_base", info)

    for block_type, rows in related_blocks(sections.get("related_table")):
        if block_type == "ingredient":
            for row in rows:
                compound = first_value(row, "Ingredient Name")
                add_compound(stores["compounds"], compound, "herb_related_ingredient", row)
                stats["herb_related_ingredient_rows"] += 1
        elif block_type == "target":
            for row in rows:
                key = protein_key(row)
                add_protein(stores["proteins"], key, "herb_related_target", row)
                stats["herb_related_target_rows"] += 1
        elif block_type == "disease":
            for row in rows:
                disease = first_value(row, "Disease Name")
                add_disease(stores["diseases"], disease, "herb_related_disease", row)
                stats["herb_related_disease_rows"] += 1

    component_target = sections.get("ingredient_target")
    if isinstance(component_target, dict):
        for row in component_target.get("data") or []:
            if not isinstance(row, dict):
                continue
            compound = first_value(row, "Ingredient Name")
            add_compound(stores["compounds"], compound, "herb_component_target", row)
            key = protein_key(row)
            add_protein(stores["proteins"], key, "herb_component_target", row)
            stats["herb_component_target_rows"] += 1

    network = sections.get("basic_network")
    if isinstance(network, dict):
        for node in network.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = clean_text(node.get("id"))
            if not node_id.startswith("TCMIP-I-"):
                continue
            label = clean_text(node.get("label") or node.get("title"))
            add_compound(
                stores["compounds"],
                label,
                "herb_network_ingredient",
                {"Ingredient Name": [label], "TCMIP ID": [node_id]},
            )
            stats["herb_network_ingredient_nodes"] += 1


def process_disease_file(path: Path, stores: Dict[str, EntityStore], stats: Counter) -> None:
    sections = load_relevant_sections(path)
    stats["disease_files_parsed"] += 1
    info = basic_info_map(sections.get("base_information"))
    disease_key = first_value(info, "Disease Name") or clean_text(path.stem.replace("_", " "))
    add_disease(stores["diseases"], disease_key, "disease_page_base", info)

    for block_type, rows in related_blocks(sections.get("related_table")):
        if block_type == "herb":
            for row in rows:
                herb = first_value(row, "Herb Name in Pinyin")
                add_herb(stores["herbs"], herb, "disease_related_herb", row)
                stats["disease_related_herb_rows"] += 1
        elif block_type == "target":
            for row in rows:
                key = protein_key(row)
                add_protein(stores["proteins"], key, "disease_related_target", row)
                stats["disease_related_target_rows"] += 1


def process_target_file(path: Path, stores: Dict[str, EntityStore], stats: Counter) -> None:
    sections = load_relevant_sections(path)
    stats["target_files_parsed"] += 1
    info = basic_info_map(sections.get("base_information"))
    key = protein_key(info) or clean_text(path.stem)
    add_protein(stores["proteins"], key, "target_page_base", info)


def iter_files(root: Path, subdir: str, max_files: Optional[int]) -> List[Path]:
    files = sorted((root / subdir).glob("*.json"))
    if max_files is not None:
        files = files[:max_files]
    return files


def write_stats(
    output_dir: Path,
    stores: Dict[str, EntityStore],
    stats: Counter,
    errors: List[Tuple[str, str]],
) -> None:
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "entity_counts": {
            "herbs": len(stores["herbs"]),
            "compounds": len(stores["compounds"]),
            "proteins": len(stores["proteins"]),
            "diseases": len(stores["diseases"]),
        },
        "source_counts": {
            name: dict(store.source_counter()) for name, store in stores.items()
        },
        "parse_stats": dict(stats),
        "error_count": len(errors),
        "errors": [{"file": file, "error": error} for file, error in errors[:200]],
    }
    (stats_dir / "entity_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# ETCM2.0 entity mapping stats",
        "",
        "## Entity counts",
        "",
    ]
    for name, count in payload["entity_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Source counts", ""])
    for name, counter in payload["source_counts"].items():
        lines.append(f"### {name}")
        for source, count in sorted(counter.items()):
            lines.append(f"- {source}: {count}")
        lines.append("")
    lines.extend(["## Parse stats", ""])
    for key, value in sorted(payload["parse_stats"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", f"Errors: {len(errors)}", ""])
    if errors:
        lines.append("First errors:")
        for file, error in errors[:20]:
            lines.append(f"- {file}: {error}")
    (stats_dir / "entity_stats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/ETCM2.0", help="ETCM2.0 root directory")
    parser.add_argument(
        "--output",
        default="dataset/ETCM2.0_processed",
        help="Output directory for mapping tables and stats",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug limit per subdir")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    root = Path(args.input)
    output_dir = Path(args.output)
    mapping_dir = output_dir / "mappings"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    stores = {
        "herbs": EntityStore(
            "herb_key",
            [
                "herb_name_pinyin",
                "herb_name_latin",
                "herb_name_english",
                "property",
                "flavor",
                "meridian_tropism",
            ],
        ),
        "compounds": EntityStore(
            "compound_key",
            [
                "ingredient_names",
                "tcmip_ids",
                "molecular_formula",
                "molecular_weight",
                "qed",
                "fdamdd",
                "standards",
            ],
        ),
        "proteins": EntityStore(
            "protein_key",
            ["gene_symbols", "uniprot_accessions", "target_names", "proteins", "organisms"],
        ),
        "diseases": EntityStore(
            "disease_key",
            ["disease_names", "global_categories", "anatomical_categories"],
        ),
    }
    stats: Counter = Counter()
    errors: List[Tuple[str, str]] = []

    jobs = [
        ("etcm_herbs", process_herb_file),
        ("etcm_diseases", process_disease_file),
        ("etcm_targets", process_target_file),
    ]

    for subdir, processor in jobs:
        files = iter_files(root, subdir, args.max_files)
        for index, path in enumerate(files, start=1):
            try:
                processor(path, stores, stats)
            except Exception as exc:  # keep long batch jobs reviewable
                errors.append((str(path), repr(exc)))
            if args.progress_every and index % args.progress_every == 0:
                print(
                    f"{subdir}: {index}/{len(files)} files | "
                    f"H={len(stores['herbs'])} C={len(stores['compounds'])} "
                    f"P={len(stores['proteins'])} D={len(stores['diseases'])}",
                    flush=True,
                )

    stores["herbs"].write_csv(mapping_dir / "herb_id_map.csv", "herb_id")
    stores["compounds"].write_csv(mapping_dir / "compound_id_map.csv", "compound_id")
    stores["proteins"].write_csv(mapping_dir / "protein_id_map.csv", "protein_id")
    stores["diseases"].write_csv(mapping_dir / "disease_id_map.csv", "disease_id")
    write_stats(output_dir, stores, stats, errors)

    print("Done.")
    print(f"Mappings: {mapping_dir}")
    print(f"Stats: {output_dir / 'stats'}")
    print(
        f"Entity counts: H={len(stores['herbs'])}, C={len(stores['compounds'])}, "
        f"P={len(stores['proteins'])}, D={len(stores['diseases'])}"
    )
    if errors:
        print(f"Errors: {len(errors)}; see stats/entity_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

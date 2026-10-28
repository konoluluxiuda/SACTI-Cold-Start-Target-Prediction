#!/usr/bin/env python3
"""Build a compact ETCM2.0 ingredient evidence store for external validation.

The generated files are deliberately separate from model training data.
Confirmed targets, potential targets, and herb context are never written to
the active H_C/C_P/P_D/H_D files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "dataset" / "ETCM2.0" / "etcm_ingredients"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "dataset" / "ETCM2.0_validation"
DEFAULT_DATASETS = (
    REPOSITORY_ROOT / "dataset" / "ETCM2.0_processed",
    REPOSITORY_ROOT / "dataset" / "ETCM2.0_core",
    REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_mention10",
    REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_cpdeg3",
    REPOSITORY_ROOT / "dataset" / "ETCM2.0_core_cpdeg5",
)
WANTED_RELATIONS = ("herb", "target", "similar_target")
SUMMARY_ENTRY_RE = re.compile(
    rb'"type"\s*:\s*"([^"]+)"\s*,\s*'
    rb'"expected"\s*:\s*(\d+)\s*,\s*'
    rb'"actual"\s*:\s*(\d+)\s*,\s*'
    rb'"status"\s*:\s*"([^"]+)"'
)
TCMIP_RE = re.compile(r"TCMIP-I-\d+")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


INGREDIENT_FIELDS = (
    "tcmip_id",
    "compound_id_processed",
    "compound_name",
    "molecular_formula",
    "molecular_weight",
    "cas_number",
    "pubchem_cids",
    "structure_url",
    "qed",
    "synthetic_accessibility",
    "natural_product_likeness",
    "lipinski_rule",
    "gsk_rule",
    "golden_triangle",
    "fdamdd",
    "herb_count",
    "confirmed_target_count",
    "potential_target_count",
    "disease_count",
    "formula_count",
    "patent_drug_count",
    "mapping_status",
    "source_file",
)
ALIAS_FIELDS = ("tcmip_id", "compound_name", "alias", "source_file")
HERB_FIELDS = (
    "tcmip_id",
    "compound_id_processed",
    "compound_name",
    "herb_name_pinyin",
    "herb_name_latin",
    "property",
    "flavor",
    "meridian_tropism",
    "source_file",
)
EVIDENCE_FIELDS = (
    "evidence_id",
    "tcmip_id",
    "compound_id_processed",
    "compound_name",
    "gene_symbol",
    "protein_id_processed",
    "activity",
    "similar_score",
    "reference_labels",
    "reference_urls",
    "source_file",
)
PAIR_FIELDS = (
    "tcmip_id",
    "compound_id_processed",
    "compound_name",
    "gene_symbol",
    "protein_id_processed",
    "evidence_count",
    "activity_count",
    "reference_count",
    "similar_score_min",
    "similar_score_mean",
    "similar_score_max",
    "mapping_status",
)
VALIDATION_FIELDS = (
    "tcmip_id",
    "compound_id",
    "compound_name",
    "gene_symbol",
    "protein_id",
    "status",
    "evidence_count",
    "activity_count",
    "reference_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--validation-dataset",
        action="append",
        default=[],
        help="Dataset directory containing mappings/ and C_P.txt; repeatable.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_text(value) -> str:
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = TAG_RE.sub("", value)
    value = SPACE_RE.sub(" ", value).strip()
    return "" if value.upper() == "NULL" else value


def as_values(value) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def first_value(row: Mapping, key: str) -> str:
    values = as_values(row.get(key))
    return values[0] if values else ""


def stable_join(values: Iterable[str]) -> str:
    return "|".join(sorted({clean_text(value) for value in values if clean_text(value)}))


def open_writer(stack: ExitStack, path: Path, fields: Sequence[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(
        handle,
        fieldnames=list(fields),
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    return writer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_from_tail(path: Path) -> Dict[str, Dict[str, object]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 16384))
        tail = handle.read()
    result: Dict[str, Dict[str, object]] = {}
    for relation, expected, actual, status in SUMMARY_ENTRY_RE.findall(tail):
        result[relation.decode("utf-8")] = {
            "expected": int(expected),
            "actual": int(actual),
            "status": status.decode("utf-8"),
        }
    return result


def brace_delta(text: str) -> int:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
    return depth


def consume_object(handle: Iterator[str], history: deque) -> Dict:
    lines = list(history)
    start = None
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].lstrip().startswith("{"):
            start = index
            break
    if start is None:
        raise ValueError("Could not locate object start before JSON marker")
    selected = lines[start:]
    depth = brace_delta("".join(selected))
    while depth > 0:
        line = next(handle)
        selected.append(line)
        depth += brace_delta(line)
    if depth != 0:
        raise ValueError("Unbalanced JSON object")
    payload = "".join(selected).rstrip()
    if payload.endswith(","):
        payload = payload[:-1]
    return json.loads(payload)


def parse_selected_page(
    path: Path,
    summary: Mapping[str, Mapping[str, object]],
) -> Tuple[Optional[Dict], Dict[str, Dict]]:
    expected_relations = {
        relation
        for relation in WANTED_RELATIONS
        if int(summary.get(relation, {}).get("actual", 0)) > 0
    }
    base = None
    relations: Dict[str, Dict] = {}
    history: deque = deque(maxlen=6)
    with path.open(encoding="utf-8", errors="replace") as handle:
        iterator = iter(handle)
        for line in iterator:
            history.append(line)
            if base is None and '"id": "base_information"' in line:
                base = consume_object(iterator, history)
                history.clear()
            else:
                relation = next(
                    (
                        name
                        for name in expected_relations - relations.keys()
                        if f'"type": "{name}"' in line
                    ),
                    None,
                )
                if relation:
                    relations[relation] = consume_object(iterator, history)
                    history.clear()
            if base is not None and expected_relations.issubset(relations):
                break
    return base, relations


def collect_key_values(value, output: MutableMapping[str, object]) -> None:
    if isinstance(value, dict):
        key = clean_text(value.get("key"))
        if key and "value" in value:
            output[key] = value.get("value")
        for child in value.values():
            if isinstance(child, (dict, list)):
                collect_key_values(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_key_values(child, output)


def tcmip_from_fields(fields: Mapping[str, object]) -> str:
    match = TCMIP_RE.search(clean_text(fields.get("2D Structure")))
    return match.group() if match else ""


def relation_rows(block: Optional[Mapping]) -> List[Dict]:
    if not block:
        return []
    rows = block.get("value") or []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def reference_values(row: Mapping) -> Tuple[List[str], List[str]]:
    labels = as_values(row.get("References"))
    linkformat = row.get("linkformat") or {}
    reference = linkformat.get("References") or {} if isinstance(linkformat, dict) else {}
    urls = as_values(reference.get("param")) if isinstance(reference, dict) else []
    return labels, urls


def load_entity_maps(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    compound_map: Dict[str, str] = {}
    protein_map: Dict[str, str] = {}
    mapping_dir = dataset_dir / "mappings"
    with (mapping_dir / "compound_id_map.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            for tcmip_id in (row.get("tcmip_ids") or "").split("|"):
                if tcmip_id.strip():
                    compound_map[tcmip_id.strip()] = row["compound_id"]
    with (mapping_dir / "protein_id_map.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            symbols = row.get("gene_symbols") or row.get("protein_key") or ""
            for symbol in symbols.split("|"):
                if symbol.strip():
                    protein_map[symbol.strip().upper()] = row["protein_id"]
    return {"compounds": compound_map, "proteins": protein_map}


def load_edges(path: Path) -> Set[Tuple[str, str]]:
    edges = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                edges.add((parts[0], parts[1]))
    return edges


def prepare_output(path: Path, overwrite: bool) -> Path:
    staging = path.with_name(path.name + ".building")
    for candidate in (path, staging):
        if candidate.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{candidate} already exists; use --overwrite to replace it"
                )
            shutil.rmtree(candidate)
    staging.mkdir(parents=True)
    return staging


def pair_mapping_status(compound_id: str, protein_id: str) -> str:
    if compound_id and protein_id:
        return "mapped"
    if not compound_id and not protein_id:
        return "compound_and_protein_unmapped"
    if not compound_id:
        return "compound_unmapped"
    return "protein_unmapped"


def update_pair(
    store: MutableMapping[Tuple[str, str], Dict[str, object]],
    key: Tuple[str, str],
    activity: str,
    reference_urls: Sequence[str],
    score: str,
) -> None:
    record = store.setdefault(
        key,
        {
            "evidence_count": 0,
            "activities": set(),
            "reference_urls": set(),
            "score_count": 0,
            "score_sum": 0.0,
            "score_min": None,
            "score_max": None,
        },
    )
    record["evidence_count"] += 1
    if activity:
        record["activities"].add(activity)
    record["reference_urls"].update(reference_urls)
    if score:
        numeric = float(score)
        record["score_count"] += 1
        record["score_sum"] += numeric
        record["score_min"] = (
            numeric if record["score_min"] is None else min(record["score_min"], numeric)
        )
        record["score_max"] = (
            numeric if record["score_max"] is None else max(record["score_max"], numeric)
        )


def pair_row(
    key: Tuple[str, str],
    record: Mapping[str, object],
    ingredient_names: Mapping[str, str],
    compound_map: Mapping[str, str],
    protein_map: Mapping[str, str],
) -> Dict[str, object]:
    tcmip_id, gene_symbol = key
    compound_id = compound_map.get(tcmip_id, "")
    protein_id = protein_map.get(gene_symbol, "")
    score_count = int(record["score_count"])
    return {
        "tcmip_id": tcmip_id,
        "compound_id_processed": compound_id,
        "compound_name": ingredient_names.get(tcmip_id, ""),
        "gene_symbol": gene_symbol,
        "protein_id_processed": protein_id,
        "evidence_count": record["evidence_count"],
        "activity_count": len(record["activities"]),
        "reference_count": len(record["reference_urls"]),
        "similar_score_min": record["score_min"] if score_count else "",
        "similar_score_mean": (
            f"{record['score_sum'] / score_count:.6f}" if score_count else ""
        ),
        "similar_score_max": record["score_max"] if score_count else "",
        "mapping_status": pair_mapping_status(compound_id, protein_id),
    }


def write_validation_sets(
    staging: Path,
    confirmed_pairs: Mapping[Tuple[str, str], Mapping[str, object]],
    ingredient_names: Mapping[str, str],
    dataset_dirs: Sequence[Path],
) -> Dict[str, Dict[str, object]]:
    results = {}
    for dataset_dir in dataset_dirs:
        maps = load_entity_maps(dataset_dir)
        training_edges = load_edges(dataset_dir / "C_P.txt")
        counts = Counter()
        validation_dir = staging / "validation" / dataset_dir.name
        with ExitStack() as stack:
            writers = {
                status: open_writer(
                    stack, validation_dir / f"{status}.tsv", VALIDATION_FIELDS
                )
                for status in (
                    "training_overlap",
                    "unseen_confirmed",
                    "out_of_vocabulary",
                )
            }
            for (tcmip_id, gene_symbol), record in sorted(confirmed_pairs.items()):
                compound_id = maps["compounds"].get(tcmip_id, "")
                protein_id = maps["proteins"].get(gene_symbol, "")
                if not compound_id or not protein_id:
                    status = "out_of_vocabulary"
                elif (compound_id, protein_id) in training_edges:
                    status = "training_overlap"
                else:
                    status = "unseen_confirmed"
                counts[status] += 1
                writers[status].writerow(
                    {
                        "tcmip_id": tcmip_id,
                        "compound_id": compound_id,
                        "compound_name": ingredient_names.get(tcmip_id, ""),
                        "gene_symbol": gene_symbol,
                        "protein_id": protein_id,
                        "status": status,
                        "evidence_count": record["evidence_count"],
                        "activity_count": len(record["activities"]),
                        "reference_count": len(record["reference_urls"]),
                    }
                )
        results[dataset_dir.name] = {
            "path": str(dataset_dir.resolve()),
            "training_edge_count": len(training_edges),
            "confirmed_pair_count": len(confirmed_pairs),
            **dict(counts),
        }
    return results


def write_statistics_markdown(
    path: Path,
    stats: Mapping[str, object],
    validation: Mapping[str, Mapping[str, object]],
) -> None:
    relations = stats["relation_summary"]
    fields = stats["field_coverage"]
    lines = [
        "# ETCM2.0 Ingredient Validation Evidence Statistics",
        "",
        "该目录仅用于冻结模型后的外部验证和案例解释，不参与训练、调参、早停或模型选择。",
        "",
        "## 文件与解析",
        "",
        f"- JSON 文件：{stats['source_file_count']:,}",
        f"- 成功页面：{stats['successful_pages']:,}",
        f"- 无数据页面：{stats['failed_pages']:,}",
        f"- 原始体积：{stats['source_size_bytes'] / 1024 ** 3:.2f} GiB",
        "",
        "## 基础字段覆盖",
        "",
        "| 字段 | 数量 | 覆盖率 |",
        "|---|---:|---:|",
    ]
    for field, count in fields.items():
        rate = count / stats["successful_pages"] if stats["successful_pages"] else 0
        lines.append(f"| {field} | {count:,} | {rate:.2%} |")
    lines.extend(
        [
            "",
            "## 原始关系摘要",
            "",
            "| 关系 | 页面数 | 原始行数 |",
            "|---|---:|---:|",
        ]
    )
    for relation, values in relations.items():
        lines.append(
            f"| {relation} | {values['pages']:,} | {values['raw_rows']:,} |"
        )
    lines.extend(
        [
            "",
            "## 紧凑关系表",
            "",
            f"- 确认靶点唯一关系：{stats['confirmed_pair_count']:,}",
            f"- 潜在靶点唯一关系：{stats['potential_pair_count']:,}",
            f"- 药材唯一关系：{stats['herb_pair_count']:,}",
            "",
            "## 确认边与训练数据隔离审计",
            "",
            "| 数据集 | 训练重叠 | 未见确认边 | OOV |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset, values in validation.items():
        lines.append(
            f"| {dataset} | {values.get('training_overlap', 0):,} | "
            f"{values.get('unseen_confirmed', 0):,} | "
            f"{values.get('out_of_vocabulary', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- `confirmed_target_pairs.tsv` 可用于外部阳性验证。",
            "- `potential_target_pairs.tsv` 是相似性推断关系，只能用于一致性参考。",
            "- `ingredient_herb.tsv` 可用于药材上下文解释。",
            "- 未记录的 C-P 关系属于 unlabeled，不能直接宣称为生物学负样本。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    dataset_dirs = [
        Path(value).expanduser().resolve() for value in args.validation_dataset
    ] or [path.resolve() for path in DEFAULT_DATASETS if path.exists()]
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    for dataset_dir in dataset_dirs:
        for required in (
            dataset_dir / "mappings" / "compound_id_map.csv",
            dataset_dir / "mappings" / "protein_id_map.csv",
            dataset_dir / "C_P.txt",
        ):
            if not required.exists():
                raise FileNotFoundError(required)

    staging = prepare_output(output_dir, args.overwrite)
    processed_maps = load_entity_maps(REPOSITORY_ROOT / "dataset" / "ETCM2.0_processed")
    files = sorted(input_dir.glob("*.json"), key=lambda path: path.name.casefold())
    if args.max_files:
        files = files[: args.max_files]

    source_fingerprint = hashlib.sha256()
    source_size = 0
    stats = Counter()
    relation_stats: Dict[str, Counter] = defaultdict(Counter)
    field_coverage = Counter()
    ingredient_names: Dict[str, str] = {}
    confirmed_pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
    potential_pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
    herb_pairs: Set[Tuple[str, str]] = set()
    parse_issues: List[Dict[str, str]] = []
    confirmed_evidence_id = 0
    potential_evidence_id = 0

    with ExitStack() as stack:
        ingredient_writer = open_writer(
            stack, staging / "entities" / "ingredient.tsv", INGREDIENT_FIELDS
        )
        alias_writer = open_writer(
            stack, staging / "entities" / "ingredient_alias.tsv", ALIAS_FIELDS
        )
        herb_writer = open_writer(
            stack, staging / "relations" / "ingredient_herb.tsv", HERB_FIELDS
        )
        confirmed_writer = open_writer(
            stack,
            staging / "relations" / "confirmed_target_evidence.tsv",
            EVIDENCE_FIELDS,
        )
        potential_writer = open_writer(
            stack,
            staging / "relations" / "potential_target_evidence.tsv",
            EVIDENCE_FIELDS,
        )
        failure_writer = open_writer(
            stack,
            staging / "audit" / "failed_pages.tsv",
            ("source_file", "status", "message"),
        )

        for index, path in enumerate(files, start=1):
            size = path.stat().st_size
            source_size += size
            source_fingerprint.update(path.name.encode("utf-8"))
            source_fingerprint.update(b"\0")
            source_fingerprint.update(str(size).encode("ascii"))
            source_fingerprint.update(b"\n")
            summary = summary_from_tail(path)
            if not summary:
                stats["failed_pages"] += 1
                head = path.read_text(encoding="utf-8", errors="replace")[:512]
                message = ""
                try:
                    message = clean_text(json.loads(head).get("msg"))
                except (json.JSONDecodeError, AttributeError):
                    message = clean_text(head)
                failure_writer.writerow(
                    {
                        "source_file": path.name,
                        "status": "no_data_or_missing_summary",
                        "message": message,
                    }
                )
                continue

            stats["successful_pages"] += 1
            for relation, values in summary.items():
                if int(values["expected"]) != int(values["actual"]):
                    parse_issues.append(
                        {
                            "source_file": path.name,
                            "issue": "summary_count_mismatch",
                            "detail": f"{relation}:{values}",
                        }
                    )
                if int(values["actual"]) > 0:
                    relation_stats[relation]["pages"] += 1
                    relation_stats[relation]["raw_rows"] += int(values["actual"])

            try:
                base, relation_blocks = parse_selected_page(path, summary)
            except Exception as exc:
                parse_issues.append(
                    {
                        "source_file": path.name,
                        "issue": "selective_parse_error",
                        "detail": repr(exc),
                    }
                )
                continue
            if not base:
                parse_issues.append(
                    {
                        "source_file": path.name,
                        "issue": "missing_base_information",
                        "detail": "",
                    }
                )
                continue

            missing_blocks = {
                relation
                for relation in WANTED_RELATIONS
                if int(summary.get(relation, {}).get("actual", 0)) > 0
                and relation not in relation_blocks
            }
            if missing_blocks:
                parse_issues.append(
                    {
                        "source_file": path.name,
                        "issue": "missing_selected_relation_block",
                        "detail": stable_join(missing_blocks),
                    }
                )
                continue

            fields: Dict[str, object] = {}
            collect_key_values(base, fields)
            tcmip_id = tcmip_from_fields(fields)
            compound_name = clean_text(fields.get("Component Name IN English")) or path.stem
            ingredient_names[tcmip_id] = compound_name
            compound_id = processed_maps["compounds"].get(tcmip_id, "")
            aliases = as_values(fields.get("Component Alias"))
            pubchem = as_values(fields.get("PubChem"))
            present = {
                "english_name": bool(clean_text(fields.get("Component Name IN English"))),
                "aliases": bool(aliases),
                "molecular_formula": bool(clean_text(fields.get("Molecular Formula"))),
                "molecular_weight": bool(clean_text(fields.get("Molecular Weight"))),
                "cas_number": bool(clean_text(fields.get("CAS Number"))),
                "pubchem_cid": bool(pubchem),
                "structure_2d": bool(clean_text(fields.get("2D Structure"))),
            }
            field_coverage.update(key for key, value in present.items() if value)
            counts = {
                relation: int(summary.get(relation, {}).get("actual", 0))
                for relation in summary
            }
            ingredient_writer.writerow(
                {
                    "tcmip_id": tcmip_id,
                    "compound_id_processed": compound_id,
                    "compound_name": compound_name,
                    "molecular_formula": clean_text(fields.get("Molecular Formula")),
                    "molecular_weight": clean_text(fields.get("Molecular Weight")),
                    "cas_number": clean_text(fields.get("CAS Number")),
                    "pubchem_cids": stable_join(pubchem),
                    "structure_url": clean_text(fields.get("2D Structure")),
                    "qed": clean_text(
                        fields.get("(Quantitative Estimate of Drug-likeness)")
                    ),
                    "synthetic_accessibility": clean_text(
                        fields.get("(Synthetic accessibility score)")
                    ),
                    "natural_product_likeness": clean_text(
                        fields.get("(Natural Product-likeness score)")
                    ),
                    "lipinski_rule": clean_text(fields.get("Lipinski Rule")),
                    "gsk_rule": clean_text(fields.get("GSK Rule")),
                    "golden_triangle": clean_text(fields.get("Golden Triangle")),
                    "fdamdd": clean_text(
                        fields.get("FDAMDD(FDA Maximum (Recommended) Daily Dose)")
                    ),
                    "herb_count": counts.get("herb", 0),
                    "confirmed_target_count": counts.get("target", 0),
                    "potential_target_count": counts.get("similar_target", 0),
                    "disease_count": counts.get("disease", 0),
                    "formula_count": counts.get(
                        "traditional_chinese_medicine_formula", 0
                    ),
                    "patent_drug_count": counts.get("chinese_patent_drug", 0),
                    "mapping_status": "mapped" if compound_id else "unmapped",
                    "source_file": path.name,
                }
            )
            for alias in sorted(set(aliases)):
                alias_writer.writerow(
                    {
                        "tcmip_id": tcmip_id,
                        "compound_name": compound_name,
                        "alias": alias,
                        "source_file": path.name,
                    }
                )

            for row in relation_rows(relation_blocks.get("herb")):
                herb_names = as_values(row.get("Herb Name in Pinyin"))
                for herb_name in herb_names:
                    key = (tcmip_id, herb_name)
                    if key in herb_pairs:
                        continue
                    herb_pairs.add(key)
                    herb_writer.writerow(
                        {
                            "tcmip_id": tcmip_id,
                            "compound_id_processed": compound_id,
                            "compound_name": compound_name,
                            "herb_name_pinyin": herb_name,
                            "herb_name_latin": stable_join(
                                as_values(row.get("Herb Name in Latin"))
                            ),
                            "property": clean_text(row.get("Property")),
                            "flavor": clean_text(row.get("Flavor")),
                            "meridian_tropism": clean_text(row.get("Meridian Tropism")),
                            "source_file": path.name,
                        }
                    )

            for relation, pair_store, writer, prefix in (
                ("target", confirmed_pairs, confirmed_writer, "confirmed"),
                ("similar_target", potential_pairs, potential_writer, "potential"),
            ):
                for row in relation_rows(relation_blocks.get(relation)):
                    genes = as_values(row.get("Gene Symbol"))
                    labels, urls = reference_values(row)
                    activity = clean_text(row.get("Activity"))
                    score = clean_text(row.get("Similar Score"))
                    for gene_symbol in genes:
                        gene_symbol = gene_symbol.upper()
                        protein_id = processed_maps["proteins"].get(gene_symbol, "")
                        if relation == "target":
                            confirmed_evidence_id += 1
                            evidence_id = f"confirmed:{confirmed_evidence_id:08d}"
                        else:
                            potential_evidence_id += 1
                            evidence_id = f"potential:{potential_evidence_id:08d}"
                        writer.writerow(
                            {
                                "evidence_id": evidence_id,
                                "tcmip_id": tcmip_id,
                                "compound_id_processed": compound_id,
                                "compound_name": compound_name,
                                "gene_symbol": gene_symbol,
                                "protein_id_processed": protein_id,
                                "activity": activity,
                                "similar_score": score,
                                "reference_labels": stable_join(labels),
                                "reference_urls": stable_join(urls),
                                "source_file": path.name,
                            }
                        )
                        update_pair(
                            pair_store,
                            (tcmip_id, gene_symbol),
                            activity,
                            urls,
                            score,
                        )

            if args.progress_every and (
                index == 1
                or index == len(files)
                or index % args.progress_every == 0
            ):
                print(
                    f"ingredients: {index}/{len(files)} "
                    f"confirmed_pairs={len(confirmed_pairs)} "
                    f"potential_pairs={len(potential_pairs)} "
                    f"herb_pairs={len(herb_pairs)}",
                    flush=True,
                )

    with ExitStack() as stack:
        confirmed_pair_writer = open_writer(
            stack,
            staging / "relations" / "confirmed_target_pairs.tsv",
            PAIR_FIELDS,
        )
        potential_pair_writer = open_writer(
            stack,
            staging / "relations" / "potential_target_pairs.tsv",
            PAIR_FIELDS,
        )
        for key, record in sorted(confirmed_pairs.items()):
            confirmed_pair_writer.writerow(
                pair_row(
                    key,
                    record,
                    ingredient_names,
                    processed_maps["compounds"],
                    processed_maps["proteins"],
                )
            )
        for key, record in sorted(potential_pairs.items()):
            potential_pair_writer.writerow(
                pair_row(
                    key,
                    record,
                    ingredient_names,
                    processed_maps["compounds"],
                    processed_maps["proteins"],
                )
            )

    validation = write_validation_sets(
        staging, confirmed_pairs, ingredient_names, dataset_dirs
    )
    audit_dir = staging / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "parse_issues.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_file", "issue", "detail"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(parse_issues)

    statistics = {
        "source_file_count": len(files),
        "source_size_bytes": source_size,
        "successful_pages": stats["successful_pages"],
        "failed_pages": stats["failed_pages"],
        "field_coverage": dict(field_coverage),
        "relation_summary": {
            relation: {
                "pages": relation_stats[relation]["pages"],
                "raw_rows": relation_stats[relation]["raw_rows"],
            }
            for relation in sorted(relation_stats)
        },
        "confirmed_evidence_count": confirmed_evidence_id,
        "confirmed_pair_count": len(confirmed_pairs),
        "potential_evidence_count": potential_evidence_id,
        "potential_pair_count": len(potential_pairs),
        "herb_pair_count": len(herb_pairs),
        "parse_issue_count": len(parse_issues),
    }
    (staging / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_statistics_markdown(staging / "statistics.md", statistics, validation)

    output_hashes = {}
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            output_hashes[str(path.relative_to(staging))] = sha256_file(path)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "external_validation_and_explanation_only",
        "training_use_prohibited": True,
        "source_directory": str(input_dir),
        "source_file_count": len(files),
        "source_size_bytes": source_size,
        "source_inventory_sha256": source_fingerprint.hexdigest(),
        "max_files": args.max_files,
        "validation_datasets": [str(path) for path in dataset_dirs],
        "validation_summary": validation,
        "output_sha256": output_hashes,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_dir)
    print(f"Validation evidence written to: {output_dir}")


if __name__ == "__main__":
    main()

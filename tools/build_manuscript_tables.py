#!/usr/bin/env python3
"""Build manuscript tables from frozen results without training or rescoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "paper_artifacts" / "results"

DATASETS = (
    {
        "name": "TCM-Suite",
        "slug": "tcmsuite",
        "hc": ROOT / "dataset" / "TCMsuite" / "H_C.txt",
        "cp": ROOT / "dataset" / "TCMsuite" / "C_P.txt",
        "pd": ROOT / "dataset" / "TCMsuite" / "P_D.txt",
    },
    {
        "name": "TCMSP",
        "slug": "tcmsp",
        "hc": ROOT / "dataset" / "TCMSP" / "herb-compound.txt",
        "cp": ROOT / "dataset" / "TCMSP" / "compound-protein.txt",
        "pd": ROOT / "dataset" / "TCMSP" / "target-disease.txt",
    },
    {
        "name": "SymMap2.0",
        "slug": "symmap",
        "hc": ROOT / "dataset" / "Symmap" / "HI.txt",
        "cp": ROOT / "dataset" / "Symmap" / "IT.txt",
        "pd": ROOT / "dataset" / "Symmap" / "TD.txt",
    },
    {
        "name": "ETCM2.0-mention10",
        "slug": "etcm_mention10",
        "hc": ROOT / "dataset" / "ETCM2.0_core_mention10" / "H_C.txt",
        "cp": ROOT / "dataset" / "ETCM2.0_core_mention10" / "C_P.txt",
        "pd": ROOT / "dataset" / "ETCM2.0_core_mention10" / "P_D.txt",
    },
)

BASELINE_ORDER = (
    "Dual-HGNN-CTI",
    "LightGCN-CTI",
    "R-GCN-CTI",
    "HGT-CTI",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths() -> dict[str, Path]:
    manifest_path = RESULT_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {}
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError("Missing frozen artifact: {}".format(path))
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(
                "Frozen artifact hash mismatch for {}: {} != {}".format(
                    path, actual, record["sha256"]
                )
            )
        paths[record["name"]] = path
    return paths


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def relation(path: Path):
    if not path.is_file():
        raise FileNotFoundError("Missing relation file: {}".format(path))
    edges = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    "Malformed relation row {}:{}".format(path, line_number)
                )
            edges.add((fields[0], fields[1]))
    return edges


def relation_statistics(dataset, protocol_row):
    hc = relation(dataset["hc"])
    cp = relation(dataset["cp"])
    pd = relation(dataset["pd"])
    herbs = {left for left, _ in hc}
    hc_compounds = {right for _, right in hc}
    cp_compounds = {left for left, _ in cp}
    diseases = {right for _, right in pd}
    coverage = len(cp_compounds & hc_compounds) / float(len(cp_compounds))
    evaluated = protocol_row["full_candidate"]["evaluated_compounds"]
    if evaluated != len(cp_compounds):
        raise ValueError(
            "Evaluated compound mismatch for {}: {} != {}".format(
                dataset["name"], evaluated, len(cp_compounds)
            )
        )
    if protocol_row["split_audit"]["positive_count"] != len(cp):
        raise ValueError("C-P positive count mismatch for {}".format(dataset["name"]))
    return {
        "herbs": len(herbs),
        "compounds": len(cp_compounds),
        "proteins": protocol_row["full_candidate"]["candidate_count"]["maximum"],
        "diseases": len(diseases),
        "hc_edges": len(hc),
        "cp_edges": len(cp),
        "pd_edges": len(pd),
        "hc_coverage": coverage,
    }


def parse_metric(value: str):
    match = re.match(r"^([0-9.]+)\(±([0-9.]+)\)$", value)
    if not match:
        raise ValueError("Invalid metric summary: {}".format(value))
    return float(match.group(1)), float(match.group(2))


def summary(mean: float, std: float) -> str:
    return "{:.6f} ± {:.6f}".format(mean, std)


def markdown_table(headers, rows, numeric_columns=()):
    aligns = []
    for index in range(len(headers)):
        aligns.append("---:" if index in numeric_columns else "---")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(aligns) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def table_1(protocol):
    by_name = {row["dataset"]: row for row in protocol["rows"]}
    rows = []
    for dataset in DATASETS:
        stats = relation_statistics(dataset, by_name[dataset["name"]])
        rows.append(
            (
                dataset["name"],
                f"{stats['herbs']:,}",
                f"{stats['compounds']:,}",
                f"{int(stats['proteins']):,}",
                f"{stats['diseases']:,}",
                f"{stats['hc_edges']:,}",
                f"{stats['cp_edges']:,}",
                f"{stats['pd_edges']:,}",
                "{:.2%}".format(stats["hc_coverage"]),
            )
        )
    body = markdown_table(
        ("Dataset", "H", "C", "P candidates", "D", "H-C", "C-P+", "P-D", "H-C coverage"),
        rows,
        numeric_columns=tuple(range(1, 9)),
    )
    return "\n".join(
        (
            "## Table 1",
            "",
            "**Table 1. Graph scale and compound cold-start evaluation coverage.** "
            "H and D are unique entities occurring in H-C and P-D, respectively; "
            "C is the set of compounds with at least one known C-P link and therefore "
            "eligible for compound-disjoint evaluation; P candidates is the complete "
            "mapped protein candidate set. Relation counts are unique pairs. H-C "
            "coverage is the fraction of evaluated compounds with at least one H-C link.",
            "",
            body,
        )
    )


def table_2(lockbox):
    headers = (
        "Dataset", "Pred. AUPR", "SACTI AUPR", "Delta",
        "Pred. MRR", "SACTI MRR", "Delta",
        "Pred. R@20", "SACTI R@20", "Delta",
    )
    rows = []
    for record in lockbox["rows"]:
        values = []
        for metric in ("AUPR", "MRR", "Recall@20"):
            result = record["metrics"][metric]
            values.extend((
                "{:.6f}".format(result["baseline_mean"]),
                "{:.6f}".format(result["candidate_mean"]),
                "{:+.6f}".format(result["delta"]),
            ))
        rows.append((record["dataset"],) + tuple(values))
    macro = []
    for metric in ("AUPR", "MRR", "Recall@20"):
        result = lockbox["macro"][metric]
        macro.extend((
            "{:.6f}".format(result["baseline"]),
            "{:.6f}".format(result["candidate"]),
            "{:+.6f}".format(result["delta"]),
        ))
    rows.append(("Macro",) + tuple(macro))
    body = markdown_table(headers, rows, numeric_columns=tuple(range(1, 10)))
    return "\n".join(
        (
            "## Table 2",
            "",
            "**Table 2. Unseen seed-901026 lockbox comparison between SACTI and "
            "the matched HDCTI-based predecessor.** Values are means across five "
            "compound-disjoint outer folds. AUPR denotes average precision; MRR is "
            "macro-averaged over compounds within each fold. Macro is the unweighted "
            "mean of database means. Delta is SACTI minus predecessor. The "
            "lockbox was sealed before split generation and reported regardless of direction.",
            "",
            body,
        )
    )


def table_3(ablation):
    stages = (
        ("ID + H-C context", "no_transfer"),
        ("+ Prototype", "prototype_only"),
        ("+ Jaccard", "jaccard_only"),
        ("Full SACTI", "full"),
        ("No gate", "no_support_gate"),
        ("No H-C context", "no_context_expert"),
    )
    headers = ("Dataset",) + tuple(label for label, _ in stages)
    rows = []
    values = {key: [] for _, key in stages}
    for record in ablation["rows"]:
        cells = []
        for _, key in stages:
            value = record["stages"][key]["metrics"]["AUPR"]
            values[key].append(value)
            cells.append("{:.6f}".format(value))
        rows.append((record["dataset"],) + tuple(cells))
    rows.append(("Macro",) + tuple(
        "{:.6f}".format(statistics.mean(values[key])) for _, key in stages
    ))
    body = markdown_table(headers, rows, numeric_columns=tuple(range(1, len(headers))))
    return "\n".join(
        (
            "## Table 3",
            "",
            "**Table 3. Frozen seed-52026 SACTI component ablation.** Values are "
            "mean AUPR (average precision) across five outer folds on fixed 1:1 "
            "positive-to-unlabelled pairs. Macro is the unweighted mean of database means. "
            "Prototype and Jaccard retain only the named transfer branch; No gate "
            "and No context remove the indicated native component from full SACTI.",
            "",
            body,
        )
    )


def table_4(ablation, baseline_rows):
    dataset_names = tuple(dataset["name"] for dataset in DATASETS)
    by_method = {method: {} for method in BASELINE_ORDER}
    for record in baseline_rows:
        if record["method"] in by_method:
            by_method[record["method"]][record["dataset"]] = parse_metric(record["AUPR"])
    full = {record["dataset"]: record["stages"]["full"] for record in ablation["rows"]}
    rows = []
    for method in BASELINE_ORDER:
        pairs = [by_method[method][dataset] for dataset in dataset_names]
        means = [pair[0] for pair in pairs]
        rows.append((method,) + tuple(summary(mean, std) for mean, std in pairs)
                    + ("{:.6f}".format(statistics.mean(means)),))
    cells = []
    means = []
    for dataset in dataset_names:
        stage = full[dataset]
        mean = stage["metrics"]["AUPR"]
        std = statistics.stdev(stage["fold_aupr"])
        means.append(mean)
        cells.append("**{}**".format(summary(mean, std)))
    rows.append(("SACTI",) + tuple(cells) +
                ("**{:.6f}**".format(statistics.mean(means)),))
    headers = ("Method",) + dataset_names + ("Macro",)
    body = markdown_table(headers, rows, numeric_columns=tuple(range(1, len(headers))))
    return "\n".join(
        (
            "## Table 4",
            "",
            "**Table 4. Compound cold-start comparison with same-input neural "
            "controls.** Values are mean ± sample SD (ddof=1) of AUPR (average precision) across five outer folds. Macro is the unweighted "
            "mean of database means. Controls are "
            "topology-budget adaptations rather than full reproductions of their "
            "original tasks or attribute inputs.",
            "",
            body,
        )
    )


def build_tables() -> str:
    artifacts = artifact_paths()
    ablation = read_json(artifacts["sacti_component_ablation"])
    lockbox = read_json(artifacts["sacti_outer_lockbox"])
    protocol = read_json(artifacts["unified_evaluation_protocol"])
    baselines = read_tsv(artifacts["cold_external_baselines"])
    if protocol["training_optimizer_steps"] != 0 or protocol["model_selection_reopened"]:
        raise ValueError("Unified evaluation artifact is not frozen pure inference.")
    return "\n\n".join(
        (
            "# Frozen Manuscript Tables",
            "> Generated from SHA-256-verified result artifacts and local relation files. "
            "No checkpoint is restored and no training or outer-test rescoring is performed.",
            table_1(protocol),
            table_2(lockbox),
            table_3(ablation),
            table_4(ablation, baselines),
        )
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "MANUSCRIPT_TABLES_EN.md",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_tables(), encoding="utf-8")
    print("Manuscript tables written to: {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

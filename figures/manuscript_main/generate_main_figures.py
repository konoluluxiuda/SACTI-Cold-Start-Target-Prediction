#!/usr/bin/env python3
"""Generate the four SACTI main-manuscript figures from frozen artifacts."""

import csv
import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RESULTS = ROOT / "paper_artifacts" / "results"

INK = "#263238"
MUTED = "#68767D"
GRID = "#DCE3E6"
BLUE = "#397B98"
TEAL = "#2D7F7A"
GOLD = "#B9852C"
PURPLE = "#80679A"
RED = "#B24F55"
GREEN = "#3C8061"
DATASET_COLORS = ["#315A7D", "#2D7F7A", "#B47A2A", "#8A5F87"]


def configure():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_tsv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def save(fig, stem):
    svg_path = OUTPUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUTPUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def box(ax, x, y, w, h, text, face, edge, size=7, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.018",
        transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=size, color=INK,
            fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=8, linewidth=1.0, color=color,
    ))


def draw_method():
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=0.12)
    protocol = fig.add_subplot(gs[0])
    model = fig.add_subplot(gs[1])
    for ax in (protocol, model):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    protocol.text(0.0, 1.0, "a", fontweight="bold", fontsize=9, va="top")
    protocol.text(0.035, 0.98, "Compound-disjoint nested evaluation",
                  fontsize=8.5, fontweight="bold", va="top", color=INK)
    box(protocol, 0.04, 0.18, 0.25, 0.42,
        "Model training\nC-P support available", "#E8F0F4", BLUE)
    box(protocol, 0.375, 0.18, 0.25, 0.42,
        "Inner validation\ncompound-disjoint", "#F4F0E5", GOLD)
    box(protocol, 0.71, 0.18, 0.25, 0.42,
        "Outer test\nall compound C-P held out", "#F5E8E9", RED,
        weight="bold")
    arrow(protocol, 0.29, 0.39, 0.375, 0.39)
    arrow(protocol, 0.625, 0.39, 0.71, 0.39)
    protocol.text(0.835, 0.08, "H-C provenance remains observable",
                  ha="center", color=BLUE, fontsize=6.6, fontweight="bold")

    model.text(0.0, 1.0, "b", fontweight="bold", fontsize=9, va="top")
    model.text(0.035, 0.98, "Independent SACTI computation graph",
               fontsize=8.5, fontweight="bold", va="top", color=INK)

    box(model, 0.02, 0.62, 0.10, 0.14, "H-C\nincidence", "#DCEAF0", BLUE)
    box(model, 0.02, 0.23, 0.10, 0.14, "P-D\nincidence", "#E5E9F0", PURPLE)
    box(model, 0.17, 0.58, 0.16, 0.22,
        "Sparse H-C\nresidual context\nencoder", "#EAF3F2", TEAL, weight="bold")
    box(model, 0.17, 0.19, 0.16, 0.22,
        "Sparse P-D\nresidual context\nencoder", "#EEEAF3", PURPLE, weight="bold")
    arrow(model, 0.12, 0.69, 0.17, 0.69, BLUE)
    arrow(model, 0.12, 0.30, 0.17, 0.30, PURPLE)

    box(model, 0.385, 0.68, 0.12, 0.12, "Compound ID\n$z_c$", "#F1F3F4", MUTED)
    box(model, 0.385, 0.48, 0.12, 0.12, "H-C context\n$h_c$", "#E5F2EF", TEAL)
    box(model, 0.385, 0.22, 0.12, 0.12, "Protein\n$z_p$", "#EEEAF3", PURPLE)
    arrow(model, 0.33, 0.69, 0.385, 0.54, TEAL)
    arrow(model, 0.33, 0.30, 0.385, 0.28, PURPLE)

    box(model, 0.56, 0.68, 0.14, 0.13,
        "Supported-ID expert\n$q_c z_c^Tz_p$", "#F1F3F4", MUTED)
    box(model, 0.545, 0.45, 0.17, 0.15,
        "Candidate H-C expert\n$(h_c\\odot w)^Tz_p$", "#E4F1ED", TEAL,
        size=6.4, weight="bold")
    box(model, 0.545, 0.16, 0.17, 0.18,
        "Fold-local LOCO transfer\nherb prototype $r_H$\nH-C neighbour $r_J$",
        "#F1EAF4", PURPLE, size=6.0, weight="bold")
    arrow(model, 0.505, 0.74, 0.56, 0.745)
    arrow(model, 0.505, 0.54, 0.545, 0.525, TEAL)
    arrow(model, 0.505, 0.28, 0.545, 0.525, PURPLE)

    box(model, 0.79, 0.34, 0.19, 0.32,
        "Unified score\n\n$q_c z_c^Tz_p$\n$+\\ (h_c\\odot w)^Tz_p$\n$+\\ \\alpha r_H+\\beta r_J$",
        "#F5E8E9", RED, size=7, weight="bold")
    arrow(model, 0.70, 0.745, 0.81, 0.66)
    arrow(model, 0.715, 0.525, 0.79, 0.51, TEAL)
    arrow(model, 0.70, 0.25, 0.81, 0.34, PURPLE)
    model.text(0.63, 0.08, "No PageRank, full-node attention or HDCTI model inheritance",
               ha="center", fontsize=6.5, color=MUTED)
    save(fig, "figure1_method_overview")
    plt.close(fig)


def draw_lockbox():
    data = load("sacti_outer_lockbox.json")
    datasets = [r["dataset"] for r in data["rows"]]
    metrics = ["AUPR", "MRR", "Recall@20"]
    labels = ["AUPR", "MRR", "Recall@20"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    source_rows = []
    for ax, metric, label, letter in zip(axes, metrics, labels, "abc"):
        values = [r["metrics"][metric]["delta"] for r in data["rows"]]
        colors = [GREEN if value >= 0 else RED for value in values]
        y = np.arange(len(datasets))
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0, color=INK, linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(datasets if metric == "AUPR" else [])
        ax.invert_yaxis()
        ax.set_title(f"{letter}  Delta {label}", loc="left", fontweight="bold")
        ax.set_xlabel("SACTI - predecessor")
        ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(x=0.18)
        span = max(abs(np.array(values)))
        for i, value in enumerate(values):
            offset = span * 0.035
            ax.text(value + (offset if value >= 0 else -offset), i,
                    f"{value:+.3f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=6.2)
            source_rows.append({"dataset": datasets[i], "metric": metric,
                                "delta": f"{value:.12f}"})
    fig.suptitle("Previously unseen outer lockbox", y=1.02,
                 fontsize=9, fontweight="bold", color=INK)
    fig.tight_layout(w_pad=1.0)
    with (OUTPUT / "figure2_lockbox_source.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "metric", "delta"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(source_rows)
    # Preserve the historical source-data alias for older manuscript snapshots.
    shutil.copyfile(OUTPUT / "figure2_lockbox_source.tsv",
                    OUTPUT / "figure2_ablation_source.tsv")
    save(fig, "figure2_matched_ablation")
    plt.close(fig)


def draw_controls():
    data = load("sacti_component_ablation.json")
    datasets = [r["dataset"] for r in data["rows"]]
    methods = ["Dual-HGNN-CTI", "LightGCN-CTI", "R-GCN-CTI", "HGT-CTI", "SACTI"]
    baselines = {(r["method"], r["dataset"]): r["aupr"] for r in data["external_baselines"]}
    full = {r["dataset"]: r["stages"]["full"]["metrics"]["AUPR"] for r in data["rows"]}
    values = {
        method: [full[d] if method == "SACTI" else baselines[(method, d)] for d in datasets]
        for method in methods
    }
    baseline_sd = {}
    for row in load_tsv("cold_external_baselines.tsv"):
        mean, sd = row["AUPR"].rstrip(")").split("(±")
        key = (row["method"], row["dataset"])
        if not np.isclose(float(mean), baselines[key], atol=1e-12, rtol=0):
            raise ValueError(f"Inconsistent baseline mean: {key}")
        baseline_sd[key] = float(sd)
    full_sd = {r["dataset"]: float(np.std(
        r["stages"]["full"]["fold_aupr"], ddof=1)) for r in data["rows"]}
    errors = {
        method: [full_sd[d] if method == "SACTI" else baseline_sd[(method, d)]
                 for d in datasets] for method in methods
    }
    colors = ["#60758C", "#A5AFB5", "#8292A2", "#76918C", RED]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(datasets))
    width = 0.15
    for index, (method, color) in enumerate(zip(methods, colors)):
        offset = (index - 2) * width
        ax.errorbar(x + offset, values[method], yerr=errors[method],
                    fmt=["o", "s", "^", "D", "*"][index], markersize=5 if method == "SACTI" else 4, capsize=2, elinewidth=0.8,
                    label=method, color=color, zorder=3)
        if method == "SACTI":
            for xpos, value, sd in zip(x + offset, values[method], errors[method]):
                ax.text(xpos, value + sd + 0.012,
                        f"{value:.3f}", ha="center", va="bottom", fontsize=6,
                        color=RED, fontweight="bold")
    ax.set_ylabel("Outer-fold AUPR (mean ± SD)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.5, 1.015)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.04))
    ax.set_title("Same-input neural controls under compound cold-start",
                 loc="left", fontsize=9, fontweight="bold", color=INK, pad=42)
    fig.tight_layout()
    rows = []
    for method in methods:
        for dataset, value, sd in zip(datasets, values[method], errors[method]):
            rows.append({"method": method, "dataset": dataset, "aupr": f"{value:.12f}",
                         "sd": f"{sd:.12f}", "fold_count": 5})
    with (OUTPUT / "figure3_external_baselines_source.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "dataset", "aupr", "sd", "fold_count"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    save(fig, "figure3_external_baselines")
    plt.close(fig)


def distinct_compounds(rows, count):
    selected = []
    compounds = set()
    for row in rows:
        if row["compound_id"] in compounds:
            continue
        selected.append(row)
        compounds.add(row["compound_id"])
        if len(selected) == count:
            break
    return selected


def draw_etcm_validation(include_retrieval=False, case_source_note=None):
    evaluation = load("etcm_confirmed_target_evaluation.json")
    context = load("etcm_confirmed_target_context_audit.json")
    if context.get("support_population") != "verified_inner_training_positive_compounds":
        raise ValueError("Figure 4 requires the corrected inner-training donor audit.")
    pair_rows = load_tsv("etcm_confirmed_target_pair_ranks.tsv")
    evidence_rows = load_tsv("etcm_confirmed_target_evidence.tsv")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.08, 1.0], height_ratios=[1, 1],
        wspace=0.43, hspace=0.70,
    )
    curve_ax = fig.add_subplot(gs[0, 0])
    strata_ax = fig.add_subplot(gs[0, 1])
    cases_ax = fig.add_subplot(gs[1, :])

    ks = [1, 5, 10, 20, 50]
    x = np.arange(len(ks))
    curve_styles = [
        ("NoContext", MUTED, "-", 1.2),
        ("Hctx-P", TEAL, "-", 1.5),
        ("SACTI", RED, "-", 2.2),
        ("HerbPrototype-EB", GOLD, "--", 1.25),
        ("HC-Jaccard-LP", PURPLE, "--", 1.25),
    ]
    curve_source = []
    for method, color, linestyle, linewidth in curve_styles:
        summary = evaluation["methods"][method]["summary"]
        values = [float(summary["Recall@{}".format(k)]) for k in ks]
        if include_retrieval or method not in ("HerbPrototype-EB", "HC-Jaccard-LP"):
            curve_ax.plot(
                x, values, color=color, linestyle=linestyle, linewidth=linewidth,
                marker="o", markersize=3.5, label=method, zorder=3,
            )
        for k, value in zip(ks, values):
            curve_source.append({"method": method, "k": k, "recall": value})
    random_values = [k / float(evaluation["candidate_proteins"]) for k in ks]
    curve_ax.plot(
        x, random_values, color=INK, linestyle=":", linewidth=1.0,
        label="Random", zorder=2,
    )
    for k, value in zip(ks, random_values):
        curve_source.append({"method": "Random", "k": k, "recall": value})
    curve_ax.set_xticks(x)
    curve_ax.set_xticklabels(ks)
    curve_ax.set_xlabel("Candidate cutoff K")
    curve_ax.set_ylabel("Macro Recall@K")
    curve_ax.set_ylim(-0.02, 0.75)
    curve_ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    curve_ax.set_axisbelow(True)
    curve_ax.set_title(
        "a  Held-out ETCM2.0 retrieval", loc="left", fontweight="bold"
    )
    curve_ax.text(
        0.02, 0.96, "685 relations; 332 compounds; 509 targets",
        transform=curve_ax.transAxes, va="top", fontsize=6.2, color=MUTED,
    )
    curve_ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), ncol=2)

    strata = context["strata"]["supported_neighbor_count"]
    strata_names = ["1-5", "6-20", ">=21"]
    metric_specs = [("MRR", BLUE), ("Recall@20", TEAL)]
    bar_x = np.arange(len(strata_names))
    width = 0.34
    strata_source = []
    for index, (metric, color) in enumerate(metric_specs):
        values = [
            float(strata[name]["deltas"]["SACTI_minus_Hctx-P"][metric])
            for name in strata_names
        ]
        bars = strata_ax.bar(
            bar_x + (index - 0.5) * width, values, width * 0.9,
            color=color, label=metric, zorder=3,
        )
        for name, value, bar in zip(strata_names, values, bars):
            strata_source.append({
                "supported_neighbors": name,
                "metric": metric,
                "delta": value,
                "compounds": int(strata[name]["compounds"]),
                "evidence_pairs": int(strata[name]["evidence_pairs"]),
            })
            offset = 0.008 if value >= 0 else -0.010
            if index == 1 and abs(value - float(
                strata[name]["deltas"]["SACTI_minus_Hctx-P"]["MRR"]
            )) < 0.035:
                offset += 0.012 if value >= 0 else -0.012
            strata_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:+.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=5.5,
            )
    strata_ax.axhline(0, color=INK, linewidth=0.7)
    strata_ax.set_xticks(bar_x)
    strata_ax.set_xticklabels(
        [f"{name}\n(n={strata[name]['compounds']})" for name in strata_names]
    )
    strata_ax.set_ylabel("SACTI - Hctx-P")
    strata_ax.set_ylim(-0.17, 0.19)
    strata_ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    strata_ax.set_axisbelow(True)
    strata_ax.set_title(
        "b  Transfer by training support", loc="left", fontweight="bold"
    )
    strata_ax.legend(loc="lower right", ncol=2)

    evidence = {
        (row["compound_id"], row["protein_id"]): row
        for row in evidence_rows
    }
    eligible = []
    for row in pair_rows:
        key = (row["compound_id"], row["protein_id"])
        if key not in evidence or int(row["evidence_count"]) < 2:
            continue
        merged = dict(row)
        merged["compound_name"] = evidence[key]["compound_name"]
        eligible.append(merged)
    successes = distinct_compounds(
        sorted(
            (row for row in eligible if int(row["SACTI_rank"]) <= 20),
            key=lambda row: (
                int(row["SACTI_rank"]), -int(row["evidence_count"]),
                int(row["compound_id"]), int(row["protein_id"]),
            ),
        ),
        2,
    )
    misses = distinct_compounds(
        sorted(
            (row for row in eligible if int(row["SACTI_rank"]) > 20),
            key=lambda row: (
                -int(row["evidence_count"]), -int(row["activity_count"]),
                int(row["SACTI_rank"]), int(row["compound_id"]),
                int(row["protein_id"]),
            ),
        ),
        2,
    )
    case_rows = successes + misses
    # Source-review categories are distinct from model ranking outcomes.
    reviews = {
        ("147", "CA4"): ("Primary paper", "CA IV inhibition; stereochemistry unresolved"),
        ("16547", "CHRNA7"): ("Primary paper", "Receptor binding, not functional activation"),
        ("17498", "DRD4"): ("ChEMBL record", "Binding record; source full text unavailable"),
        ("13593", "SLC6A2"): ("Identity disputed", "Cited source concerns cocaine, not this isomer"),
    }
    cases_ax.axis("off")
    cases_ax.set_title("c  Case ranks and source verification", loc="left", fontweight="bold")
    cell_rows = []
    for row in case_rows:
        source, boundary = reviews[(str(row["compound_id"]), row["gene_symbol"])]
        cell_rows.append([
            row["compound_name"] + "\n" + row["gene_symbol"],
            str(row["SACTI_rank"]) + " / 509",
            source, boundary,
        ])
    table = cases_ax.table(
        cellText=cell_rows,
        colLabels=["Compound / target", "Rank", "Evidence source", "Supported interpretation / boundary"],
        colWidths=[0.23, 0.12, 0.18, 0.47],
        cellLoc="left", colLoc="left", bbox=[0, 0.04, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.4)
        cell.PAD = 0.035
        if row == 0:
            cell.set_facecolor("#EAF0F3")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#F7F9FA" if row % 2 else "white")
            if col == 1:
                cell.set_text_props(weight="bold", color=INK)
            if row == 4 and col >= 2:
                cell.set_facecolor("#FCF1E6")
    cases_ax.text(
        0, -0.035, case_source_note or "Post-hoc cases; source details in Supplementary Table S5. Rank is not evidence strength.",
        transform=cases_ax.transAxes, fontsize=6, color=MUTED,
    )

    fig.suptitle(
        "Same-resource held-out evidence: enrichment with explicit boundaries",
        x=0.5, y=1.01, fontsize=9, fontweight="bold", color=INK,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.91)

    with (OUTPUT / "figure4_retrieval_curve_source.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "k", "recall"],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curve_source)
    with (OUTPUT / "figure4_support_strata_source.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "supported_neighbors", "metric", "delta", "compounds",
                "evidence_pairs",
            ],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(strata_source)
    with (OUTPUT / "figure4_case_examples_source.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_group", "compound_id", "compound_name",
                "protein_id", "gene_symbol", "evidence_count",
                "activity_count", "reference_count", "SACTI_rank",
            ],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(case_rows):
            writer.writerow({
                "selection_group": "recovered" if index < len(successes) else "missed",
                "compound_id": row["compound_id"],
                "compound_name": row["compound_name"],
                "protein_id": row["protein_id"],
                "gene_symbol": row["gene_symbol"],
                "evidence_count": row["evidence_count"],
                "activity_count": row["activity_count"],
                "reference_count": row["reference_count"],
                "SACTI_rank": row["SACTI_rank"],
            })
    save(fig, "figureS1_etcm_extended" if include_retrieval else "figure4_etcm_validation")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=('all', '3', '4'), default='all')
    parser.add_argument('--sync-submission', action='store_true',
                        help='Copy the selected Figure 3/4 PDFs to the submission directory.')
    args = parser.parse_args()
    configure()
    if args.figure == 'all':
        draw_method()
        draw_lockbox()
    if args.figure in ('all', '3'):
        draw_controls()
    if args.figure in ('all', '4'):
        draw_etcm_validation(include_retrieval=True)
        draw_etcm_validation()
    if args.sync_submission:
        if args.figure in ('all', '4'):
            shutil.copyfile(OUTPUT / 'figureS1_etcm_extended.pdf',
                            ROOT / 'manuscript/jbi_elsarticle_submission/figures/figureS1_etcm_extended.pdf')
        stems = {'3': 'figure3_external_baselines', '4': 'figure4_etcm_validation'}
        for number, stem in stems.items():
            if args.figure in ('all', number):
                shutil.copyfile(OUTPUT / f'{stem}.pdf',
                                ROOT / f'manuscript/jbi_elsarticle_submission/figures/{stem}.pdf')
    print(f"SACTI manuscript figures written to: {OUTPUT}")


if __name__ == "__main__":
    main()

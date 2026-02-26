#!/usr/bin/env python3
"""Generate a clean ParE vs GyrB evidence summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGS = ROOT / "figures"


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    matrix = pd.read_csv(DATA / "pare_vs_gyrb_evidence_matrix.tsv", sep="\t").set_index("protein")
    winners = pd.read_csv(DATA / "pare_vs_gyrb_evidence_metric_winners.tsv", sep="\t")
    comp = pd.read_csv(DATA / "pare_vs_gyrb_evidence_composite.tsv", sep="\t").set_index("protein")

    colors = {"PARE": "#1f4e79", "GYRB": "#c45a1a"}

    numeric_cols = [
        "a_count_r3",
        "c_count_r3",
        "r1r2_removed",
        "survives_minus_r1r2",
        "rim_rsa_mean",
        "portal_geodesic_A",
        "via_hyd_geodesic_A",
        "net_charge_6A",
        "net_charge_8A",
        "potential_q_over_r_6A",
        "potential_q_over_r_8A",
        "acidic_count_6A",
        "acidic_count_8A",
        "basic_count_6A",
        "basic_count_8A",
        "rim_exposed_geo_mean_A",
        "via_hyd_node_radius_min_A",
        "via_hyd_edge_radius_min_A",
    ]
    for c in numeric_cols:
        matrix[c] = pd.to_numeric(matrix[c], errors="coerce")

    fig = plt.figure(figsize=(14.5, 9.2), dpi=220, facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)

    # A) Compact comparison table
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")

    pare = matrix.loc["PARE"]
    gyrb = matrix.loc["GYRB"]

    rows = [
        ("R3 class", pare["pulldown_region_r3"], gyrb["pulldown_region_r3"]),
        ("A/C counts", f"{int(pare['a_count_r3'])}/{int(pare['c_count_r3'])}", f"{int(gyrb['a_count_r3'])}/{int(gyrb['c_count_r3'])}"),
        ("R1/R2 removed", "yes" if int(pare["r1r2_removed"]) else "no", "yes" if int(gyrb["r1r2_removed"]) else "no"),
        (
            "In filtered proteome",
            "yes" if int(pare["survives_minus_r1r2"]) else "no",
            "yes" if int(gyrb["survives_minus_r1r2"]) else "no",
        ),
        ("Rim RSA mean", f"{pare['rim_rsa_mean']:.3f}", f"{gyrb['rim_rsa_mean']:.3f}"),
        ("Portal geodesic (A)", f"{pare['portal_geodesic_A']:.2f}", f"{gyrb['portal_geodesic_A']:.2f}"),
        ("Hyd-path geodesic (A)", f"{pare['via_hyd_geodesic_A']:.2f}", f"{gyrb['via_hyd_geodesic_A']:.2f}"),
        ("Net charge 6A", f"{pare['net_charge_6A']:.1f}", f"{gyrb['net_charge_6A']:.1f}"),
        ("Net charge 8A", f"{pare['net_charge_8A']:.1f}", f"{gyrb['net_charge_8A']:.1f}"),
        ("Potential proxy 6A", f"{pare['potential_q_over_r_6A']:.3f}", f"{gyrb['potential_q_over_r_6A']:.3f}"),
        ("Potential proxy 8A", f"{pare['potential_q_over_r_8A']:.3f}", f"{gyrb['potential_q_over_r_8A']:.3f}"),
    ]
    table_data = [[r[0], r[1], r[2]] for r in rows]

    tbl = ax_a.table(
        cellText=table_data,
        colLabels=["Metric", "ParE", "GyrB"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f3f6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f9fbfd")
        if c == 1:
            cell.set_text_props(color=colors["PARE"])
        if c == 2:
            cell.set_text_props(color=colors["GYRB"])
    ax_a.set_title("A. Side-by-Side Evidence Table", loc="left", fontsize=13, fontweight="bold")

    # B) Charged residue composition near DHRY
    ax_b = fig.add_subplot(gs[0, 1])
    labels = ["ParE 6A", "ParE 8A", "GyrB 6A", "GyrB 8A"]
    acidic = [
        float(pare["acidic_count_6A"]),
        float(pare["acidic_count_8A"]),
        float(gyrb["acidic_count_6A"]),
        float(gyrb["acidic_count_8A"]),
    ]
    basic = [
        float(pare["basic_count_6A"]),
        float(pare["basic_count_8A"]),
        float(gyrb["basic_count_6A"]),
        float(gyrb["basic_count_8A"]),
    ]
    x = np.arange(len(labels))
    ax_b.bar(x, acidic, width=0.64, color="#d74d4d", label="Acidic (D/E)")
    ax_b.bar(x, basic, width=0.64, bottom=acidic, color="#3f7db6", label="Basic (K/R/H)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=12, ha="right")
    ax_b.set_ylabel("Residue count")
    ax_b.set_title("B. Local Charge Composition (6-8 A)", loc="left", fontsize=13, fontweight="bold")
    ax_b.legend(frameon=False)
    ax_b.set_ylim(0, max(np.array(acidic) + np.array(basic)) + 1.6)

    # C) Direction-corrected effect sizes by metric (positive favors ParE)
    ax_c = fig.add_subplot(gs[1, 0])
    metric_spec = [
        ("rim_rsa_mean", "Rim RSA mean", "high"),
        ("rim_exposed_geo_mean_A", "Rim geodesic mean", "low"),
        ("portal_geodesic_A", "Portal geodesic", "low"),
        ("net_charge_6A", "Net charge 6A", "low"),
        ("potential_q_over_r_6A", "Potential 6A", "low"),
        ("net_charge_8A", "Net charge 8A", "low"),
        ("potential_q_over_r_8A", "Potential 8A", "low"),
        ("via_hyd_geodesic_A", "Hyd-path geodesic", "low"),
        ("via_hyd_node_radius_min_A", "Hyd-path min node radius", "high"),
        ("via_hyd_edge_radius_min_A", "Hyd-path min edge radius", "high"),
    ]

    metric_labels = []
    effects = []
    for key, label, direction in metric_spec:
        p = float(pare[key])
        g = float(gyrb[key])
        raw = (p - g) if direction == "high" else (g - p)
        scale = max(abs(p), abs(g), 1e-9)
        effects.append(raw / scale)
        metric_labels.append(label)

    y = np.arange(len(metric_labels))[::-1]
    effects_arr = np.array(effects)
    bar_colors = [colors["PARE"] if v >= 0 else colors["GYRB"] for v in effects_arr]
    ax_c.barh(y, effects_arr, color=bar_colors, alpha=0.9)
    ax_c.axvline(0, color="#333333", lw=1)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(metric_labels, fontsize=9.5)
    ax_c.set_xlim(-1.05, 1.05)
    ax_c.set_xlabel("Direction-corrected relative effect ( + favors ParE )")
    ax_c.set_title("C. Metric-Level Directional Effects", loc="left", fontsize=13, fontweight="bold")

    # D) Composite score + quick takeaway
    ax_d = fig.add_subplot(gs[1, 1])
    wins = [float(comp.loc["PARE", "wins"]), float(comp.loc["GYRB", "wins"])]
    bars = ax_d.bar(["ParE", "GyrB"], wins, color=[colors["PARE"], colors["GYRB"]], width=0.58)
    for b in bars:
        ax_d.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.25,
            f"{int(b.get_height())}/10",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax_d.set_ylim(0, 10.8)
    ax_d.set_ylabel("Directional metrics won")
    ax_d.set_title("D. Composite Evidence Score", loc="left", fontsize=13, fontweight="bold")
    ax_d.grid(axis="x", visible=False)

    summary = (
        f"Charge inversion near DHRY: ParE net charge is negative\n"
        f"(6A {pare['net_charge_6A']:.1f}, 8A {pare['net_charge_8A']:.1f}) while GyrB is positive\n"
        f"(6A {gyrb['net_charge_6A']:.1f}, 8A {gyrb['net_charge_8A']:.1f})."
    )
    ax_d.text(
        0.03,
        0.97,
        summary,
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=9.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f9fc", "edgecolor": "#d6dde6"},
    )

    fig.suptitle("ParE vs GyrB: DHRY Patch Comparison", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.01, 0.01, "Input: pare_vs_gyrb_evidence_matrix.tsv, metric_winners.tsv, evidence_composite.tsv", fontsize=8.5, color="#5b6570")

    FIGS.mkdir(parents=True, exist_ok=True)
    out_png = FIGS / "pare_vs_gyrb_evidence_summary.png"
    out_svg = FIGS / "pare_vs_gyrb_evidence_summary.svg"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()

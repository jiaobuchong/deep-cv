#!/usr/bin/env python3
"""Figure 4: holdout accuracy and peak training memory against trainable parameters, for every adaptation method
under the shared protocol (prototype head, 3 epochs, 20,000-image holdout, A100-80GB, bf16).

Writes figures/fig4_methods_params.png (slides) and report/figures/fig4_methods_params.pdf (report).
Numbers are copied from each run's summary.json (run names in the comments)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BLUE, INK, INK2, MUTED, GRID, BG = "#2A78D6", "#15202B", "#4A5563", "#8A8F98", "#E3E2DB", "#FCFCFB"

# (label, trainable backbone params in % of 85,798,656, holdout top-1 %, peak training memory GB)
TOP_K = [  # last K blocks trained, then the whole backbone
    ("K=1", 8.263, 67.97, 3.8),    # 1102_1942_1_partial_k1 (3.8 GB is the feature-extraction peak; training used less)
    ("K=2", 16.524, 70.99, 4.1),   # 1102_1942_2_partial_k2
    ("K=4", 33.046, 71.88, 7.0),   # 1102_1942_3_partial_k4
    ("K=6", 49.568, 72.89, 10.0),  # 1102_1942_4_partial_k6
    ("Full", 100.0, 73.38, 18.9),  # 1019_1120_1_full
]
SPREAD = [  # small changes in every block
    ("BitFit", 0.120, 69.81, 12.7),       # 1102_2023_1_bitfit
    ("LoRA", 1.375, 71.98, 19.9),         # 1019_1319_3_lora
    ("Adapter", 1.387, 71.54, 12.8),      # 1102_2017_1_adapter
    ("AdaptFormer", 1.387, 71.17, 12.8),  # 1102_2212_1_adaptformer
]
LINEAR_PROBE = 61.94  # 1102_1149_2_linear_lp: no backbone parameters, so it has no place on a log axis

# label offsets in points, chosen so that the three methods near 1.4% do not overlap; None = no label
ACC_OFFSETS = {"K=1": (6, -8), "K=2": (6, -8), "K=4": (6, -8), "K=6": (6, -8), "Full": (0, 9),
               "BitFit": (0, 9), "LoRA": (-8, 3), "Adapter": (8, 3), "AdaptFormer": (8, -7)}
MEM_OFFSETS = {"K=1": (-7, 0), "K=2": (0, -10), "K=4": (7, -2), "K=6": (7, -2), "Full": (-8, 6),
               "BitFit": (0, 9), "LoRA": (-8, 2), "Adapter": (8, 0), "AdaptFormer": None}
MEM_NAMES = {"Adapter": "Adapter, AdaptFormer"}


def style(ax):
    ax.set_facecolor(BG)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 170)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBDB3")
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_xlabel("trainable backbone parameters (log scale)", color=INK2, fontsize=9.5)


def label(ax, name, x, y, offsets, bold=False, names=None):
    if offsets[name] is None:
        return
    dx, dy = offsets[name]
    ha = "left" if dx > 0 else "right" if dx < -2 else "center"
    ax.annotate((names or {}).get(name, name), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=9,
                color=INK, fontweight="bold" if bold else "normal")


def panel(ax, col, offsets, title, names=None):
    style(ax)
    xs, ys = [p[1] for p in TOP_K], [p[col] for p in TOP_K]
    ax.plot(xs, ys, color=INK2, linewidth=1.8, marker="o", markersize=6, markeredgecolor=BG, markeredgewidth=1.2,
            label="partial → full fine-tuning (last K blocks)", zorder=3)
    for name, x, *vals in TOP_K:
        label(ax, name, x, vals[col - 2], offsets)
    for name, x, *vals in SPREAD:
        ax.plot([x], [vals[col - 2]], linestyle="none", marker="o", markersize=7 if name == "LoRA" else 5.5, color=BLUE,
                markeredgecolor=BG, markeredgewidth=1.2, zorder=4,
                label="BitFit, Adapter, AdaptFormer, LoRA (every block)" if name == "BitFit" else None)
        label(ax, name, x, vals[col - 2], offsets, bold=name == "LoRA", names=names)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)


def main():
    plt.rcParams["font.family"] = "Arial"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 2.75), dpi=200)
    fig.patch.set_facecolor(BG)

    panel(a, 2, ACC_OFFSETS, "Holdout top-1 (%)")
    a.set_ylim(66, 74.5)
    a.annotate(f"linear probe (no backbone parameters): {LINEAR_PROBE:.1f}%, below this range", (0.075, 66.25),
               ha="left", va="bottom", fontsize=8, color=MUTED)

    panel(b, 3, MEM_OFFSETS, "Peak training memory (GB)", names=MEM_NAMES)
    b.set_ylim(0, 22.5)

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01),
               labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.5)

    png = os.path.join(HERE, "fig4_methods_params.png")
    pdf = os.path.join(ROOT, "report", "figures", "fig4_methods_params.pdf")
    fig.savefig(png, facecolor=BG)
    fig.savefig(pdf, facecolor=BG)
    print("wrote", png, "and", pdf)


if __name__ == "__main__":
    main()

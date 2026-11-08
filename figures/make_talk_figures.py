#!/usr/bin/env python3
"""Two result charts for the slides, drawn at their size on the slide (points), so font sizes here are slide font sizes.

fig5_head_pairs.png: each pair changes only the head initialization (linear-probe vs prototype head).
fig6_ablations.png:  one change at a time, as the gain or loss in top-1 points.
fig7_acc_params.png: holdout top-1 against trainable parameters (one panel of fig4, sized for half a slide).
fig8_resolution.png: holdout top-1 against the test image size, for a model trained at 224 px and one trained at 288 px.
fig9_changes_why.png: every recipe change, its effect in top-1 points and the reason it helped or not.
fig10_loss_slide.png: training loss with the linear-probe and the prototype head (full fine-tuning, 8 epochs, all data).
Numbers are copied from the runs' summary.json files (see the report, Tables 5-7, and cs604_results_summary.md)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE, INK, INK2, MUTED, GRID = "#FAF9F6", "#15202B", "#4A5563", "#6B7280", "#E3E2DB"
BLUE, BLUE_DARK, ORANGE, GRAY = "#2A78D6", "#1F5FAD", "#EB6834", "#9AA1AB"
ORANGE_DARK = "#C2531F"
plt.rcParams["font.family"] = "Arial"

# Training loss every 100 steps (1,757 steps per epoch) of the two final-recipe runs that differ only in the head:
# 1018_2306_1_full (linear-probe head) and 1019_1319_1_full (prototype head), from their history.json files.
LOSS_LP = [
    2.781, 2.649, 2.588, 2.54, 2.521, 2.483, 2.453, 2.443, 2.43, 2.42, 2.408, 2.398, 2.403, 2.399, 2.403, 2.38, 2.388,
    2.299, 2.3, 2.273, 2.284, 2.294, 2.289, 2.269, 2.28, 2.298, 2.265, 2.263, 2.269, 2.255, 2.269, 2.258, 2.254, 2.274,
    2.253, 2.078, 2.096, 2.106, 2.109, 2.1, 2.094, 2.116, 2.109, 2.114, 2.113, 2.102, 2.113, 2.098, 2.104, 2.115, 2.11,
    2.108, 1.967, 1.967, 1.971, 1.973, 1.979, 1.985, 1.978, 1.979, 1.984, 1.976, 1.982, 1.993, 1.983, 1.987, 1.977,
    1.988, 1.984, 1.987, 1.884, 1.879, 1.888, 1.898, 1.892, 1.887, 1.891, 1.879, 1.895, 1.892, 1.896, 1.884, 1.897,
    1.89, 1.888, 1.891, 1.892, 1.829, 1.826, 1.815, 1.823, 1.817, 1.827, 1.817, 1.821, 1.813, 1.828, 1.82, 1.837, 1.824,
    1.827, 1.827, 1.816, 1.821, 1.812, 1.787, 1.773, 1.786, 1.781, 1.785, 1.782, 1.782, 1.784, 1.791, 1.777, 1.79,
    1.781, 1.779, 1.781, 1.787, 1.782, 1.782, 1.765, 1.765, 1.771, 1.761, 1.763, 1.768, 1.764, 1.769, 1.77, 1.772,
    1.764, 1.768, 1.761, 1.768, 1.767, 1.757, 1.765, 1.76]
LOSS_PROTO = [
    5.073, 4.361, 3.949, 3.686, 3.526, 3.406, 3.362, 3.294, 3.224, 3.18, 3.126, 3.102, 3.074, 3.024, 2.999, 2.996,
    2.962, 2.754, 2.758, 2.753, 2.743, 2.73, 2.732, 2.712, 2.701, 2.711, 2.695, 2.669, 2.664, 2.657, 2.66, 2.635, 2.63,
    2.645, 2.624, 2.373, 2.362, 2.387, 2.372, 2.37, 2.371, 2.382, 2.38, 2.381, 2.368, 2.372, 2.392, 2.385, 2.37, 2.362,
    2.371, 2.358, 2.185, 2.153, 2.171, 2.161, 2.177, 2.182, 2.172, 2.181, 2.18, 2.193, 2.18, 2.19, 2.18, 2.178, 2.18,
    2.171, 2.169, 2.166, 2.032, 2.021, 2.022, 2.021, 2.031, 2.015, 2.037, 2.035, 2.032, 2.038, 2.033, 2.042, 2.035,
    2.046, 2.042, 2.035, 2.036, 1.933, 1.932, 1.941, 1.932, 1.933, 1.944, 1.926, 1.925, 1.94, 1.938, 1.938, 1.933,
    1.949, 1.939, 1.937, 1.939, 1.939, 1.939, 1.873, 1.869, 1.871, 1.873, 1.869, 1.88, 1.866, 1.875, 1.876, 1.875,
    1.886, 1.87, 1.876, 1.872, 1.88, 1.875, 1.874, 1.933, 1.852, 1.847, 1.849, 1.849, 1.85, 1.852, 1.86, 1.849, 1.85,
    1.849, 1.841, 1.851, 1.842, 1.849, 1.852, 1.842, 1.852]


def canvas(w, h):
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.patch.set_facecolor(PAGE)
    return fig, ax


def text(ax, x, y, s, size=11, color=INK2, ha="left", va="center", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=5)


def head_pairs():
    """Dumbbell chart: linear-probe head (orange) to prototype head (blue) for four runs."""
    W, H = 832, 250
    rows = [("Full fine-tuning, 3 epochs", "holdout", 70.88, 73.38),
            ("LoRA on all linear layers, 3 epochs", "holdout", 69.45, 71.98),
            ("AdaptFormer, 3 epochs", "holdout", 69.05, 71.17),
            ("Full FT, 8 epochs, all data, flip TTA", "CodaBench val", 73.10, 76.22)]
    lo, hi, x0, x1 = 68.0, 77.0, 300.0, 748.0
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    fig, ax = canvas(W, H)
    top, step = 52, 44
    for v in range(68, 78, 2):                                     # grid and ticks
        ax.plot([sx(v), sx(v)], [top - 18, top + 3 * step + 18], color=GRID, linewidth=0.8, zorder=1)
        text(ax, sx(v), top + 3 * step + 32, f"{v}%", size=10.5, color=MUTED, ha="center")
    text(ax, x0, 14, "●", size=12, color=ORANGE)
    text(ax, x0 + 14, 14, "linear-probe head", size=11.5)
    text(ax, x0 + 140, 14, "●", size=12, color=BLUE)
    text(ax, x0 + 154, 14, "prototype head (ours)", size=11.5)
    text(ax, W - 6, 14, "gain", size=11.5, color=MUTED, ha="right")
    for i, (name, split, lp, proto) in enumerate(rows):
        y = top + i * step
        text(ax, 0, y - 7, name, size=12.5, color=INK, weight="bold" if i == 3 else "normal")
        text(ax, 0, y + 9, split, size=10, color=MUTED)
        ax.plot([sx(lp), sx(proto)], [y, y], color=GRAY, linewidth=2.2, solid_capstyle="round", zorder=2)
        for v, c in ((lp, ORANGE), (proto, BLUE)):
            ax.add_patch(Circle((sx(v), y), 6, facecolor=c, edgecolor=PAGE, linewidth=1.5, zorder=3))
        text(ax, sx(lp) - 11, y, f"{lp:.2f}", size=10.5, color=INK2, ha="right")
        text(ax, sx(proto) + 11, y, f"{proto:.2f}", size=10.5, color=INK2)
        text(ax, W - 6, y, f"+{proto - lp:.2f}", size=13, color=BLUE_DARK, ha="right", weight="bold")
    path = os.path.join(HERE, "fig5_head_pairs.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


def ablations():
    """Diverging bars: the change in top-1 points from one setting at a time, in three groups."""
    W, H = 832, 270
    groups = [("LoRA capacity (linear-probe head, 3 epochs, holdout)", [
                  ("learning rate 1e-3 → 5e-4 (qkv only)", 68.57, 68.45),
                  ("qkv only → all four linear layers", 68.57, 69.45)]),
              ("Training length (prototype head)", [
                  ("LoRA: 3 → 6 epochs (holdout)", 71.98, 71.52),
                  ("full FT, all data, TTA: 3 → 8 epochs (val)", 74.01, 76.22)]),
              ("Final-model steps (full FT, 8 epochs, CodaBench val)", [
                  ("+ flip TTA", 72.57, 72.84),
                  ("43 → 45 images per class", 72.84, 73.10)])]
    lo, hi, x0, x1 = -1.0, 2.5, 380.0, 690.0
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    fig, ax = canvas(W, H)
    y, rows_y = 12, []
    for title, items in groups:
        text(ax, 0, y, title, size=11.5, color=INK, weight="bold")
        y += 20
        for label, before, after in items:
            rows_y.append((y, label, before, after))
            y += 22
        y += 8
    y_end = y - 8
    for v in (-1, 0, 1, 2):                                        # grid and ticks
        ax.plot([sx(v), sx(v)], [4, y_end - 4], color=INK2 if v == 0 else GRID, linewidth=1.0 if v == 0 else 0.8, zorder=1)
        text(ax, sx(v), y_end + 10, f"{v:+d}".replace("-", "−") if v else "0", size=10.5, color=MUTED, ha="center")
    text(ax, (x0 + x1) / 2, y_end + 24, "change in top-1 (points)", size=10.5, color=MUTED, ha="center")
    text(ax, W - 4, 12, "before → after", size=10.5, color=MUTED, ha="right")
    for y, label, before, after in rows_y:
        d = round(after - before, 2)
        text(ax, 14, y, label, size=11.5)
        left, right = sorted((sx(0), sx(d)))
        ax.add_patch(FancyBboxPatch((left, y - 6), right - left, 12, boxstyle="round,pad=0,rounding_size=2",
                                    facecolor=BLUE if d > 0 else GRAY, edgecolor="none", zorder=3))
        text(ax, sx(d) + (6 if d > 0 else -6), y, f"{d:+.2f}".replace("-", "−"), size=11.5,
             color=BLUE_DARK if d > 0 else INK2, ha="left" if d > 0 else "right", weight="bold")
        text(ax, W - 4, y, f"{before:.2f} → {after:.2f}", size=10.5, color=MUTED, ha="right")
    path = os.path.join(HERE, "fig6_ablations.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


def acc_params():
    """One panel: holdout top-1 against the share of trained backbone weights (log axis), same data as fig4."""
    top_k = [("K=1", 8.263, 67.97), ("K=2", 16.524, 70.99), ("K=4", 33.046, 71.88), ("K=6", 49.568, 72.89),
             ("Full", 100.0, 73.38)]
    spread = [("BitFit", 0.120, 69.81), ("LoRA", 1.375, 71.98), ("Adapter", 1.387, 71.54), ("AdaptFormer", 1.387, 71.17)]
    offsets = {"K=1": (7, -9), "K=2": (7, -9), "K=4": (7, -9), "K=6": (7, -9), "Full": (0, 11),
               "BitFit": (0, 11), "LoRA": (-9, 4), "Adapter": (9, 4), "AdaptFormer": (9, -8)}
    w, h = 392, 286
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=300)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.78])
    ax.set_facecolor(PAGE)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 190)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_ylim(66, 74.6)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBDB3")
    ax.tick_params(colors=INK2, length=0, labelsize=10.5)
    ax.set_xlabel("trainable backbone weights (log scale)", color=INK2, fontsize=11)
    ax.set_ylabel("holdout top-1 (%)", color=INK2, fontsize=11)
    ax.plot([p[1] for p in top_k], [p[2] for p in top_k], color=GRAY, linewidth=2, marker="o", markersize=6.5,
            markeredgecolor=PAGE, markeredgewidth=1.2, zorder=3)
    for name, x, y in top_k + spread:
        if name in dict((n, 0) for n, _, _ in spread):
            ax.plot([x], [y], linestyle="none", marker="o", markersize=8 if name == "LoRA" else 6.5, color=BLUE,
                    markeredgecolor=PAGE, markeredgewidth=1.2, zorder=4)
        dx, dy = offsets[name]
        ha = "left" if dx > 0 else "right" if dx < -2 else "center"
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=10.5,
                    color=INK, fontweight="bold" if name in ("LoRA", "Full") else "normal")
    ax.annotate("last K blocks,\nfrom the top", (14, 68.6), ha="left", va="center", fontsize=10.5, color=INK2,
                linespacing=1.15)
    ax.annotate("small changes in\nevery block", (0.085, 73.6), ha="left", va="center", fontsize=10.5, color=BLUE_DARK,
                linespacing=1.15)
    ax.annotate("linear probe: 61.94%, below this range", (0.075, 66.3), ha="left", va="bottom", fontsize=9.5, color=MUTED)
    path = os.path.join(HERE, "fig7_acc_params.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


def resolution():
    """Holdout top-1 against the test size: 1019_1120_1_full (trained at 224 px, from 08) and 1105_1031_1_full_288 (09)."""
    runs = [("trained at 224 px", GRAY, INK2, {224: 73.375, 256: 75.21, 288: 75.505, 320: 75.28, 352: 74.33, 384: 72.94}),
            ("trained at 288 px", BLUE, BLUE_DARK, {224: 73.10, 256: 75.745, 288: 77.235, 320: 77.775, 352: 77.63,
                                                    384: 77.41, 416: 77.035, 448: 76.105})]
    w, h = 520, 262
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=300)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes([0.1, 0.17, 0.86, 0.79])
    ax.set_facecolor(PAGE)
    ax.set_xlim(208, 470)
    ax.set_ylim(72.2, 78.8)
    ax.set_xticks([224, 256, 288, 320, 352, 384, 416, 448])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBDB3")
    ax.tick_params(colors=INK2, length=0, labelsize=10.5)
    ax.set_xlabel("test image size (px)", color=INK2, fontsize=11)
    ax.set_ylabel("holdout top-1 (%)", color=INK2, fontsize=11)
    for name, color, text_color, acc in runs:
        xs = sorted(acc)
        ax.plot(xs, [acc[x] for x in xs], color=color, linewidth=2.2, marker="o", markersize=6.5, markeredgecolor=PAGE,
                markeredgewidth=1.2, zorder=3)
        best = max(xs, key=lambda x: acc[x])
        ax.plot([best], [acc[best]], marker="o", markersize=10, color=color, markeredgecolor=PAGE, markeredgewidth=1.5,
                zorder=4)
        above = color == BLUE                                      # the 224 px label goes below its point, clear of the blue line
        ax.annotate(f"{acc[best]:.2f}% at {best} px", (best, acc[best]), xytext=(0, 12 if above else -12),
                    textcoords="offset points", ha="center", va="bottom" if above else "top", fontsize=10.5,
                    color=text_color, fontweight="bold")
        ax.annotate(name, (xs[-1], acc[xs[-1]]), xytext=(4, -14), textcoords="offset points", ha="right", va="center",
                    fontsize=10.5, color=text_color)
    path = os.path.join(HERE, "fig8_resolution.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


def changes_why():
    """Every recipe change, one at a time: the gain or loss in top-1 points, the numbers, and why it helped or not."""
    W, H = 832, 352
    groups = [("The head (Part 3)", [
                  ("linear-probe → prototype head", "full FT, 8 epochs, val", 73.10, 76.22,
                   "the head no longer memorizes the training photos")]),
              ("LoRA (holdout, 3 epochs)", [
                  ("learning rate 1e-3 → 5e-4", "qkv only, linear-probe head", 68.57, 68.45,
                   "within noise: the learning rate was not the problem"),
                  ("qkv only → all four linear layers", "linear-probe head", 68.57, 69.45,
                   "the MLP holds 2/3 of each block's weights"),
                  ("3 → 6 epochs", "prototype head", 71.98, 71.52, "train loss 1.95 → 1.70, yet holdout falls")]),
              ("Full fine-tuning (CodaBench val)", [
                  ("3 → 8 epochs", "prototype head, all data, TTA", 74.01, 76.22,
                   "still improving at epoch 3; 8 epochs fit the budget"),
                  ("+ flip TTA", "linear-probe head, 8 epochs", 72.57, 72.84, "averages each photo with its mirror image"),
                  ("43 → 45 photos per class", "linear-probe head, 8 epochs", 72.84, 73.10,
                   "the 2 holdout photos per class also train")])]
    lo, hi, x0, x1 = -1.0, 3.5, 318.0, 532.0
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    why_x = 560
    fig, ax = canvas(W, H)
    text(ax, why_x, 10, "why", size=11, color=MUTED, weight="bold")
    y, rows = 10, []
    for title, items in groups:
        text(ax, 0, y, title, size=12, color=INK, weight="bold")
        y += 23
        for item in items:
            rows.append((y,) + item)
            y += 34
        y += 4
    y_end = y - 14
    for v in (-1, 0, 1, 2, 3):                                     # grid and ticks
        ax.plot([sx(v), sx(v)], [22, y_end], color=INK2 if v == 0 else GRID, linewidth=1.0 if v == 0 else 0.8, zorder=1)
        text(ax, sx(v), y_end + 11, f"{v:+d}".replace("-", "−") if v else "0", size=10.5, color=MUTED, ha="center")
    text(ax, (x0 + x1) / 2, y_end + 26, "change in top-1 (points)", size=10.5, color=MUTED, ha="center")
    for y, label, setting, before, after, why in rows:
        d = round(after - before, 2)
        text(ax, 12, y - 7, label, size=12, color=INK)
        text(ax, 12, y + 9, f"{setting}: {before:.2f} → {after:.2f}", size=9.5, color=MUTED)
        left, right = sorted((sx(0), sx(d)))
        ax.add_patch(FancyBboxPatch((left, y - 7), right - left, 14, boxstyle="round,pad=0,rounding_size=2",
                                    facecolor=BLUE if d > 0 else GRAY, edgecolor="none", zorder=3))
        text(ax, sx(d) + (6 if d > 0 else -6), y, f"{d:+.2f}".replace("-", "−"), size=11.5,
             color=BLUE_DARK if d > 0 else INK2, ha="left" if d > 0 else "right", weight="bold")
        text(ax, why_x, y, why, size=11.5, color=INK2)
    path = os.path.join(HERE, "fig9_changes_why.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


def loss_curves():
    """Training loss of full fine-tuning on all data for 8 epochs, linear-probe vs prototype head, sized for the slide."""
    w, h = 640, 172
    fig = plt.figure(figsize=(w / 72, h / 72), dpi=300)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes([0.075, 0.2, 0.6, 0.76])
    ax.set_facecolor(PAGE)
    epochs = [100 * (i + 1) / 1757 for i in range(len(LOSS_LP))]
    for loss, color in ((LOSS_LP, ORANGE), (LOSS_PROTO, BLUE)):
        ax.plot(epochs, loss, color=color, linewidth=1.8, solid_capstyle="round")
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 5.2)
    ax.set_xticks(range(9))
    ax.set_yticks([2, 3, 4, 5])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BFBDB3")
    ax.tick_params(colors=INK2, length=0, labelsize=10.5)
    ax.set_xlabel("epoch", color=INK2, fontsize=11)
    ax.set_ylabel("training loss", color=INK2, fontsize=11)
    # the message sits to the right of the plot: lower training loss, yet lower accuracy
    for y, color, name, end, val in ((0.66, BLUE_DARK, "prototype head", "1.85", "76.22%"),
                                    (0.30, ORANGE_DARK, "linear-probe head", "1.77", "73.10%")):
        fig.text(0.7, y + 0.1, name, color=color, fontsize=12, fontweight="bold", va="center")
        fig.text(0.7, y - 0.03, f"final loss {end}, val {val}", color=INK2, fontsize=11.5, va="center")
    path = os.path.join(HERE, "fig10_loss_slide.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    head_pairs()
    ablations()
    acc_params()
    resolution()
    changes_why()
    loss_curves()

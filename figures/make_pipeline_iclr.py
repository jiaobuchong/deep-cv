#!/usr/bin/env python3
"""Pipeline figure for the ICLR report: our changes (blue, "OURS") next to the usual approach (orange).

Writes report_iclr/figures/fig_pipeline.pdf and .png. Coordinates are in 0.1-inch units."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_iclr", "figures", "fig_pipeline")

W, H = 60, 32                                   # 6.0 x 3.2 inches
GREY = dict(fc="#F3F2EE", ec="#A9A79F")
BLUE = dict(fc="#E7F0FB", ec="#2B6CB0")
ORANGE = dict(fc="#FDEDE3", ec="#DD6B20")
INK, BLUE_INK, ORANGE_INK, MUTED = "#1F2328", "#1F5A99", "#B4531A", "#5B5F66"

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "mathtext.fontset": "dejavusans"})
fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y0, x1, y1, style, lw=0.8, r=0.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                lw=lw, **style))


def text(x, y, s, size=7, color=INK, ha="center", va="center", weight="normal", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, linespacing=1.25, **kw)


def arrow(p, q, color=MUTED, dashed=False, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0))


def badge(x, y, label, color):
    ax.text(x, y, label, fontsize=5.6, color="white", weight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=color, ec="none"))


# ---------------------------------------------------------------- step 1: start the head
text(0.3, 31.2, "1  Start the new 10,000-class head (once, before training; counted in the budget)",
     size=7.6, ha="left", weight="bold")
box(0.3, 20.6, 7.3, 27.0, GREY)
text(3.8, 23.8, "450,000\ntraining\nimages")
arrow((7.3, 23.8), (8.8, 23.8))
box(8.8, 20.6, 15.8, 27.0, GREY)
text(12.3, 23.8, "frozen\nViT-B/16\n(6 min)")
arrow((15.8, 23.8), (17.3, 23.8))
box(17.3, 20.6, 24.6, 27.0, GREY)
text(20.95, 23.8, "feature $z_i$\nper image\n(768-d)")

# usual: linear probe (LP-FT)
arrow((24.6, 25.4), (26.4, 27.0), color=ORANGE_INK)
box(26.4, 24.4, 59.7, 30.4, ORANGE)
badge(27.0, 29.4, "USUAL (LP-FT)", ORANGE_INK)
text(27.0, 26.5, "Train a linear probe on $z_i$. It fits the same images that are\n"
                 "later used for fine-tuning, so the backbone learns to memorize them.\n"
                 "Starts at 62.1%, but ends 2.1\u20133.1 points lower after fine-tuning.",
     ha="left", size=6.6)
# ours: prototype head
arrow((24.6, 22.2), (26.4, 20.8), color=BLUE_INK)
box(26.4, 17.2, 59.7, 23.6, BLUE, lw=1.3)
badge(27.0, 22.6, "OURS: prototype head", BLUE_INK)
text(27.0, 19.6, r"$W_c = \tau\,\mu_c/\overline{\|z\|}$, where $\mu_c$ = normalized mean feature of class $c$, $\tau = 40$." "\n"
                 "No gradient steps on the training images, so it memorizes nothing.\n"
                 "Starts at 44.0%, ends best: +3.12 points on validation.",
     ha="left", size=6.6)

# ---------------------------------------------------------------- step 2: fine-tune
text(0.3, 15.6, "2  Fine-tune within 1.2 GPU-hours", size=7.6, ha="left", weight="bold")
box(0.3, 4.6, 8.8, 11.4, GREY)
text(4.55, 8.0, "image\n224 \u00d7 224\n(crop + flip)")
arrow((8.8, 8.0), (10.6, 8.0))

box(10.6, 0.4, 41.4, 14.4, GREY)
text(26.0, 13.4, "ViT-B/16 encoder: 12 identical blocks", weight="bold")
# attention group
box(11.4, 7.6, 25.4, 12.0, dict(fc="white", ec="#C9C7BF"), lw=0.6, r=0.4)
text(18.4, 11.2, "attention", size=6.2, color=MUTED)
for x0, lab, ours in [(12.0, "qkv", True), (16.5, "attn", False), (21.0, "proj", True)]:
    box(x0, 8.1, x0 + 3.8, 10.3, BLUE if ours else GREY, lw=1.1 if ours else 0.6, r=0.3)
    text(x0 + 1.9, 9.2, lab, size=6.6, weight="bold" if ours else "normal")
# MLP group
box(26.6, 7.6, 40.6, 12.0, dict(fc="white", ec="#C9C7BF"), lw=0.6, r=0.4)
text(33.6, 11.2, "MLP (2/3 of the block's weights)", size=6.2, color=MUTED)
for x0, lab, ours in [(27.2, "fc1", True), (31.7, "GELU", False), (36.2, "fc2", True)]:
    box(x0, 8.1, x0 + 3.8, 10.3, BLUE if ours else GREY, lw=1.1 if ours else 0.6, r=0.3)
    text(x0 + 1.9, 9.2, lab, size=6.6, weight="bold" if ours else "normal")
arrow((25.4, 9.8), (26.6, 9.8), lw=0.7)

text(11.4, 6.0, "Full fine-tuning (final model): every weight is trained, 8 epochs.", ha="left", size=6.4)
badge(11.4, 3.9, "OURS", BLUE_INK)
text(14.6, 3.9, "LoRA r = 8 on all four linear layers (blue), not just qkv:", ha="left", size=6.4,
     color=BLUE_INK)
text(14.6, 2.2, "+0.88 points; merged into the weights after training.", ha="left", size=6.4,
     color=BLUE_INK)

# head
arrow((41.4, 8.0), (43.2, 8.0))
box(43.2, 4.6, 50.6, 11.4, BLUE, lw=1.3)
text(46.9, 8.0, "head\n10,000 × 768\n(from step 1)")
arrow((46.9, 17.2), (46.9, 11.4), color=BLUE_INK, dashed=True)

# ---------------------------------------------------------------- step 3: predict
text(52.2, 15.6, "3  Predict", size=7.6, ha="left", weight="bold")
arrow((50.6, 8.0), (52.2, 8.0))
box(52.2, 2.6, 59.7, 14.0, GREY)
text(55.95, 9.4, "logits($x$)\n+ logits(flip $x$)", size=6.6)
text(55.95, 5.3, "full FT: +0.27,\n2× latency\nLoRA: no extra\nlatency", size=5.9, color=MUTED)

fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".png", dpi=250)
print("wrote", OUT + ".pdf/.png")

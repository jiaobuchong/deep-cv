#!/usr/bin/env python3
"""Architecture diagrams for the seven method slides, drawn in the style of the classic AlexNet figure: tensors as
3D boxes with their sizes, layers as boxes with their weight shapes and parameter counts. Blue = trained, grey = frozen.

Each diagram is 832 x 190 pt, the exact size it takes on the slide, so font sizes here are the font sizes on the slide.
Writes figures/diagrams/method_<name>.png (used by slides/build_pptx.py)."""
import colorsys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diagrams")
W, H = 832, 190

PAGE, INK, INK2, MUTED = "#FAF9F6", "#15202B", "#4A5563", "#6B7280"
BLUE, BLUE_DARK, BLUE_LIGHT = "#2A78D6", "#1F5FAD", "#D6E6FA"
FROZEN, FROZEN_EDGE, LINE, WHITE = "#E6E2DA", "#B9B4AA", "#9AA1AB", "#FFFFFF"
K_SHADES = {1: "#1F5FAD", 2: "#2A78D6", 4: "#7FB2F0", 6: "#BFD7F5"}   # block trained when K >= 1, 2, 4 or 6

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


# ---------------------------------------------------------------- drawing helpers (slide coordinates: points, y down)

def canvas():
    fig = plt.figure(figsize=(W / 72, H / 72), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(PAGE)
    return fig, ax


def shade(color, amount):
    """Lighten (amount > 0) or darken (amount < 0) a color, keeping its hue."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
    l = l + (1 - l) * amount if amount > 0 else l * (1 + amount)
    return to_hex(colorsys.hls_to_rgb(h, l, s))


def text(ax, x, y, s, size=9, color=INK2, ha="center", va="center", weight="normal", rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, rotation=rotation, zorder=6)


def tensor(ax, x, y, w, h, d, fill=WHITE, edge="#8A8F98"):
    """A tensor as a 3D box: front face at (x, y) of size w x h, depth d drawn up and to the right."""
    faces = [([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], 0.45),               # top
             ([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], -0.10),  # side
             ([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], 0.0)]                         # front
    for pts, amount in faces:
        ax.add_patch(Polygon(pts, closed=True, facecolor=shade(fill, amount), edgecolor=edge, linewidth=0.7,
                             joinstyle="round", zorder=3))


def layer(ax, x, y, w, h, lines, state):
    """A layer as a rounded box with up to three lines: name, shape, parameters. state: train, frozen or op."""
    fill, edge, colors = {"train": (BLUE, BLUE_DARK, (WHITE, "#E8F1FC", "#E8F1FC")),
                          "frozen": (FROZEN, FROZEN_EDGE, (INK, INK2, MUTED)),
                          "op": (WHITE, FROZEN_EDGE, (INK2, MUTED, MUTED))}[state]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", facecolor=fill, edgecolor=edge,
                                linewidth=0.9, zorder=4))
    sizes = [10.5, 8.5, 8.5]
    heights = [s * 1.25 for s in sizes[:len(lines)]]
    top = y + (h - sum(heights)) / 2
    for i, line in enumerate(lines):
        text(ax, x + w / 2, top + heights[i] / 2, line, size=sizes[i], color=colors[i],
             weight="bold" if i == 0 else "normal")
        top += heights[i]


def adder(ax, cx, cy, r=8):
    ax.add_patch(Circle((cx, cy), r, facecolor=WHITE, edgecolor=LINE, linewidth=0.9, zorder=4))
    text(ax, cx, cy - 0.4, "+", size=10, color=INK2, weight="bold")


def line(ax, pts, color=LINE, lw=1.0, dashed=False, arrow=False):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=(0, (3, 2)) if dashed else "-", solid_capstyle="butt", zorder=2)
    if arrow:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7),
                    zorder=2)


def legend(ax, x, y, items):
    for kind, label in items:
        if kind == "tensor":
            tensor(ax, x, y - 3, 9, 9, 3)
        else:
            layer(ax, x, y - 4.5, 12, 9, [], kind)
        text(ax, x + 18, y, label, size=9, color=INK2, ha="left")
        x += 18 + len(label) * 4.9 + 14


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"method_{name}.png")
    fig.savefig(path, facecolor=PAGE)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- the whole network (linear probe, partial, full)

def network(mode):
    """Image -> patch embedding -> 197 tokens -> 12 blocks -> norm -> [CLS] -> head -> 10,000 logits."""
    fig, ax = canvas()
    yc = 86
    state = lambda part: "train" if (mode == "full" or part == "head" or (mode == "partial" and part == "norm")) else "frozen"

    tensor(ax, 26, yc - 46, 5, 92, 30)                                     # image, 224 x 224 x 3
    text(ax, 18, yc + 6, "224", rotation=90)
    text(ax, 68, yc - 44, "224", rotation=45)
    text(ax, 28, yc + 54, "3")
    text(ax, 40, yc + 70, "image", weight="bold", color=INK)
    line(ax, [(64, yc), (80, yc)], arrow=True)
    layer(ax, 82, yc - 26, 88, 52, ["Patch embed", "16×16 conv, stride 16", "0.59M"], state("patch"))
    line(ax, [(170, yc), (186, yc)], arrow=True)
    tensor(ax, 188, yc - 14, 44, 28, 12)                                   # 14 x 14 x 768
    text(ax, 210, yc + 24, "768")
    text(ax, 195, yc, "14", rotation=90)
    text(ax, 252, yc - 22, "14", rotation=45)
    text(ax, 216, yc + 70, "14 × 14 patches", weight="bold", color=INK)
    line(ax, [(246, yc), (270, yc)], arrow=True)
    text(ax, 258, yc + 9, "flatten", size=8.5)
    tensor(ax, 272, yc - 40, 44, 80, 4)                                    # 197 x 768 tokens
    text(ax, 281, yc, "197", rotation=90)
    text(ax, 294, yc + 50, "768")
    text(ax, 296, yc + 70, "197 tokens", weight="bold", color=INK)
    text(ax, 296, yc + 82, "+ [CLS], + position: 0.15M", size=8.5, color=BLUE if mode == "full" else MUTED)
    line(ax, [(322, yc), (338, yc)], arrow=True)

    x0, bw, gap = 340, 14, 4                                               # 12 Transformer blocks
    for i in range(12):
        x = x0 + i * (bw + gap)
        need = 12 - i                                                      # block i + 1 is trained when K >= need
        if mode == "partial" and need <= 6:
            k = next(k for k in (1, 2, 4, 6) if k >= need)
            ax.add_patch(FancyBboxPatch((x, yc - 40), bw, 80, boxstyle="round,pad=0,rounding_size=3",
                                        facecolor=K_SHADES[k], edgecolor=BLUE_DARK, linewidth=0.8, zorder=4))
        else:
            layer(ax, x, yc - 40, bw, 80, [], "train" if mode == "full" else "frozen")
        text(ax, x + bw / 2, yc - 47, str(i + 1), size=8.5, color=INK2)
    x_end = x0 + 12 * bw + 11 * gap
    text(ax, (x0 + x_end) / 2, yc - 62, "12 Transformer blocks, 7.09M each", size=9.5, color=INK, weight="bold")
    text(ax, (x0 + x_end) / 2, yc + 50, "each keeps 197 × 768", size=8.5)
    if mode == "partial":
        for j, k in enumerate((1, 2, 4, 6)):
            by = yc + 60 + j * 10
            bx = x0 + (12 - k) * (bw + gap)
            ax.add_patch(FancyBboxPatch((bx, by), x_end - bx, 3.5, boxstyle="round,pad=0,rounding_size=1.5",
                                        facecolor=K_SHADES[k], edgecolor="none", zorder=4))
            text(ax, bx - 5, by + 1.5, f"K = {k}", size=8.5, ha="right")
    line(ax, [(x_end + 2, yc), (x_end + 16, yc)], arrow=True)
    xn = x_end + 18
    layer(ax, xn, yc - 26, 44, 52, ["Norm", "768", "1,536"], state("norm"))
    line(ax, [(xn + 44, yc), (xn + 62, yc)], arrow=True)
    xc = xn + 64
    tensor(ax, xc, yc - 30, 6, 60, 4)                                     # [CLS] feature, 768
    text(ax, xc + 5, yc + 40, "768")
    text(ax, xc + 5, yc + 70, "[CLS]", weight="bold", color=INK)
    line(ax, [(xc + 12, yc), (xc + 26, yc)], arrow=True)
    xh = xc + 28
    layer(ax, xh, yc - 26, 82, 52, ["Head", "768 → 10,000", "7.69M"], "train")
    line(ax, [(xh + 82, yc), (xh + 96, yc)], arrow=True)
    xl = xh + 98
    tensor(ax, xl, yc - 46, 6, 92, 4)                                     # logits, 10,000
    text(ax, xl + 5, yc + 56, "10,000")
    text(ax, xl + 5, yc + 70, "logits", weight="bold", color=INK)

    legend(ax, 470 if mode != "partial" else 586, 180,
           [("train", "trained"), ("frozen", "frozen"), ("tensor", "activations")])
    return fig


# ---------------------------------------------------------------- inside one block (BitFit, Adapter, AdaptFormer, LoRA)

LINEAR = {"qkv": ("768 → 2304", "1.77M", "2304", ("A 8×768", "B 2304×8", "24,576")),
          "proj": ("768 → 768", "0.59M", "768", ("A 8×768", "B 768×8", "12,288")),
          "fc1": ("768 → 3072", "2.36M", "3072", ("A 8×768", "B 3072×8", "30,720")),
          "fc2": ("3072 → 768", "2.36M", "768", ("A 8×3072", "B 768×8", "30,720"))}


def block(method):
    fig, ax = canvas()
    yc, bh, gap = 82, 46, 9
    text(ax, 16, 9, "Inside one Transformer block (all 12 are the same). Every token tensor is 197 × 768.",
         size=9.5, color=INK, ha="left", weight="bold")

    seq = [("x", "tensor", 12), ("LN", "norm", 32), ("qkv", "lin", 56), ("Attention", "op", 72), ("proj", "lin", 56),
           ("+", "add", 16), ("LN", "norm", 32), ("fc1", "lin", 56), ("h", "hidden", 30), ("GELU", "op", 46),
           ("fc2", "lin", 56)]
    if method == "adapter":
        seq += [("down", "adown", 58), ("z", "bottleneck", 6), ("up", "aup", 58), ("s+", "sadd", 12)]
    seq += [("+", "add", 16), ("y", "tensor", 12)]
    pos, x = {}, 16
    for i, (name, kind, w) in enumerate(seq):
        pos.setdefault(name, []).append((x, w))
        x += w + (6 if kind in ("tensor", "hidden", "bottleneck") else 0) + gap
    end = x - gap

    line(ax, [(16, yc), (end, yc)])                                         # the main path
    (a1x, a1w), (a2x, a2w) = pos["+"]
    split1, split2 = 16 + 18 + gap / 2, a1x + a1w + gap / 2
    ry = yc - 44
    for sx, ax_ in ((split1, a1x + a1w / 2), (split2, a2x + a2w / 2)):   # residual connections
        line(ax, [(sx, yc), (sx, ry), (ax_, ry), (ax_, yc - 8)], arrow=True)
    text(ax, (split1 + a1x) / 2, ry - 6, "residual", size=8.5)
    text(ax, (split2 + a2x) / 2, ry - 6, "residual", size=8.5)

    for name, kind, w in seq:
        x, _ = pos[name].pop(0) if len(pos[name]) > 1 else pos[name][0]
        if kind == "tensor":
            tensor(ax, x, yc - 26, w, 52, 6)
            if name == "x":
                text(ax, x + 6, yc + 36, "input")
            else:
                text(ax, x + 6, yc + 36, "output")
        elif kind == "hidden":
            tensor(ax, x, yc - 26, w, 52, 6)
            text(ax, x + w / 2 + 3, yc + 36, "197 × 3072")
        elif kind == "norm":
            layer(ax, x, yc - bh / 2, w, bh, ["LN", "768", "1,536"], "frozen")
            if method == "bitfit":
                bias_tag(ax, x + w / 2, yc + bh / 2, "β 768")
        elif kind == "lin":
            shape, count, bias, lora = LINEAR[name]
            layer(ax, x, yc - bh / 2, w, bh, [name, shape, count], "frozen")
            if method == "bitfit":
                bias_tag(ax, x + w / 2, yc + bh / 2, f"b {bias}")
            if method == "lora":
                cx = x + w / 2
                line(ax, [(cx, yc + bh / 2), (cx, yc + bh / 2 + 9)], color=BLUE)
                ax.add_patch(FancyBboxPatch((x, yc + bh / 2 + 9), w, 26, boxstyle="round,pad=0,rounding_size=3",
                                            facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=0.9, zorder=4))
                text(ax, cx, yc + bh / 2 + 16, lora[0], size=8.5, color=WHITE)
                text(ax, cx, yc + bh / 2 + 27, lora[1], size=8.5, color=WHITE)
                text(ax, cx, yc + bh / 2 + 43, lora[2], size=8.5, color=BLUE_DARK, weight="bold")
        elif kind == "op":
            layer(ax, x, yc - bh / 2, w, bh, [name, "12 heads × 64", "no weights"] if name == "Attention"
                  else [name, "no weights"], "op")
        elif kind == "add":
            adder(ax, x + w / 2, yc)
        elif kind == "adown":
            layer(ax, x, yc - bh / 2, w, bh, ["down", "768 → 64", "49,216"], "train")
        elif kind == "bottleneck":
            tensor(ax, x, yc - 26, w, 52, 6, fill=BLUE_LIGHT, edge=BLUE)
            text(ax, x + 5, yc + 36, "197 × 64", color=BLUE_DARK)
        elif kind == "aup":
            layer(ax, x, yc - bh / 2, w, bh, ["up", "64 → 768", "49,920"], "train")
        elif kind == "sadd":
            adder(ax, x + w / 2, yc, r=6)

    if method == "adapter":                                                 # the adapter's own skip connection
        (dx, _), = pos["down"]
        (sx_, sw), = pos["s+"]
        sy = yc - 32
        line(ax, [(dx - gap / 2, yc), (dx - gap / 2, sy), (sx_ + sw / 2, sy), (sx_ + sw / 2, yc - 6)], color=BLUE,
             arrow=True)
        text(ax, (dx + sx_ + sw) / 2, sy - 6, "Adapter, in series: h + up(ReLU(down(h)))", size=8.5, color=BLUE_DARK)
    if method == "adaptformer":                                             # side branch from the block input
        by = yc + 62
        fx, _ = pos["fc1"][0]
        bx = fx - 4
        line(ax, [(split2, yc), (split2, by), (bx, by)], color=BLUE)
        layer(ax, bx, by - 15, 56, 30, ["down", "768 → 64"], "train")
        tensor(ax, bx + 66, by - 15, 5, 30, 5, fill=BLUE_LIGHT, edge=BLUE)
        text(ax, bx + 70, by + 23, "197 × 64", color=BLUE_DARK, size=8.5)
        line(ax, [(bx + 56, by), (bx + 66, by)], color=BLUE)
        line(ax, [(bx + 76, by), (bx + 84, by)], color=BLUE)
        layer(ax, bx + 84, by - 15, 56, 30, ["up", "64 → 768"], "train")
        line(ax, [(bx + 140, by), (bx + 148, by)], color=BLUE)
        layer(ax, bx + 148, by - 11, 30, 22, ["× s"], "train")
        ax_ = a2x + a2w / 2
        line(ax, [(bx + 178, by), (ax_, by), (ax_, yc + 8)], color=BLUE, arrow=True)
        text(ax, bx + 200, by - 8, "s = 0.1", size=8.5, color=BLUE_DARK, ha="left")
        text(ax, bx - 6, by + 20, "Adapter beside the MLP: 99,136 per block", size=8.5, color=BLUE_DARK, ha="right")

    notes = {"bitfit": "Trained: 8,448 biases per block, plus 1,536 in the patch embedding and final norm.",
             "adapter": "Trained: down + up = 99,136 per block.",
             "adaptformer": "",
             "lora": "Trained: ΔW = BA next to each frozen W, rank 8: 98,304 per block. Merged into W after training."}[method]
    if notes:
        text(ax, 16, 180, notes, size=9, color=BLUE_DARK, ha="left")
    legend(ax, 596, 180, [("train", "trained"), ("frozen", "frozen"), ("op", "no weights")])
    return fig


def bias_tag(ax, cx, top, label):
    line(ax, [(cx, top), (cx, top + 8)], color=BLUE)
    w = len(label) * 5.2 + 10
    ax.add_patch(FancyBboxPatch((cx - w / 2, top + 8), w, 14, boxstyle="round,pad=0,rounding_size=3", facecolor=BLUE,
                                edgecolor=BLUE_DARK, linewidth=0.8, zorder=4))
    text(ax, cx, top + 15, label, size=8.5, color=WHITE)


def main():
    for mode in ("lp", "partial", "full"):
        save(network(mode), mode)
    for method in ("bitfit", "adapter", "adaptformer", "lora"):
        save(block(method), method)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the CS604 Team 06 slide deck as a 16:9 .pptx, plus optional PNG previews of every slide.

Text is measured with the same TrueType fonts the deck uses (Georgia for titles and numbers, Arial for text),
so any text that would not fit its box is reported here, before the deck is opened in PowerPoint.

Usage:
    python3 slides/build_pptx.py                 # writes cs604_team06_slides.pptx next to the notebooks
    python3 slides/build_pptx.py --preview DIR   # also renders slide-NN.png previews into DIR
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "cs604_team06_slides.pptx")

W, H = 960, 540                      # slide size in points (13.333 x 7.5 in)
LEFT, CONTENT_W, CONTENT_BOTTOM = 64, 832, 484
SUPP = "/System/Library/Fonts/Supplemental/"
FONT_FILES = {("Georgia", False): SUPP + "Georgia.ttf", ("Georgia", True): SUPP + "Georgia Bold.ttf",
              ("Arial", False): SUPP + "Arial.ttf", ("Arial", True): SUPP + "Arial Bold.ttf"}

INK, INK2, MUTED, EYEBROW = "15202B", "4A5563", "6B7280", "1F5FAD"
BLUE, ORANGE, ORANGE_TEXT, GRAY = "2A78D6", "EB6834", "C2531F", "9AA1AB"      # proto = blue, lp = orange (as in the charts)
PAGE, CARD, CARD_BLUE, HEAD_ROW, BAND = "FAF9F6", "F1EFEA", "E7F0FB", "E6E2DA", "F4F2EE"
DARK, DARK_TEXT, DARK_SUB, DARK_BLUE, DARK_ORANGE, DARK_MUTED = "15202B", "F5F3EE", "C9CFD6", "7FB2F0", "F0915F", "9AA4B1"

# ---------------------------------------------------------------- measuring

_fonts = {}


def font(family, bold, px):
    key = (family, bold, px)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_FILES[(family, bold)], max(1, px))
    return _fonts[key]


def text_width(text, family, size, bold):
    return font(family, bold, round(size * 4)).getlength(text) / 4


def wrap(runs, family, size, max_w, bold):
    """Greedy word wrap over styled runs; returns lines as lists of (word, bold, color)."""
    words = [(part, opts.get("bold", bold), opts.get("color"))
             for text, opts in runs for part in text.split(" ") if part]
    space = text_width(" ", family, size, False)
    limit = max_w * 0.96               # PowerPoint wraps a little earlier than raw font metrics
    lines, cur, cur_w = [], [], 0.0
    for word in words:
        ww = text_width(word[0], family, size, word[1])
        add = ww if not cur else space + ww
        if cur and cur_w + add > limit:
            lines.append(cur)
            cur, cur_w = [word], ww
        else:
            cur.append(word)
            cur_w += add
    if cur:
        lines.append(cur)
    return lines


def norm(paras):
    return [[(p, {})] if isinstance(p, str) else p for p in paras]


def T(x, y, w, paras, **kw):
    """A text box. paras: list of strings or lists of (text, {bold, color}) runs."""
    e = dict(kind="text", x=x, y=y, w=w, paras=norm(paras), family="Arial", size=15, color=INK, bold=False,
             line=1.2, after=0, align="left", spacing=0, role=None, caps=False, bullet=0, footer=False)
    e.update(kw)
    if e["caps"]:
        e["paras"] = [[(t.upper(), o) for t, o in p] for p in e["paras"]]
    inner = e["w"] - e["bullet"]
    e["lines"] = [wrap(p, e["family"], e["size"], inner, e["bold"]) for p in e["paras"]]
    e["need"] = sum(len(ls) * e["size"] * e["line"] for ls in e["lines"]) + e["after"] * (len(e["paras"]) - 1)
    e.setdefault("h", e["need"])
    if "h" not in kw:
        e["h"] = e["need"]
    return e


def rect(x, y, w, h, fill, radius=0, outline=None, outline_w=1.0, label=None, name=None, label_color="FFFFFF",
         label_size=13, label_bold=True):
    return dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, radius=radius, outline=outline, outline_w=outline_w,
                label=label, name=name, label_color=label_color, label_size=label_size, label_bold=label_bold)


def oval(x, y, d, fill, label=None, label_color="FFFFFF", label_size=13):
    return dict(kind="oval", x=x, y=y, w=d, h=d, fill=fill, label=label, label_color=label_color, label_size=label_size,
                label_bold=True)


def arrow(x, y, w, h, fill):
    return dict(kind="arrow", x=x, y=y, w=w, h=h, fill=fill)


def image(name, x, y, w, alt):
    path = os.path.join(FIG, name)
    iw, ih = Image.open(path).size
    return dict(kind="image", path=path, x=x, y=y, w=w, h=w * ih / iw, alt=alt)


def figure_with_caption(el, name, top, caption, alt):
    """A centered figure as wide as the space above the footer allows, with a caption under it."""
    cap = T(LEFT, 0, CONTENT_W, [caption], size=11, color=MUTED, line=1.3)
    iw, ih = Image.open(os.path.join(FIG, name)).size
    w = min(CONTENT_W, (CONTENT_BOTTOM - top - 10 - cap["h"]) * iw / ih)
    img = image(name, (W - w) / 2, top, w, alt)
    cap["y"] = img["y"] + img["h"] + 10
    el += [img, cap]


def table(x, y, cols, rows, align, size=14, highlight=(), bold_rows=()):
    """rows[0] is the header. Row heights are measured from the cell text."""
    heights = []
    for i, row in enumerate(rows):
        need = max(len(wrap([(c, {})], "Arial", size, cw - 12, i == 0 or i in bold_rows)) for c, cw in zip(row, cols))
        heights.append(need * size * 1.15 + 10)
    fills = [HEAD_ROW] + [CARD_BLUE if i in highlight else (BAND if i % 2 == 0 else "FFFFFF") for i in range(1, len(rows))]
    return dict(kind="table", x=x, y=y, w=sum(cols), h=sum(heights), cols=cols, rows=rows, align=align, size=size,
                row_h=heights, fills=fills, bold_rows=set(bold_rows))


# ---------------------------------------------------------------- slide furniture

def header(el, eyebrow, title):
    el.append(T(LEFT, 50, CONTENT_W, [eyebrow], size=12, bold=True, color=EYEBROW, spacing=1.5, caps=True))
    t = T(LEFT, 70, CONTENT_W, [title], family="Georgia", size=36, bold=True, color=INK, line=1.12, role="title")
    el.append(t)
    return 70 + t["h"] + 22


def footer(el, n, note="", dark=False):
    color = DARK_MUTED if dark else MUTED
    el.append(T(LEFT, 504, 720, ["CS604 · Team 06" + (" · " + note if note else "")], size=11, color=color, footer=True))
    el.append(T(816, 504, 80, [str(n)], size=11, color=color, align="right", footer=True))


def card_row(el, y, cards, fills, gap=20, pad=18, radius=10, outlines=None):
    """cards: list of item lists; an item is T() kwargs plus 'paras' and optional 'gap' and 'dot' (a color)."""
    n = len(cards)
    w = (CONTENT_W - gap * (n - 1)) / n
    placed, height = [], 0
    for items in cards:
        yy, built = pad, []
        for it in items:
            kw = {k: v for k, v in it.items() if k not in ("paras", "gap", "dot")}
            dx = 16 if it.get("dot") else 0
            t = T(0, 0, w - 2 * pad - dx, it["paras"], **kw)
            built.append((t, yy, dx, it.get("dot")))
            yy += t["h"] + it.get("gap", 8)
        placed.append(built)
        height = max(height, yy - 8 + pad)
    for i, built in enumerate(placed):
        x = LEFT + i * (w + gap)
        el.append(rect(x, y, w, height, fills[i], radius=radius,
                       outline=(outlines[i] if outlines else None), outline_w=1.5))
        for t, yy, dx, dot in built:
            if dot:
                el.append(oval(x + pad, y + yy + (t["size"] * t["line"] - 10) / 2, 10, dot))
            t["x"], t["y"] = x + pad + dx, y + yy
            el.append(t)
    return y + height


# ---------------------------------------------------------------- method diagrams

LADDER = ["LP", "Partial", "Full", "BitFit", "Adapter", "AdaptFormer", "LoRA"]
FROZEN, LINE = HEAD_ROW, "B9B4AA"
K_SHADES = {1: "1F5FAD", 2: BLUE, 4: "7FB2F0", 6: "BFD7F5"}   # block trained for K >= 1, 2, 4 or 6
PROTOCOL = "Same protocol for all seven: prototype head, 3 epochs, 20,000-image holdout, A100"


def pills(el, current):
    """The seven methods as pills in the top right corner; the current one is filled."""
    widths = [text_width(n, "Arial", 10, True) + 16 for n in LADDER]
    gap = 5
    x = LEFT + CONTENT_W - sum(widths) - gap * (len(widths) - 1)
    for i, (name, w) in enumerate(zip(LADDER, widths)):
        fill, color = (BLUE, "FFFFFF") if i == current else (CARD_BLUE, BLUE) if i < current else (CARD, MUTED)
        el.append(rect(x, 46, w, 20, fill, radius=10, label=name, label_color=color, label_size=10))
        x += w + gap


def part(el, x, y, w, h, text, state, size=11):
    """One box of a diagram. state: 'train' (blue), 'frozen' (grey) or 'op' (no parameters, outlined)."""
    if state == "train":
        el.append(rect(x, y, w, h, BLUE, radius=4, label=text, label_size=size))
    elif state == "frozen":
        el.append(rect(x, y, w, h, FROZEN, radius=4, label=text, label_color=INK2, label_size=size, label_bold=False))
    else:
        el.append(rect(x, y, w, h, "FFFFFF", radius=4, outline=FROZEN, outline_w=1.25, label=text, label_color=MUTED,
                       label_size=size, label_bold=False))


def legend(el, x, y, items, note=None):
    """One row of swatches; items are (state, text). An optional note follows in muted text."""
    for state, text in items:
        part(el, x, y + 1, 12, 12, "", state)
        t = T(x + 18, y, text_width(text, "Arial", 11, False) + 4, [text], size=11, color=INK2, line=1.2)
        el.append(t)
        x += 18 + t["w"] + 16
    if note:
        el.append(T(x + 8, y, LEFT + CONTENT_W - x - 8, [note], size=11, color=MUTED, line=1.2))


def stack_diagram(el, top, mode):
    """Patch embedding, the 12 blocks, the final norm and the head in one row. mode: 'lp', 'partial' or 'full'."""
    h, y, bw, gap = 40, top + 6, 38, 6
    el.append(rect(LEFT, y + h / 2 - 0.75, CONTENT_W - 6, 1.5, LINE))
    x = LEFT
    part(el, x, y, 92, h, "Patch embed", "train" if mode == "full" else "frozen")
    x += 92 + 16
    xs = []
    for i in range(12):
        need = 12 - i                       # block i + 1 is trained when K >= need
        if mode == "full":
            part(el, x, y, bw, h, str(i + 1), "train")
        elif mode == "partial" and need <= 6:
            k = next(k for k in (1, 2, 4, 6) if k >= need)
            el.append(rect(x, y, bw, h, K_SHADES[k], radius=4, label=str(i + 1),
                           label_color="FFFFFF" if k in (1, 2) else INK, label_size=11))
        else:
            part(el, x, y, bw, h, str(i + 1), "frozen")
        xs.append(x)
        x += bw + gap
    x += 16 - gap
    part(el, x, y, 52, h, "Norm", "frozen" if mode == "lp" else "train")
    x += 52 + 16
    part(el, x, y, 112, h, "Head (10,000)", "train")
    if mode == "partial":
        for j, k in enumerate((1, 2, 4, 6)):
            by = y + h + 14 + j * 13
            bx = xs[12 - k]
            el.append(rect(bx, by, xs[11] + bw - bx, 4, K_SHADES[k], radius=2))
            el.append(T(bx - 52, by - 5, 46, [f"K = {k}"], size=10, color=INK2, align="right", line=1.2))
    legend(el, LEFT, y + h + 16, [("train", "trained"), ("frozen", "frozen")])


def block_diagram(el, top, method):
    """One transformer block from left to right; the parts `method` trains are blue."""
    yc, bh, gap = top + 44, 32, 12
    seq = [("LN", 36, "norm"), ("qkv", 46, "lin"), ("Attention", 78, "op"), ("proj", 46, "lin"), ("+", 22, "add"),
           ("LN", 36, "norm"), ("fc1", 46, "lin"), ("GELU", 52, "op"), ("fc2", 46, "lin")]
    if method == "adapter":
        seq.append(("Adapter", 70, "adapter"))
    seq.append(("+", 22, "add"))
    pos, x = [], LEFT + 18
    for name, w, kind in seq:
        pos.append((name, kind, x, w))
        x += w + gap
    end = x - gap + 18
    el.append(rect(LEFT, yc - 0.75, end - LEFT, 1.5, LINE))            # the main path
    adds = [p for p in pos if p[1] == "add"]
    split2 = adds[0][2] + adds[0][3] + gap / 2                          # where the second residual leaves the path
    ry = yc - 38
    for sx, add in ((LEFT + 8, adds[0]), (split2, adds[1])):           # residual connections, above the path
        ax = add[2] + add[3] / 2
        el.append(rect(sx - 0.75, ry, 1.5, yc - ry, LINE))
        el.append(rect(sx - 0.75, ry, ax - sx + 1.5, 1.5, LINE))
        el.append(rect(ax - 0.75, ry, 1.5, yc - 11 - ry, LINE))
    if method == "adaptformer":                                         # side branch under the MLP
        fc1, fc2, add2 = pos[6], pos[8], adds[1]
        by, bw2 = yc + 40, 96
        bx = (fc1[2] + fc2[2] + fc2[3]) / 2 - bw2 / 2
        ax = add2[2] + add2[3] / 2
        el.append(rect(split2 - 0.75, yc, 1.5, by - yc, LINE))
        el.append(rect(split2 - 0.75, by - 0.75, ax - split2 + 1.5, 1.5, LINE))
        el.append(rect(ax - 0.75, yc + 11, 1.5, by - yc - 11, LINE))
        el.append(rect(bx, by - 12, bw2, 24, BLUE, radius=4, label="Adapter × s", label_size=11))
    for name, kind, x, w in pos:
        if kind == "add":
            o = oval(x, yc - 11, 22, "FFFFFF", label="+", label_color=INK2, label_size=13)
            o.update(outline=LINE, outline_w=1.25)
            el.append(o)
        else:
            part(el, x, yc - bh / 2, w, bh, name, "train" if kind == "adapter" else "op" if kind == "op" else "frozen")
        if method == "bitfit" and kind in ("norm", "lin"):
            el.append(oval(x + w - 8, yc - bh / 2 - 8, 15, BLUE, label="b", label_size=9))
        if method == "lora" and kind == "lin":
            cx = x + w / 2
            el.append(rect(cx - 0.75, yc + bh / 2, 1.5, 12, LINE))
            el.append(rect(cx - 19, yc + bh / 2 + 12, 38, 20, BLUE, radius=4, label="BA", label_size=10))
    note = {"bitfit": "One of the 12 identical blocks. b = a trained bias.",
            "adapter": "One of the 12 identical blocks. Adapter: 768 → 64 → 768.",
            "adaptformer": "One of the 12 identical blocks. Same adapter, beside the MLP.",
            "lora": "One of the 12 identical blocks. BA: rank-8 update of W."}[method]
    legend(el, LEFT, yc + 64, [("train", "trained"), ("frozen", "frozen"), ("op", "no weights")], note)


def method_slide(slides, n, idx, title, diagram, bullets, notes):
    el = []
    ct = header(el, f"Method {idx + 1} of 7", title)
    pills(el, idx)
    diagram(el, ct)
    paras = [[(lead + " ", {"bold": True, "color": INK}), (text, {})] for lead, text in bullets]
    el.append(T(LEFT, ct + 146, CONTENT_W, paras, size=15, color=INK2, line=1.3, after=10, bullet=16))
    footer(el, n, PROTOCOL)
    slides.append(dict(bg=PAGE, el=el, notes=notes))


# ---------------------------------------------------------------- slides

def build_slides():
    slides = []

    # 1. Cover
    el = []
    el.append(T(LEFT, 64, 700, ["CS604 course project · Team 06"], size=14, bold=True, color=DARK_BLUE, spacing=2, caps=True))
    for i, c in enumerate([GRAY, BLUE, ORANGE]):
        el.append(oval(852 + i * 16, 66, 10, c))
    t = T(LEFT, 150, CONTENT_W, ["Prototype-initialized adaptation of ViT-B/16 for 10,000-class fine-grained classification"],
          family="Georgia", size=44, bold=True, color=DARK_TEXT, line=1.1, role="title")
    el.append(t)
    el.append(T(LEFT, t["y"] + t["h"] + 26, 760, ["Seven adaptation methods, one budget. Our recipe: 76.22% (full fine-tuning) "
                                                  "and 71.90% (LoRA) on CodaBench val."], size=18, color=DARK_SUB, line=1.35))
    el.append(T(LEFT, 468, CONTENT_W, ["[Member names] · 7 November 2026"], size=14, color=DARK_SUB, pinned=True))
    slides.append(dict(bg=DARK, el=el, big_title=True, notes=(
        "Good afternoon. We are Team 06. Our task was to adapt one pretrained Vision Transformer, ViT-B/16, to a 10,000-class "
        "fine-grained dataset under a fixed compute budget.\n\nThe talk has three parts. First, seven adaptation methods, from the "
        "simplest to the most complex, all run with the same protocol. Second, our recipe: how we start the classifier head, and "
        "how we spend the budget. Third, a summary.\n\nThe headline: our full fine-tuning model reaches 76.22% top-1 on the "
        "CodaBench validation set, and our LoRA model reaches 71.90% while training only 1.375% of the backbone.")))

    # 2. Task and reference baselines
    el = []
    ct = header(el, "The task", "One ViT, 10,000 classes, a fixed budget")
    num = dict(family="Georgia", size=34, bold=True, color=BLUE, line=1.1, gap=4)
    lab = dict(size=14, color=INK2, line=1.25)
    bottom = card_row(el, ct, [
        [dict(paras=["10,000"], **num), dict(paras=["fine-grained classes, anonymized"], **lab)],
        [dict(paras=["45"], **num), dict(paras=["training images per class, 450,000 in total"], **lab)],
        [dict(paras=["1.2 h"], **num), dict(paras=["per run on one A100-80GB"], **lab)],
        [dict(paras=["e.g. < 5%"], **num), dict(paras=["trainable share the brief suggests"], **lab)],
    ], [CARD] * 4, gap=16, pad=14)
    tb = table(LEFT, bottom + 18, [400, 144, 144, 144], [
        ["Reference baseline (project brief)", "Trainable", "Reference", "Ours (holdout)"],
        ["Zero-shot: prototype head, no training", "0%", "43.7%", "43.95%"],
        ["Full fine-tuning, 3 epochs", "100%", "70.8%", "70.88%"],
        ["LoRA r = 8 on qkv, 3 epochs", "0.34%", "68.3%", "68.57%"],
    ], ["left", "right", "right", "right"], size=14)
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 14, CONTENT_W, [[("Holdout:", {"bold": True, "color": INK}),
                                                           (" 2 training images per class (20,000). It tracks CodaBench val "
                                                            "within 0.3 points.", {})]], size=14, color=INK2, line=1.3))
    footer(el, 2, "Reference = numbers from a PhD student in the project brief")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The dataset has 10,000 fine-grained classes but only 45 training images per class, 450,000 images in total, so per "
        "class it is close to few-shot. The backbone is fixed: ViT-B/16, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. "
        "Every run must fit in 2 hours on 48 GB; on one A100 with 80 GB that is 1.2 hours per run. The brief suggests that "
        "efficient methods train a small share of the parameters, for example under 5%.\n\nWe first reproduced the three reference "
        "numbers from the brief: zero-shot 43.95 against 43.7, full fine-tuning 70.88 against 70.8, and LoRA on qkv 68.57 against "
        "68.3. The validation labels are hidden, so we hold out 2 training images per class. Holdout agrees with CodaBench within "
        "0.3 points, so the comparisons that follow use holdout.")))

    # 3–9. The seven methods, simplest first
    method_slide(slides, 3, 0, "Linear probe: train only the head", lambda el, top: stack_diagram(el, top, "lp"), [
        ("Idea:", "freeze the whole backbone and train only the new 10,000-class head."),
        ("Setting:", "the head starts from class prototypes; 10 epochs on cached features, LR 1e-3."),
        ("Parameters:", "0 in the backbone. Head: 768 × 10,000 + 10,000 = 7.69M, trained in every method."),
        ("Result:", "61.94% holdout in 0.10 GPU-hours. The untrained prototype head alone: 43.95%."),
        ("Problem:", "the features never adapt to the new classes. Next: unfreeze the top blocks."),
    ], (
        "We go from the simplest method to the most complex. Every method on the next seven slides uses the same protocol: the "
        "same prototype-initialized head, 3 epochs, the same 20,000-image holdout and the same A100. The diagram shows what each "
        "method trains in blue and what it freezes in grey.\n\nThe linear probe freezes the whole backbone and trains only the "
        "new 10,000-class head on cached features. It needs no backbone gradients at all and takes 0.10 GPU-hours. It reaches "
        "61.94%, up from 43.95% for the untrained prototype head. But the features never change, so this is a ceiling for frozen "
        "features. Next we unfreeze the top blocks."))

    method_slide(slides, 4, 1, "Partial fine-tuning: the last K blocks", lambda el, top: stack_diagram(el, top, "partial"), [
        ("Idea:", "train the last K blocks and the final norm; freeze everything below them."),
        ("Setting:", "K = 1 and 2 at LR 3e-4; K = 4 and 6 use the full fine-tuning recipe."),
        ("Parameters:", "K × 7,087,872 + 1,536, so 8.26% (K = 1) to 49.6% (K = 6)."),
        ("Result:", "67.97%, 70.99%, 71.88%, 72.89% for K = 1, 2, 4, 6; memory 3.8 to 10.0 GB."),
        ("Problem:", "each extra block adds less, and the lower blocks never adapt. Next: train every block."),
    ], (
        "Partial fine-tuning unfreezes the last K transformer blocks plus the final norm. We tried K equal to 1, 2, 4 and 6. "
        "Each block has about 7.09 million weights, so even one block is 8.3% of the backbone, already above the 5% the brief "
        "gives as an example.\n\nAccuracy rises from 67.97% with one block to 72.89% with six, but each extra block adds less: "
        "plus 6 points for the first block, plus 3 for the second, then about 1 point per two blocks. Note that K equal to 4 and "
        "6 also use the full fine-tuning recipe, with a lower learning rate, layer decay and drop path.\n\nIt is cheap to train: "
        "gradients only flow through the top blocks, so memory stays between 3.8 and 10 GB. The problem is that the lower blocks "
        "never adapt. So next we train every block."))

    method_slide(slides, 5, 2, "Full fine-tuning: the upper bound", lambda el, top: stack_diagram(el, top, "full"), [
        ("Idea:", "train every backbone weight together with the head."),
        ("Setting:", "LR 1e-4, layer-wise decay 0.75, drop path 0.1, weight decay 0.05."),
        ("Parameters:", "85,798,656 = 100% of the backbone, a 374 MB checkpoint per task."),
        ("Result:", "73.38% holdout, the best of all seven; 18.9 GB, 1,139 img/s, 0.48 GPU-hours."),
        ("Problem:", "every task stores a full copy of the model. Next: train far fewer weights."),
    ], (
        "Full fine-tuning trains every weight. To keep the pretrained features, we use a small learning rate, 1e-4, that "
        "shrinks by 0.75 per block toward the input, plus drop path.\n\nIt is the most accurate method in our comparison: 73.38% "
        "on holdout. It is also the most expensive: 18.9 GB of training memory, and every task needs its own 374 MB checkpoint. "
        "The rest of the ladder asks how close we can get while training far fewer weights."))

    method_slide(slides, 6, 3, "BitFit: train only the biases", lambda el, top: block_diagram(el, top, "bitfit"), [
        ("Idea:", "freeze every weight matrix; train only the bias terms and the head."),
        ("Setting:", "LR 1e-3, no weight decay."),
        ("Parameters:", "12 × 8,448 + 1,536 = 102,912 biases, only 0.120% of the backbone."),
        ("Result:", "69.81% holdout, 3.6 below full; no extra inference cost (7.33 ms)."),
        ("Problem:", "too little capacity to close the gap. Next: add small trainable modules."),
    ], (
        "Now the parameter-efficient methods. From here on the diagram zooms into one of the 12 identical transformer blocks.\n\n"
        "BitFit is the most extreme: it freezes every weight matrix and trains only the bias vectors, marked b. Per block that "
        "is 8,448 numbers, plus the biases of the patch embedding and the final norm: 102,912 in total, 0.12% of the backbone.\n\n"
        "With so little, it still reaches 69.81%, almost 8 points above the linear probe. Because it only changes numbers that "
        "already exist, inference costs nothing extra. But it stays 3.6 points below full fine-tuning: it lacks capacity. So "
        "next we add small trainable modules."))

    method_slide(slides, 7, 4, "Adapter: a bottleneck after each MLP", lambda el, top: block_diagram(el, top, "adapter"), [
        ("Idea:", "insert a small bottleneck, 768 → 64 → 768, after each MLP, in series."),
        ("Setting:", "bottleneck 64, scale 1.0, LR 1e-3; the up-projection starts at zero."),
        ("Parameters:", "12 × (768 × 64 + 64 + 64 × 768 + 768) = 1,189,632 = 1.387%."),
        ("Result:", "71.54% holdout; 12.8 GB; latency 9.74 ms vs 7.33 ms for the plain model."),
        ("Problem:", "it rewrites the MLP output in series. Next: a parallel, scaled branch."),
    ], (
        "The Adapter adds a small bottleneck after the MLP of every block: down to 64 dimensions, a ReLU, and back up to 768, "
        "with its own skip connection. The up-projection starts at zero, so at the start the model is unchanged.\n\nThat is "
        "1.39% of the backbone, and accuracy rises to 71.54%. The cost shows at inference: the adapter is an extra layer in the "
        "main path and cannot be folded into the frozen weights, so batch-1 latency goes from 7.33 to 9.74 ms.\n\nIt also "
        "rewrites the MLP output directly. AdaptFormer, next, moves the same module to a side branch."))

    method_slide(slides, 8, 5, "AdaptFormer: a scaled side branch", lambda el, top: block_diagram(el, top, "adaptformer"), [
        ("Idea:", "run the same bottleneck beside the MLP and add its output, scaled by s."),
        ("Setting:", "bottleneck 64, s = 0.1, LR 1e-3."),
        ("Parameters:", "same as Adapter: 1,189,632 = 1.387% of the backbone."),
        ("Result:", "71.17% holdout, close to Adapter (71.54%); latency 10.23 ms."),
        ("Problem:", "neither adapter merges into the weights, so inference stays slower. Next: a mergeable update."),
    ], (
        "AdaptFormer uses the same bottleneck, but in parallel with the MLP: the block input goes through both, and the adapter "
        "output is added with a small scale, s equal to 0.1, so the original MLP output stays intact.\n\nSame parameter count, "
        "and 71.17% on holdout. That is 0.37 below the serial Adapter, about the size of our run-to-run noise, so in our setting "
        "the placement makes little difference. The two also use different scales, so this is not a pure placement test.\n\n"
        "Both adapters share one problem: they add layers at inference, about 30 to 40% more latency at batch size 1. We want "
        "an update that disappears after training. That is LoRA."))

    method_slide(slides, 9, 6, "LoRA: a low-rank update that merges", lambda el, top: block_diagram(el, top, "lora"), [
        ("Idea:", "learn ΔW = BA beside each frozen linear layer; merge W + (α/r)·BA after training."),
        ("Setting:", "rank r = 8, α = 16, on qkv, proj, fc1 and fc2; LR 1e-3."),
        ("Parameters:", "12 × 8 × (3,072 + 1,536 + 3,840 + 3,840) = 1,179,648 = 1.375%."),
        ("Result:", "71.98% holdout (CodaBench 71.90%), best efficient method; 7.61 ms after merging."),
        ("Problem:", "slowest training here (888 img/s, 19.9 GB). Next: our recipe, used by all seven."),
    ], (
        "LoRA learns a low-rank update next to each frozen linear layer: delta W equals B times A, with rank 8. For a layer with "
        "n inputs and m outputs that is 8 times n plus m weights. We put it on all four linear layers of every block, 1.375% "
        "of the backbone.\n\nIt is the most accurate efficient method, 71.98% on holdout and 71.90% on CodaBench. And after "
        "training we add BA into W, so the deployed model is exactly as fast as the original.\n\nIts cost is training: in our "
        "runs it was the slowest method, 888 images per second, with the most memory, 19.9 GB, because of 96 extra small matrix "
        "multiplications per forward pass, 8 per block, and their stored activations.\n\nAll seven methods used the same prototype head. Next: "
        "what that head is, and what it is worth."))

    # 10. Our recipe and what the head is worth
    el = []
    ct = header(el, "Our method", "Our recipe: a prototype head first")
    steps = [("1", "Extract features", "Run the frozen ViT once; keep each image's 768-d feature.", CARD, INK),
             ("2", "Prototype head", "Class weight = normalized class-mean feature × 40. No training.", CARD_BLUE, BLUE),
             ("3", "Fine-tune", "Any of the seven methods; we keep full fine-tuning and LoRA.", CARD, INK),
             ("4", "Predict", "Full model: average the logits of each image and its flip.", CARD, INK)]
    gap, pad = 34, 14
    bw = (CONTENT_W - 3 * gap) / 4
    built = [(T(0, 0, bw - 2 * pad, [title], size=15, bold=True, line=1.2),
              T(0, 0, bw - 2 * pad, [body], size=13, color=INK2, line=1.3)) for _, title, body, _, _ in steps]
    bh = max(pad + 24 + 10 + tt["h"] + 6 + tbd["h"] + pad for tt, tbd in built)
    y0 = ct + 2
    for i, ((n, _, _, fill, badge), (tt, tbd)) in enumerate(zip(steps, built)):
        x = LEFT + i * (bw + gap)
        el.append(rect(x, y0, bw, bh, fill, radius=10))
        el.append(oval(x + pad, y0 + pad, 24, badge, label=n))
        tt["x"], tt["y"] = x + pad, y0 + pad + 24 + 10
        tbd["x"], tbd["y"] = x + pad, tt["y"] + tt["h"] + 6
        el += [tt, tbd]
        if i < 3:
            el.append(arrow(x + bw + 8, y0 + bh / 2 - 8, gap - 16, 16, GRAY))
    tb = table(LEFT, y0 + bh + 20, [392, 150, 150, 140], [
        ["Same run, only the head changes", "Linear-probe head", "Prototype head", "Gain"],
        ["Full fine-tuning, 3 epochs", "70.88%", "73.38%", "+2.50"],
        ["LoRA on all linear layers, 3 epochs", "69.45%", "71.98%", "+2.53"],
        ["AdaptFormer, 3 epochs", "69.05%", "71.17%", "+2.12"],
        ["Full FT, 8 epochs, all data, flip TTA (CodaBench)", "73.10%", "76.22%", "+3.12"],
    ], ["left", "right", "right", "right"], size=14, highlight=(4,), bold_rows=(4,))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 12, CONTENT_W, ["A random head ties the prototype head (71.95% vs 72.00%, LoRA): "
                                                          "only the linear-probe head hurts."], size=14, color=INK2, line=1.3))
    footer(el, 10, "First three rows: holdout. Last row: CodaBench val.")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Here is our recipe. Step one: run the frozen backbone once over all training images and keep the 768-dimensional "
        "features. Step two, the key choice: each class weight in the new head is the normalized mean feature of that class, "
        "times a temperature of 40. No gradient steps. Step three: fine-tune with any of the seven methods; we keep full "
        "fine-tuning and LoRA. Step four: for the full model we average the logits of each image and its flip.\n\nThe common "
        "alternative, from LP-FT, is to train a linear probe first and then fine-tune. The table compares the two heads with "
        "everything else fixed. The prototype head wins every time: plus 2.5 for full fine-tuning, plus 2.5 for LoRA, plus 2.1 "
        "for AdaptFormer, and plus 3.1 for our final setting on CodaBench. A random head ends level with the prototype head, so "
        "the problem is specifically the linear-probe head.")))

    # 11. Why the linear-probe head hurts
    el = []
    ct = header(el, "Our method · why it works", "Linear-probe heads start high, finish low")
    img = image("fig2_head_init_holdout.png", (W - 640) / 2, ct, 640,
                "Line charts of holdout top-1 accuracy by epoch. Full fine-tuning: the linear-probe head goes from 62.1% "
                "before training to 70.9% after 3 epochs, the prototype head from 44.0% to 73.4%. LoRA on all linear "
                "layers: linear-probe head 62.1% to 69.6%, prototype head 44.0% to 72.0%.")
    el.append(img)
    el.append(T(LEFT, img["y"] + img["h"] + 12, CONTENT_W, [
        "The linear-probe head was trained on these same images, so it starts at 62%.",
        "Its loss starts low (2.29 vs 4.39), so little gradient reaches the backbone.",
        "In all four pairs it ends with lower training loss but lower accuracy: memorization.",
    ], size=14, color=INK2, line=1.3, after=5, bullet=16))
    footer(el, 11, "Blue = prototype head, orange = linear-probe head; “head only” = before any training")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Why does the stronger start lose? The leftmost point is the head on its own: the linear-probe head starts at 62%, far "
        "above the prototype head at 44%. After one epoch the prototype run has caught up, and it stays ahead.\n\nOur "
        "explanation, consistent with the logs: the linear-probe head was trained on frozen features of the very same training "
        "images, so it already fits them. The first training loss is low, 2.29 against 4.39 in the LoRA runs, so only a weak "
        "gradient reaches the backbone. And in all four pairs, the linear-probe run ends with a lower training loss but a lower "
        "accuracy, which is memorization. With only 43 to 45 images per class, fitting the head on the training images is a "
        "liability.")))

    # 12. Ablations
    el = []
    ct = header(el, "Ablations", "LoRA needs capacity; full FT needs epochs")
    tb = table(LEFT, ct + 2, [452, 130, 130, 120], [
        ["One change at a time", "Before", "After", "Change"],
        ["LoRA, lp head: LR 1e-3 → 5e-4 (qkv only)", "68.57%", "68.45%", "−0.12"],
        ["LoRA, lp head: qkv only → all linear layers", "68.57%", "69.45%", "+0.88"],
        ["LoRA, proto head: 3 → 6 epochs", "71.98%", "71.52%", "−0.46"],
        ["Full FT, proto head, all data, flip TTA: 3 → 8 epochs *", "74.01%", "76.22%", "+2.21"],
        ["Full FT, lp head, 8 epochs: + flip TTA *", "72.57%", "72.84%", "+0.27"],
        ["Full FT, lp head, 8 epochs, TTA: 43 → 45 images per class *", "72.84%", "73.10%", "+0.26"],
    ], ["left", "right", "right", "right"], size=14, highlight=(2, 4), bold_rows=(2, 4))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 18, CONTENT_W, [
        "LoRA is limited by capacity, not learning rate: let it adapt the MLP too.",
        "Full fine-tuning keeps gaining with epochs; LoRA starts to memorize after about 3.",
    ], size=15, color=INK2, line=1.3, after=8, bullet=16))
    footer(el, 12, "* CodaBench val; other rows: holdout. lp = linear-probe head, proto = prototype head")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Ablations on the other parts of the recipe, one change at a time.\n\nFor LoRA, halving the learning rate changed "
        "nothing, minus 0.12, within noise. Putting LoRA on all four linear layers instead of only qkv added 0.88 points. So "
        "LoRA is limited by where it has capacity, not by its learning rate.\n\nTraining length behaves in opposite ways. Full "
        "fine-tuning gains 2.2 points from 3 to 8 epochs, which is how we spend the budget for the final model. LoRA with the "
        "prototype head is 0.46 worse at 6 epochs than at 3: at learning rate 1e-3 it starts to memorize, so we stop it at 3 "
        "epochs.\n\nThe last two rows are small steps for the final model: flip test-time augmentation plus 0.27, and training "
        "on all 45 images per class plus 0.26, at the edge of noise but in the expected direction.")))

    # 13. Final results
    el = []
    ct = header(el, "Final models", "Final results, inside the compute budget")
    tb = table(LEFT, ct + 2, [262, 96, 70, 82, 104, 100, 118], [
        ["Model", "CodaBench val", "Test", "Trainable", "Latency bs1", "GPU-hours", "Peak VRAM"],
        ["Reference: full fine-tuning", "70.8%", "–", "100%", "–", "–", "–"],
        ["Reference: LoRA", "68.3%", "–", "0.34%", "–", "–", "–"],
        ["Ours: full FT, proto head, 8 epochs, all data, flip TTA", "76.22%", PLACEHOLDER, "100%", "14.95 ms *", "1.03 h",
         "18.9 GB"],
        ["Ours: LoRA on all linear layers, proto head, 3 epochs", "71.90%", PLACEHOLDER, "1.375%", "7.61 ms", "0.54 h",
         "19.9 GB"],
    ], ["left"] + ["right"] * 6, size=14, highlight=(3, 4), bold_rows=(3, 4))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 18, CONTENT_W, [
        "Full fine-tuning: +5.4 over the reference, using 1.03 of the 1.2 allowed GPU-hours.",
        "LoRA: +3.6 over the reference LoRA, training 1.375% of the backbone.",
        "* Flip TTA runs two forward passes per image; one pass takes 7.35 ms.",
    ], size=15, color=INK2, line=1.3, after=8, bullet=16))
    footer(el, 13, "Trainable = share of the 85.8M-parameter backbone; the 7.69M-parameter head is extra")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The two final models next to the references, on every metric the brief asks for.\n\nFull fine-tuning with the "
        "prototype head, 8 epochs, all data and flip TTA reaches 76.22% on CodaBench val, 5.4 points above the reference, in "
        "1.03 of the 1.2 allowed GPU-hours. Its latency is 14.95 ms because of the two TTA passes; one pass takes 7.35 ms.\n\n"
        "Our LoRA model trains 1.375% of the backbone and reaches 71.90%, 3.6 points above the reference LoRA, in 0.54 "
        "GPU-hours. After merging it is as fast as the base model.\n\nThe test column is filled in after Phase 2.")))

    # 14. Summary table
    el = []
    ct = header(el, "Summary", "Seven methods, one protocol")
    tb = table(LEFT, ct + 2, [168, 128, 118, 112, 186, 120], [
        ["Method", "Trainable params", "% of backbone", "Train memory", "Inference overhead", "Holdout top-1"],
        ["Linear probe", "0", "0%", "3.2 GB", "none", "61.94%"],
        ["Partial, K = 1", "7,089,408", "8.26%", "≤ 3.8 GB", "none", "67.97%"],
        ["Partial, K = 2", "14,177,280", "16.5%", "4.1 GB", "none", "70.99%"],
        ["Partial, K = 4", "28,353,024", "33.0%", "7.0 GB", "none", "71.88%"],
        ["Partial, K = 6", "42,528,768", "49.6%", "10.0 GB", "none", "72.89%"],
        ["Full fine-tuning", "85,798,656", "100%", "18.9 GB", "none", "73.38%"],
        ["BitFit", "102,912", "0.120%", "12.7 GB", "none", "69.81%"],
        ["Adapter", "1,189,632", "1.387%", "12.8 GB", "+1.19M params, +2.4 ms", "71.54%"],
        ["AdaptFormer", "1,189,632", "1.387%", "12.8 GB", "+1.19M params, +2.9 ms", "71.17%"],
        ["LoRA", "1,179,648", "1.375%", "19.9 GB", "none after merging", "71.98%"],
    ], ["left"] + ["right"] * 5, size=13, highlight=(6, 10), bold_rows=(6, 10))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 12, CONTENT_W, ["Overhead is against the plain model (7.33 ms at batch 1). Peak "
                                                          "memory for K = 1 comes from feature extraction."],
                size=13, color=MUTED, line=1.3))
    footer(el, 14, PROTOCOL)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "All seven methods in one table, with the columns the brief asks for.\n\nFull fine-tuning is the most accurate, 73.38%. "
        "Among the efficient methods LoRA is the most accurate, 71.98%, and after merging it adds nothing at inference. The two "
        "adapters add 1.19 million weights and 2.4 to 2.9 ms per image. BitFit is the smallest by far, 0.12%, and still beats "
        "partial fine-tuning of one block.\n\nThe memory column holds a surprise, which the next slide explains: the methods "
        "that train the fewest weights do not use the least memory.")))

    # 15. Accuracy and memory against trainable parameters
    el = []
    ct = header(el, "Summary", "Spread small changes over every block")
    img = image("fig4_methods_params.png", (W - 760) / 2, ct, 760,
                "Two scatter plots against trainable backbone parameters on a log axis. Left, holdout top-1: BitFit 69.8% at "
                "0.12%; LoRA 72.0%, Adapter 71.5% and AdaptFormer 71.2% near 1.4%; the partial fine-tuning line rises from "
                "68.0% at 8.3% (K = 1) to 72.9% at 49.6% (K = 6) and 73.4% at 100%. Right, peak training memory: BitFit "
                "12.7 GB, Adapter and AdaptFormer 12.8 GB, LoRA 19.9 GB; partial fine-tuning 3.8, 4.1, 7.0 and 10.0 GB for "
                "K = 1, 2, 4, 6, and full fine-tuning 18.9 GB.")
    el.append(img)
    el.append(T(LEFT, img["y"] + img["h"] + 12, CONTENT_W, [
        "LoRA (1.375%) matches K = 4 (33%); BitFit (0.12%) beats K = 1 (8.26%).",
        "Memory follows how far gradients flow back, not how many weights are trained.",
    ], size=15, color=INK2, line=1.3, after=8, bullet=16))
    footer(el, 15, PROTOCOL)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Two findings from the comparison. Left: accuracy against the share of trained weights, on a log scale. The blue methods "
        "make small changes in every block; the grey line trains whole blocks from the top. The blue points sit far to the left "
        "of the line: LoRA with 1.375% of the weights matches training the last four blocks, which is 33%, and BitFit with "
        "0.12% beats training the last block, 8.3%. So where the weights are matters more than how many.\n\nRight: peak "
        "training memory. Partial fine-tuning adds about 1.5 GB per trained block; extended to all 12 blocks that gives 18.8 "
        "GB, which is what full fine-tuning used. The blue methods train almost nothing, yet need 12.7 to 19.9 GB, because "
        "their gradients must flow back to the first block. Memory follows how far gradients flow back, not how many weights "
        "are trained.")))

    # 16. Takeaways
    el = []
    el.append(T(LEFT, 60, CONTENT_W, ["Takeaways"], family="Georgia", size=36, bold=True, color=DARK_TEXT, line=1.12, role="title"))
    y = 128
    for i, (text, color) in enumerate([
            ("Start the head from class prototypes: +2.1 to +3.1 points for every method we tried.", DARK_BLUE),
            ("Spread capacity over all blocks: LoRA at 1.375% matches partial fine-tuning at 33%.", DARK_BLUE),
            ("Training memory follows how far gradients flow back, not the number of trained weights.", DARK_BLUE),
            ("Full fine-tuning is most accurate; LoRA is the best efficient method, at no inference cost.", DARK_BLUE),
            ("Final: 76.22% (full) and 71.90% (LoRA) on CodaBench val, inside the budget.", DARK_ORANGE)], 1):
        el.append(T(LEFT, y - 5, 40, [str(i)], family="Georgia", size=28, bold=True, color=color, line=1.1))
        t = T(LEFT + 52, y, CONTENT_W - 52, [text], size=19, color=DARK_TEXT, line=1.3, role="point")
        el.append(t)
        y += t["h"] + 22
    footer(el, 16, dark=True)
    slides.append(dict(bg=DARK, el=el, notes=(
        "To sum up. One: start the 10,000-class head from class prototypes, not a linear probe; that is worth 2.1 to 3.1 points "
        "for every method we tried. Two: spread the trainable weights over all blocks; LoRA at 1.375% matches training a third "
        "of the network. Three: training memory follows how far gradients flow back, not how many weights you train. Four: full "
        "fine-tuning is the most accurate, and LoRA is the best efficient method and costs nothing at inference after merging. "
        "And the outcome: 76.22% and 71.90% on CodaBench val, both inside the budget.")))

    # 17. Thank you
    el = []
    el.append(T(LEFT, 64, 700, ["CS604 · Team 06"], size=14, bold=True, color=EYEBROW, spacing=2, caps=True))
    for i, c in enumerate([GRAY, BLUE, ORANGE]):
        el.append(oval(852 + i * 16, 66, 10, c))
    t = T(LEFT, 196, CONTENT_W, ["Thank you"], family="Georgia", size=60, bold=True, color=INK, line=1.05, role="title")
    el.append(t)
    el.append(T(LEFT, t["y"] + t["h"] + 18, CONTENT_W, ["Questions are welcome."], size=22, color=INK2, line=1.3))
    el.append(T(LEFT, 468, CONTENT_W, ["[Member names]"], size=14, color=INK2, pinned=True))
    slides.append(dict(bg=CARD, el=el, big_title=True, notes=(
        "Thank you. We are happy to take questions. Likely questions: why prototypes beat the linear probe, why LoRA was not "
        "faster to train, and whether the gains hold on the test set. The backup slides have the exact hyperparameters, how we "
        "counted GPU time, and the step-by-step path to 76.22%.")))

    # 18. Backup: implementation details
    el = []
    ct = header(el, "Backup", "Implementation details and budget accounting")
    colw = (CONTENT_W - 32) / 2
    lists = [("Training", [
        "Backbone: timm vit_base_patch16_224.augreg2_in21k_ft_in1k, 85.8M parameters",
        "AdamW, 0.3 warmup epochs then cosine, bf16, batch 256, label smoothing 0.1",
        "Augmentation: RandomResizedCrop (area 0.25–1) and horizontal flip",
        "Full and partial K = 4, 6: LR 1e-4, layer decay 0.75, drop path 0.1",
        "LoRA r = 8, α = 16; adapters 64-d; all efficient methods use LR 1e-3"]),
        ("Measurement", [
        "Budget: 96 GB·h ÷ 80 GB = 1.2 h per run on an A100-80GB",
        "Counted: feature extraction for the head, training, per-epoch holdout evaluation",
        "Latency: random input, forward pass only, bf16, batch 1",
        "Software: torch 2.11.0 (CUDA 13.0), timm 1.0.30, peft 0.21.1",
        "One ablation batch ran on an RTX PRO 6000; used for accuracy only"])]
    for i, (head, items) in enumerate(lists):
        x = LEFT + i * (colw + 32)
        h3 = T(x, ct + 2, colw, [head], size=16, bold=True, line=1.2)
        el.append(h3)
        el.append(T(x, h3["y"] + h3["h"] + 10, colw, items, size=14, color=INK2, line=1.25, after=8, bullet=14))
    footer(el, 18)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Backup slide for questions about implementation. Left: the training recipe. Right: how we measured. GPU time counts "
        "everything that uses the GPU for training, including the feature extraction for the prototype head, but not unpacking "
        "data, saving checkpoints or the final prediction. Latency uses random input, forward pass only, bf16, batch size 1; "
        "identical models differ by about 0.5 ms between runs.")))

    # 19. Backup: the path to the final score
    el = []
    ct = header(el, "Backup", "72.57% to 76.22%, one change at a time")
    figure_with_caption(el, "fig1_codabench_progression.png", ct,
                        "CodaBench val top-1. Each step changes one setting; grey is the reference from the brief. The axis "
                        "starts at 70% because the dots encode position, not length.",
                        "Dot plot of CodaBench val top-1: reference 70.80%; full fine-tuning with the linear-probe head, 43 images "
                        "per class, 8 epochs 72.57%; plus flip TTA 72.84% (+0.27); plus all 45 images per class 73.10% (+0.26); "
                        "plus the prototype head, the final model, 76.22% (+3.12).")
    footer(el, 19)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The path to our final CodaBench score, one change per step, with full fine-tuning for 8 epochs throughout. The "
        "linear-probe head scored 72.57%; flip TTA added 0.27; all 45 images per class added 0.26; switching only the head to "
        "prototypes added 3.12, more than everything before it combined.")))
    return slides


# ---------------------------------------------------------------- checks

_cmaps = {}
PLACEHOLDER = "[待填]"   # marks a result we do not have yet; PowerPoint draws the two CJK characters with a fallback font
MAX_BULLETS, MAX_WORDS = 5, 20


def missing_glyphs(text, family, bold):
    """Characters the font file has no glyph for (PowerPoint would substitute another font or show a box)."""
    if (family, bold) not in _cmaps:
        from reportlab.pdfbase.ttfonts import TTFont as ReportLabFont
        _cmaps[(family, bold)] = ReportLabFont(f"{family}-{bold}", FONT_FILES[(family, bold)]).face.charToGlyph
    text = text.replace(PLACEHOLDER, "")
    return sorted({ch for ch in text if ord(ch) > 32 and ord(ch) not in _cmaps[(family, bold)]})


def check(slides):
    problems = []
    for n, s in enumerate(slides, 1):
        for e in s["el"]:
            right, bottom = e["x"] + e["w"], e["y"] + e["h"]
            if right > W - 60 + 0.5 or e["x"] < 60 - 0.5:
                problems.append(f"slide {n}: {e['kind']} outside the side margins (x {e['x']:.0f}–{right:.0f})")
            if e.get("label") and text_width(e["label"], "Arial", e["label_size"], e["label_bold"]) > e["w"] - 4:
                problems.append(f"slide {n}: label {e['label']!r} is wider than its {e['kind']}")
            if e["kind"] == "text" and (e["bullet"] or e.get("role") == "point"):
                if len(e["paras"]) > MAX_BULLETS:
                    problems.append(f"slide {n}: {len(e['paras'])} bullets (max {MAX_BULLETS})")
                for p in e["paras"]:
                    words = "".join(t for t, _ in p).split()
                    if len(words) > MAX_WORDS:
                        problems.append(f"slide {n}: bullet has {len(words)} words (max {MAX_WORDS}): {' '.join(words[:6])} ...")
            if e["kind"] == "table":
                for row in e["rows"]:
                    for cell in row:
                        for bold in (False, True):
                            if missing_glyphs(cell, "Arial", bold):
                                problems.append(f"slide {n}: no glyph for {missing_glyphs(cell, 'Arial', bold)} in Arial")
            if e["kind"] == "text":
                if e["need"] > e["h"] + 0.5:
                    problems.append(f"slide {n}: text overflows its box ({e['need']:.0f} > {e['h']:.0f} pt): {e['paras'][0][0][0][:50]!r}")
                for p in e["paras"]:
                    for t, o in p:
                        bold = o.get("bold", e["bold"])
                        gaps = missing_glyphs(t, e["family"], bold)
                        if gaps:
                            problems.append(f"slide {n}: no glyph for {gaps} in {e['family']}{' Bold' if bold else ''}: {t[:40]!r}")
                        for word in t.split(" "):
                            if word and text_width(word, e["family"], e["size"], bold) > e["w"] - e["bullet"]:
                                problems.append(f"slide {n}: word wider than its box: {word!r}")
                if not (e["footer"] or e.get("pinned") or e.get("role") == "title") and bottom > CONTENT_BOTTOM + 0.5:
                    problems.append(f"slide {n}: text runs into the footer zone (bottom {bottom:.0f})")
            elif bottom > CONTENT_BOTTOM + 0.5:
                problems.append(f"slide {n}: {e['kind']} runs into the footer zone (bottom {bottom:.0f})")
        title = next((e for e in s["el"] if e.get("role") == "title"), None)
        if title and not s.get("big_title"):
            print(f"  slide {n:2d} title: {len(title['lines'][0])} line(s)")
    return problems


# ---------------------------------------------------------------- PPTX writer

def emu(pt):
    return Emu(int(round(pt * 12700)))


def rgb(hexstr):
    return RGBColor.from_string(hexstr)


def write_text(slide, e):
    if e.get("role") == "title":
        shp = slide.shapes.title
        shp.left, shp.top, shp.width, shp.height = emu(e["x"]), emu(e["y"]), emu(e["w"]), emu(e["h"])
    else:
        shp = slide.shapes.add_textbox(emu(e["x"]), emu(e["y"]), emu(e["w"]), emu(e["h"]))
    tf = shp.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    last = len(e["paras"]) - 1
    for i, runs in enumerate(e["paras"]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        for r in list(p.runs):
            p._p.remove(r._r)
        p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT, "center": PP_ALIGN.CENTER}[e["align"]]
        p.line_spacing = Pt(e["size"] * e["line"])
        if i < last and e["after"]:
            p.space_after = Pt(e["after"])
        if e["bullet"]:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(e["bullet"] * 12700)))
            pPr.set("indent", str(int(-e["bullet"] * 12700)))
            pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
            pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = e["family"]
            f.size = Pt(e["size"])
            f.bold = opts.get("bold", e["bold"])
            f.color.rgb = rgb(opts.get("color") or e["color"])
            if e["spacing"]:
                r._r.get_or_add_rPr().set("spc", str(int(e["spacing"] * 100)))


def write_shape(slide, e):
    kind = {"rect": MSO_SHAPE.ROUNDED_RECTANGLE if e.get("radius") else MSO_SHAPE.RECTANGLE,
            "oval": MSO_SHAPE.OVAL, "arrow": MSO_SHAPE.RIGHT_ARROW}[e["kind"]]
    shp = slide.shapes.add_shape(kind, emu(e["x"]), emu(e["y"]), emu(e["w"]), emu(e["h"]))
    if e["kind"] == "rect" and e.get("radius"):
        shp.adjustments[0] = min(0.5, e["radius"] / min(e["w"], e["h"]))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(e["fill"])
    if e.get("outline"):
        shp.line.color.rgb = rgb(e["outline"])
        shp.line.width = Pt(e["outline_w"])
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = False
    if e.get("label"):
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = e["label"]
        r.font.name, r.font.size, r.font.bold = "Arial", Pt(e.get("label_size", 13)), e.get("label_bold", True)
        r.font.color.rgb = rgb(e.get("label_color", "FFFFFF"))


def write_image(slide, e):
    pic = slide.shapes.add_picture(e["path"], emu(e["x"]), emu(e["y"]), emu(e["w"]), emu(e["h"]))
    pic._element.nvPicPr.cNvPr.set("descr", e["alt"])


def write_table(slide, e):
    shp = slide.shapes.add_table(len(e["rows"]), len(e["cols"]), emu(e["x"]), emu(e["y"]), emu(e["w"]), emu(e["h"]))
    tbl = shp.table
    tbl.first_row, tbl.horz_banding = True, False
    for j, cw in enumerate(e["cols"]):
        tbl.columns[j].width = emu(cw)
    for i, rh in enumerate(e["row_h"]):
        tbl.rows[i].height = emu(rh)
    for i, row in enumerate(e["rows"]):
        for j, value in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(e["fills"][i])
            cell.margin_left = cell.margin_right = emu(6)
            cell.margin_top = cell.margin_bottom = emu(5)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if e["align"][j] == "right" else PP_ALIGN.LEFT
            p.line_spacing = Pt(e["size"] * 1.15)
            r = p.add_run()
            r.text = value
            r.font.name, r.font.size = "Arial", Pt(e["size"])
            r.font.bold = i == 0 or i in e["bold_rows"]
            r.font.color.rgb = rgb(INK)


def write_pptx(slides, path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = emu(W), emu(H)
    prs.core_properties.title = "Prototype-initialized adaptation of ViT-B/16 for 10,000-class fine-grained classification"
    prs.core_properties.author = "CS604 Team 06"
    layout = next(l for l in prs.slide_layouts if l.name == "Title Only")
    for s in slides:
        slide = prs.slides.add_slide(layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = rgb(s["bg"])
        for e in s["el"]:
            {"text": write_text, "rect": write_shape, "oval": write_shape, "arrow": write_shape,
             "image": write_image, "table": write_table}[e["kind"]](slide, e)
        slide.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(path)


# ---------------------------------------------------------------- previews (same geometry, drawn with PIL)

SC = 2


def render(slides, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n, s in enumerate(slides, 1):
        img = Image.new("RGB", (W * SC, H * SC), "#" + s["bg"])
        d = ImageDraw.Draw(img)
        for e in s["el"]:
            box = [e["x"] * SC, e["y"] * SC, (e["x"] + e["w"]) * SC, (e["y"] + e["h"]) * SC]
            k = e["kind"]
            if k in ("rect", "oval"):
                if k == "rect":
                    d.rounded_rectangle(box, radius=min(e["radius"] * SC, (box[3] - box[1]) / 2, (box[2] - box[0]) / 2),
                                        fill="#" + e["fill"], outline=("#" + e["outline"]) if e.get("outline") else None,
                                        width=int(e["outline_w"] * SC))
                else:
                    d.ellipse(box, fill="#" + e["fill"], outline=("#" + e["outline"]) if e.get("outline") else None)
                if e.get("label"):
                    f = font("Arial", e.get("label_bold", True), round(e.get("label_size", 13) * SC))
                    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), e["label"], font=f,
                           fill="#" + e.get("label_color", "FFFFFF"), anchor="mm")
            elif k == "arrow":
                x0, y0, x1, y1 = box
                mid, shaft = (y0 + y1) / 2, (y1 - y0) * 0.25
                head = x0 + (x1 - x0) * 0.55
                d.polygon([(x0, mid - shaft), (head, mid - shaft), (head, y0), (x1, mid), (head, y1), (head, mid + shaft),
                           (x0, mid + shaft)], fill="#" + e["fill"])
            elif k == "image":
                im = Image.open(e["path"]).convert("RGB").resize((int(e["w"] * SC), int(e["h"] * SC)))
                img.paste(im, (int(e["x"] * SC), int(e["y"] * SC)))
            elif k == "table":
                y = e["y"]
                for i, (row, rh) in enumerate(zip(e["rows"], e["row_h"])):
                    x = e["x"]
                    for j, (value, cw) in enumerate(zip(row, e["cols"])):
                        d.rectangle([x * SC, y * SC, (x + cw) * SC, (y + rh) * SC], fill="#" + e["fills"][i], outline="#FFFFFF", width=2)
                        bold = i == 0 or i in e["bold_rows"]
                        lines = wrap([(value, {})], "Arial", e["size"], cw - 12, bold)
                        lh = e["size"] * 1.15
                        ty = y + (rh - len(lines) * lh) / 2
                        for line in lines:
                            text = " ".join(w for w, _, _ in line)
                            f = font("Arial", bold, round(e["size"] * SC))
                            tw = f.getlength(text) / SC
                            tx = x + cw - 6 - tw if e["align"][j] == "right" else x + 6
                            d.text((tx * SC, ty * SC), text, font=f, fill="#" + INK)
                            ty += lh
                        x += cw
                    y += rh
            elif k == "text":
                y = e["y"]
                px = round(e["size"] * SC)
                space = font(e["family"], False, px).getlength(" ") / SC
                for lines in e["lines"]:
                    lh = e["size"] * e["line"]
                    for l_i, line in enumerate(lines):
                        segments = []          # consecutive words of one style are drawn as one string
                        for w, b, c in line:
                            if segments and segments[-1][1:] == (b, c):
                                segments[-1] = (segments[-1][0] + " " + w, b, c)
                            else:
                                segments.append((w, b, c))
                        widths = [font(e["family"], b, px).getlength(t) / SC for t, b, _ in segments]
                        total = sum(widths) + space * (len(segments) - 1)
                        x = e["x"] + e["w"] - total if e["align"] == "right" else e["x"] + e["bullet"]
                        if e["bullet"] and l_i == 0:
                            d.text((e["x"] * SC, y * SC), "•", font=font("Arial", False, px), fill="#" + e["color"])
                        for (t, b, c), ww in zip(segments, widths):
                            d.text((x * SC, y * SC), t, font=font(e["family"], b, px), fill="#" + (c or e["color"]))
                            x += ww + space
                        y += lh
                    y += e["after"]
        path = os.path.join(out_dir, f"slide-{n:02d}.png")
        img.save(path)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="directory for slide-NN.png previews")
    args = ap.parse_args()
    slides = build_slides()
    problems = check(slides)
    print(f"{len(slides)} slides; layout problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    write_pptx(slides, OUT)
    print("wrote", OUT)
    if args.preview:
        for p in render(slides, args.preview):
            print(p)


if __name__ == "__main__":
    main()

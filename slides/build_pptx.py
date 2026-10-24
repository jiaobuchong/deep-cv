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


def rect(x, y, w, h, fill, radius=0, outline=None, outline_w=1.0, label=None, name=None):
    return dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, radius=radius, outline=outline, outline_w=outline_w,
                label=label, name=name)


def oval(x, y, d, fill, label=None):
    return dict(kind="oval", x=x, y=y, w=d, h=d, fill=fill, label=label)


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
    el.append(T(LEFT, t["y"] + t["h"] + 26, 760, ["Full fine-tuning reaches 76.22% and LoRA 71.90% top-1 on CodaBench val, "
                                                  "inside a 2-hour, 48 GB training budget"], size=18, color=DARK_SUB, line=1.35))
    el.append(T(LEFT, 468, CONTENT_W, ["[Member names] · 7 November 2026"], size=14, color=DARK_SUB, pinned=True))
    slides.append(dict(bg=DARK, el=el, notes=(
        "Good afternoon. We are Team 06. Our task was to adapt one pretrained Vision Transformer, ViT-B/16, to a 10,000-class "
        "fine-grained dataset under a fixed compute budget, with both full fine-tuning and parameter-efficient methods.\n\n"
        "The headline: our final full fine-tuning model reaches 76.22% top-1 on the CodaBench validation set, and our LoRA model "
        "reaches 71.90% while training only 1.375% of the backbone. The single change that mattered most was how we initialize "
        "the 10,000-class classifier head.")))

    # 2. Task
    el = []
    ct = header(el, "The task", "One ViT, 10,000 classes, a fixed budget")
    num = dict(family="Georgia", size=40, bold=True, color=BLUE, line=1.1, gap=6)
    lab = dict(size=15, color=INK2, line=1.25)
    bottom = card_row(el, ct + 4, [
        [dict(paras=["10,000"], **num), dict(paras=["fine-grained classes in an anonymized dataset"], **lab)],
        [dict(paras=["45"], **num), dict(paras=["training images per class, 450,000 in total"], **lab)],
        [dict(paras=["1.2 h"], **num), dict(paras=["per training run on one A100-80GB (the 2 h × 48 GB budget)"], **lab)],
        [dict(paras=["< 5%"], **num), dict(paras=["trainable parameters for the parameter-efficient track"], **lab)],
    ], [CARD] * 4, gap=16)
    el.append(T(LEFT, bottom + 30, CONTENT_W, ["Base model: ViT-B/16, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k (timm). "
                                               "Graded on top-1 accuracy, trainable parameters, inference latency and training GPU-hours."],
                size=15, color=INK2, line=1.35))
    footer(el, 2)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The dataset has 10,000 fine-grained classes but only 45 training images per class, 450,000 images in total. So per class "
        "this is close to a few-shot problem.\n\nThe backbone is fixed: ViT-B/16, pretrained on ImageNet-21k and fine-tuned on "
        "ImageNet-1k. Every training run must fit in 2 hours on 48 GB of GPU memory. We trained on one A100 with 80 GB, so we "
        "converted the budget to 96 GB-hours divided by 80 GB, which is 1.2 hours per run, counting head initialization, training "
        "and per-epoch evaluation.\n\nThe parameter-efficient track should train less than 5% of the parameters. We report all four "
        "metrics the brief asks for: top-1 accuracy, trainable parameters, inference latency and training GPU-hours.")))

    # 3. Baselines
    el = []
    ct = header(el, "Starting point", "We first reproduced the reference baselines")
    tb = table(LEFT, ct + 2, [400, 144, 144, 144], [
        ["Method", "Trainable params", "Reference", "Ours (holdout)"],
        ["Zero-shot: prototype head, no training", "0%", "43.7%", "43.95%"],
        ["Full fine-tuning, 3 epochs", "100%", "70.8%", "70.88%"],
        ["LoRA r = 8 on qkv, 3 epochs", "0.34%", "68.3%", "68.57%"],
    ], ["left", "right", "right", "right"], size=15)
    el.append(tb)
    note = T(LEFT + 22, 0, CONTENT_W - 44, [[("Holdout protocol.", {"bold": True}),
                                            (" We set aside 2 training images per class (20,000 images) as a local validation set. "
                                             "It tracks CodaBench val within 0.3 points (72.31% vs 72.57%; 71.98% vs 71.90%), so we "
                                             "judged every ablation on holdout and saved CodaBench submissions.", {})]],
             size=15, color=INK2, line=1.35)
    ny = tb["y"] + tb["h"] + 30
    el.append(rect(LEFT, ny, CONTENT_W, note["h"] + 36, CARD, radius=10))
    note["y"] = ny + 18
    el.append(note)
    footer(el, 3, "Reference = numbers reported by a PhD student in the project brief")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Before trying anything new, we reproduced the three reference numbers from the brief. The zero-shot prototype head gives "
        "43.95% against 43.7%, full fine-tuning for 3 epochs gives 70.88% against 70.8%, and LoRA with rank 8 on the attention qkv "
        "projection gives 68.57% against 68.3%. So our pipeline matches the reference.\n\nThe validation labels are hidden, so we "
        "held out 2 training images per class, 20,000 images, as a local validation set. Holdout and CodaBench agree within 0.3 "
        "points on the two models we checked, so the ablations that follow are on holdout.")))

    # 4. Pipeline
    el = []
    ct = header(el, "Our method", "Our pipeline: prototype head, fine-tune, flip TTA")
    steps = [("1", "Extract features", "Run the frozen ViT-B/16 once over all training images and keep the 768-d CLS features.", CARD, INK),
             ("2", "Prototype head", "Class weight = L2-normalized mean feature of the class × temperature 40. No gradient steps.", CARD_BLUE, BLUE),
             ("3", "Fine-tune", "Full fine-tuning, or LoRA r = 8 on qkv, proj, fc1 and fc2 (1.375% of the backbone).", CARD, INK),
             ("4", "Predict", "Average the logits of each image and its horizontal flip (test-time augmentation).", CARD, INK)]
    gap, pad = 36, 16
    bw = (CONTENT_W - 3 * gap) / 4
    built = []
    for _, title, body, _, _ in steps:
        tt = T(0, 0, bw - 2 * pad, [title], size=16, bold=True, line=1.2)
        tbd = T(0, 0, bw - 2 * pad, [body], size=14, color=INK2, line=1.3)
        built.append((tt, tbd))
    bh = max(pad + 26 + 12 + tt["h"] + 8 + tbd["h"] + pad for tt, tbd in built)
    y0 = ct + 6
    for i, ((n, _, _, fill, badge), (tt, tbd)) in enumerate(zip(steps, built)):
        x = LEFT + i * (bw + gap)
        el.append(rect(x, y0, bw, bh, fill, radius=10))
        el.append(oval(x + pad, y0 + pad, 26, badge, label=n))
        tt["x"], tt["y"] = x + pad, y0 + pad + 26 + 12
        tbd["x"], tbd["y"] = x + pad, tt["y"] + tt["h"] + 8
        el += [tt, tbd]
        if i < 3:
            el.append(arrow(x + bw + 8, y0 + bh / 2 - 8, gap - 16, 16, GRAY))
    el.append(T(LEFT, y0 + bh + 30, CONTENT_W, ["Every run is budget-aware: after 120 steps the measured speed sets how many steps fit "
                                                "in 1.2 GPU-hours, and the cosine learning-rate schedule is compressed to fit."],
                size=15, color=INK2, line=1.35))
    footer(el, 4)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Here is the whole pipeline. Step one: we run the frozen backbone once over all training images and keep the "
        "768-dimensional CLS features. Step two, the key design choice: each class's weight in the new 10,000-class head is the "
        "L2-normalized mean of its features, scaled by a temperature of 40. That takes no gradient steps at all. Step three: we "
        "fine-tune, either the whole network or LoRA with rank 8 on every linear layer of every block, which is 1.375% of the "
        "backbone. Step four: at prediction time we average the logits of the image and its horizontal flip.\n\nEvery run is also "
        "budget-aware. After 120 steps we measure the speed, compute how many steps fit into 1.2 GPU-hours, and compress the "
        "cosine schedule if needed, so no run can exceed the budget.")))

    # 5. Head initialization options
    el = []
    ct = header(el, "Key design choice", "Three ways to start the 10,000-class head")
    big = dict(family="Georgia", size=36, bold=True, line=1.1, gap=2)
    cap = dict(size=12, color=MUTED, line=1.2, gap=10)
    body = dict(size=14, color=INK2, line=1.3)
    bottom = card_row(el, ct + 4, [
        [dict(paras=["Random"], size=16, bold=True, dot=GRAY, gap=10), dict(paras=["≈ 0%"], color=MUTED, **big),
         dict(paras=["head-only accuracy before training"], **cap),
         dict(paras=["Small random weights (the timm default). The head starts with no class knowledge."], **body)],
        [dict(paras=["Prototype (ours)"], size=16, bold=True, dot=BLUE, gap=10), dict(paras=["44.0%"], color=BLUE, **big),
         dict(paras=["head-only accuracy before training"], **cap),
         dict(paras=["Normalized class-mean features × temperature. No training, about 6 minutes of feature extraction."], **body)],
        [dict(paras=["Linear probe"], size=16, bold=True, dot=ORANGE, gap=10), dict(paras=["62.1%"], color=ORANGE_TEXT, **big),
         dict(paras=["head-only accuracy before training"], **cap),
         dict(paras=["The prototype head, then 10 epochs of training on the frozen features of the same training images."], **body)],
    ], [CARD, CARD_BLUE, CARD])
    el.append(T(LEFT, bottom + 30, CONTENT_W, [[("Intuition says: start from the strongest head, the linear probe. ", {}),
                                                ("The experiments say otherwise.", {"bold": True})]], size=17, line=1.35))
    footer(el, 5)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The new head has 10,000 classes and 7.69 million parameters, so how we start it matters. We compared three options.\n\n"
        "Random: the timm default, small random weights, so the head knows nothing at the start.\n\nPrototype, our choice: each "
        "class weight is the normalized mean feature of that class, times a temperature. It needs no training and already gives 44% "
        "before any fine-tuning.\n\nLinear probe: we start from the prototypes and train only the head for 10 epochs on frozen "
        "features. It starts strongest, at 62%. This is the classic linear-probe-then-fine-tune recipe, and it was our first "
        "design. Intuition says the strongest starting head should win. The next slide shows it does not.")))

    # 6. Finding 1
    el = []
    ct = header(el, "Finding 1", "Prototype heads beat linear-probe heads by 2.5 to 3.1 points")
    big = dict(family="Georgia", size=48, bold=True, color=BLUE, line=1.05, gap=8)
    sett = dict(size=15, bold=True, line=1.25, gap=6)
    det = dict(size=14, color=INK2, line=1.3)
    bottom = card_row(el, ct + 4, [
        [dict(paras=["+2.5"], **big), dict(paras=["Full fine-tuning, 3 epochs"], **sett), dict(paras=["lp 70.88% → proto 73.38% (holdout)"], **det)],
        [dict(paras=["+2.5"], **big), dict(paras=["LoRA on all linear layers, 3 epochs"], **sett), dict(paras=["lp 69.45% → proto 71.98% (holdout)"], **det)],
        [dict(paras=["+3.1"], **big), dict(paras=["Full fine-tuning, 8 epochs, all data, flip TTA"], **sett),
         dict(paras=["lp 73.10% → proto 76.22% (CodaBench val)"], **det)],
    ], [CARD_BLUE] * 3)
    el.append(T(LEFT, bottom + 28, CONTENT_W, ["Everything else is identical within each pair. A random head ends level with the "
                                               "prototype head (71.95% vs 72.00%, LoRA, 3 epochs) but converges more slowly, so only "
                                               "the linear-probe head hurts."], size=15, color=INK2, line=1.35))
    footer(el, 6, "lp = linear-probe head, proto = prototype head")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "This is our main finding. In every pair, with everything else identical, the prototype head beats the linear-probe head. "
        "Full fine-tuning for 3 epochs: plus 2.5 points on holdout. LoRA on all linear layers for 3 epochs: plus 2.5 points. And our "
        "final setting, full fine-tuning for 8 epochs on all data with flip TTA: plus 3.1 points on CodaBench, from 73.10% to "
        "76.22%.\n\nWe also tried a random head. It ends level with the prototype head, 71.95% against 72.00%, but learns more "
        "slowly in the first epochs. So the problem is specifically the linear-probe head.")))

    # 7. Learning curves
    el = []
    ct = header(el, "Finding 1 · evidence", "Linear-probe heads start high, finish low")
    figure_with_caption(el, "fig2_head_init_holdout.png", ct,
                        "Holdout top-1 after each epoch; “head only” is the accuracy before any training. Left: full "
                        "fine-tuning. Right: LoRA on all linear layers. Blue = prototype head, orange = linear-probe head.",
                        "Line charts of holdout top-1 accuracy by epoch. Full fine-tuning: the linear-probe head goes from 62.1% "
                        "before training to 70.9% after 3 epochs, the prototype head from 44.0% to 73.4%. LoRA on all linear "
                        "layers: linear-probe head 62.1% to 69.6%, prototype head 44.0% to 72.0%.")
    footer(el, 7)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Here are the learning curves behind that finding. The leftmost point is the head on its own, before any training: the "
        "linear-probe head, in orange, starts at 62%, far above the prototype head, in blue, at 44%.\n\nBut after the first epoch "
        "the prototype run has caught up, and from then on it stays ahead. After 3 epochs it finishes 2.5 points higher for full "
        "fine-tuning on the left, and 2.5 points higher for LoRA on the right. With LoRA the linear-probe run even stalls in the "
        "first epoch: holdout stays at 62%.")))

    # 8. Why the linear-probe head hurts
    el = []
    ct = header(el, "Finding 1 · why", "The linear-probe head has already fit the training set")
    col = T(LEFT, ct + 4, 300, [
        [("Weak signal.", {"bold": True}), (" It was trained on frozen features of the same training images, so the first "
                                             "training loss is low (2.29 vs 4.39 with prototypes, LoRA) and little gradient "
                                             "reaches the backbone.", {})],
        [("Then a mismatch.", {"bold": True}), (" Once the features move, the fitted head no longer fits: with LoRA the "
                                                 "training loss rose from 2.22 to 2.69 in epoch 1.", {})],
        [("Prototypes generalize.", {"bold": True}), (" Higher training loss (1.85 vs 1.77), yet +3.1 points on CodaBench.", {})],
    ], size=14, color=INK2, line=1.32, after=12)
    el.append(col)
    img = image("fig3_loss_lp_vs_proto.png", 388, ct + 4, 508,
                "Training loss over 8 epochs of full fine-tuning on all data. The prototype-head run starts near 5.0 and ends at "
                "1.85; the linear-probe-head run starts near 2.8 and ends at 1.77. Both drop in steps at every epoch boundary. "
                "CodaBench: 76.22% with the prototype head, 73.10% with the linear-probe head.")
    el.append(img)
    footer(el, 8, "Chart: full fine-tuning, all data, 8 epochs")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Why does the stronger start lose? Our explanation, consistent with the logs: the linear-probe head was trained on frozen "
        "features of the very same training images, so it already fits the training set. The training loss starts low, 2.29 "
        "against 4.39 for the prototype head in the LoRA runs, so only a weak gradient reaches the backbone and it learns less.\n\n"
        "Then, as soon as the backbone features move, the fitted head no longer matches them. With LoRA the training loss actually "
        "rose from 2.22 to 2.69 during the first epoch, and the holdout accuracy stalled.\n\nThe chart shows full fine-tuning on "
        "all data. The prototype run keeps a higher training loss all the way, 1.85 against 1.77 in the last epoch, but it is 3.1 "
        "points better on CodaBench, so the linear-probe run is fitting the training images rather than learning general "
        "features. The step at every epoch boundary is the same memorization effect.")))

    # 9. PEFT design
    el = []
    ct = header(el, "Finding 2", "For PEFT, capacity matters more than learning rate")
    tb = table(LEFT, ct + 2, [300, 96, 96, 120, 104, 116], [
        ["Setting (lp head, 3 epochs)", "Trainable", "Holdout", "Train speed", "Peak VRAM", "Latency bs1"],
        ["LoRA on qkv (baseline)", "0.344%", "68.57%", "1,341 img/s", "13.5 GB", "7.22 ms"],
        ["LoRA on qkv, LR 1e-3 → 5e-4", "0.344%", "68.45%", "1,353 img/s", "13.5 GB", "7.15 ms"],
        ["LoRA on qkv, proj, fc1, fc2", "1.375%", "69.45%", "885 img/s", "19.9 GB", "7.03 ms"],
        ["AdaptFormer (parallel to the MLP)", "1.387%", "69.05%", "1,437 img/s", "12.8 GB", "9.89 ms"],
        ["Full fine-tuning (reference)", "100%", "70.88%", "1,131 img/s", "18.9 GB", "7.31 ms"],
    ], ["left"] + ["right"] * 5, size=14, highlight=(3,), bold_rows=(3,))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 28, CONTENT_W, ["Letting LoRA adapt the MLP adds 0.88 points; halving the learning "
                                                          "rate changes nothing (−0.12, within noise). AdaptFormer comes close, "
                                                          "but it cannot be merged into the backbone, so it is about 35% slower "
                                                          "at batch size 1."], size=15, color=INK2, line=1.35))
    footer(el, 9, "All on an A100-80GB, bf16")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "The second question: how should the parameter-efficient model spend its budget? All runs here use the same linear-probe "
        "head and 3 epochs, so only the PEFT design changes.\n\nHalving the LoRA learning rate did nothing: minus 0.12, which is "
        "within noise. Putting LoRA on every linear layer, so that it also adapts the MLP, gained 0.88 points, at 1.375% of the "
        "backbone, still far below the 5% limit. So the bottleneck is capacity, not the learning rate.\n\nAdaptFormer, the adapter "
        "in parallel to the MLP, is close in accuracy and the fastest to train, but its modules cannot be merged into the backbone, "
        "so inference at batch size 1 is about 35% slower than a merged LoRA.")))

    # 10. Training length
    el = []
    ct = header(el, "Finding 3", "More epochs help full fine-tuning, not LoRA")
    tb = table(LEFT, ct + 2, [440, 136, 136, 120], [
        ["Setting", "Shorter", "Longer", "Change"],
        ["Full fine-tuning, lp head (holdout)", "3 ep: 70.88%", "8 ep: 72.31%", "+1.43"],
        ["Full fine-tuning, proto head, all data, flip TTA (CodaBench)", "3 ep: 74.01%", "8 ep: 76.22%", "+2.21"],
        ["LoRA on qkv, lp head (holdout) *", "3 ep: 68.57%", "10 ep: 68.69%", "+0.12"],
        ["LoRA on all linear layers, lp head (holdout)", "3 ep: 69.45%", "6 ep: 69.87%", "+0.42"],
        ["LoRA on all linear layers, proto head (holdout)", "3 ep: 71.98%", "6 ep: 71.52%", "−0.46"],
    ], ["left", "right", "right", "right"], size=14, highlight=(2,), bold_rows=(2,))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 28, CONTENT_W, ["Full fine-tuning uses LR 1e-4 with layer-wise decay, so it moves gently "
                                                          "and keeps improving. LoRA at LR 1e-3 starts to memorize after about 3 epochs."],
                size=15, color=INK2, line=1.35))
    footer(el, 10, "* this run also lowered the head LR to 2e-4")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Third finding: training length behaves in opposite ways for the two methods. Full fine-tuning keeps improving: 3 to 8 "
        "epochs adds 1.4 points with the linear-probe head and 2.2 points on CodaBench with the prototype head, which is how we got "
        "to 76.22%.\n\nLoRA saturates after about 3 epochs. With the qkv-only LoRA, 10 epochs is no better than 3; with the "
        "prototype head, 6 epochs is even 0.46 points worse than 3. Our reading: full fine-tuning uses a small learning rate, 1e-4, "
        "with layer-wise decay, so it moves gently; LoRA uses 1e-3 and starts to memorize the training images. So we train full "
        "fine-tuning for the whole budget and stop LoRA at 3 epochs.")))

    # 11. Progression
    el = []
    ct = header(el, "Results", "72.57% to 76.22%, one change at a time")
    figure_with_caption(el, "fig1_codabench_progression.png", ct,
                        "CodaBench val top-1. Each step changes one setting; grey is the reference from the brief. The axis "
                        "starts at 70% because the dots encode position, not length.",
                        "Dot plot of CodaBench val top-1: reference 70.80%; full fine-tuning with the linear-probe head, 43 images "
                        "per class, 8 epochs 72.57%; plus flip TTA 72.84% (+0.27); plus all 45 images per class 73.10% (+0.26); "
                        "plus the prototype head, the final model, 76.22% (+3.12).")
    footer(el, 11)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "This chart is the path to our final CodaBench score, one change per step, with the same model and 8 epochs throughout.\n\n"
        "Our full fine-tuning with the linear-probe head scored 72.57%, already above the 70.8% reference. Flip test-time "
        "augmentation added 0.27 points. Training on all 45 images per class instead of holding out 2 added another 0.26. That one "
        "is at the edge of noise, but it points the expected way. Then switching only the head initialization to prototypes added "
        "3.12 points, to 76.22%. That last step is bigger than everything before it combined.")))

    # 12. Final results
    el = []
    ct = header(el, "Final models", "Final results, all inside the compute budget")
    tb = table(LEFT, ct + 2, [270, 92, 66, 82, 108, 100, 114], [
        ["Model", "CodaBench val", "Test", "Trainable", "Latency bs1", "GPU-hours", "Peak VRAM"],
        ["Reference: full fine-tuning", "70.8%", "–", "100%", "–", "–", "–"],
        ["Reference: LoRA", "68.3%", "–", "0.34%", "–", "–", "–"],
        ["Ours: full fine-tuning, proto head, 8 epochs, all data, flip TTA", "76.22%", "[__%]", "100%", "14.95 ms *", "1.03 h", "18.9 GB"],
        ["Ours: LoRA on all linear layers, proto head, 3 epochs", "71.90%", "[__%]", "1.375%", "7.61 ms", "0.54 h", "19.9 GB"],
    ], ["left"] + ["right"] * 6, size=14, highlight=(3, 4), bold_rows=(3, 4))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 24, CONTENT_W, ["* Flip TTA runs two forward passes per image; without it the same model "
                                                          "takes 7.35 ms. Budget per run: 1.2 GPU-hours on an A100-80GB (2 h × 48 GB) "
                                                          "and 48 GB of memory. Latency: batch 1, bf16, forward pass only. Test "
                                                          "accuracy is added after Phase 2 (6 November)."],
                size=13, color=INK2, line=1.35))
    footer(el, 12, "Trainable = share of the 85.8M-parameter backbone; the 7.69M-parameter head is extra")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Here are the two final models side by side with the references, on all the metrics the brief asks for.\n\nFull "
        "fine-tuning with the prototype head, 8 epochs, all data and flip TTA reaches 76.22% on CodaBench val, 5.4 points above the "
        "reference, using 1.03 of the 1.2 allowed GPU-hours and 18.9 GB of memory. Its latency is 14.95 ms per image because of the "
        "two TTA passes; without TTA it is 7.35 ms.\n\nOur LoRA model trains only 1.375% of the backbone and reaches 71.90%, 3.6 "
        "points above the reference LoRA, in 0.54 GPU-hours. After merging, its latency equals the base model.\n\nThe test-set "
        "column will be filled in after Phase 2.")))

    # 13. Efficiency
    el = []
    ct = header(el, "Cost", "What PEFT saves, and what it does not")
    tit = dict(size=16, bold=True, line=1.2, gap=10)
    big = dict(family="Georgia", size=36, bold=True, line=1.1, gap=8)
    body = dict(size=14, color=INK2, line=1.3)
    card_row(el, ct + 4, [
        [dict(paras=["Saves: parameters and storage"], **tit), dict(paras=["1.375%"], color=BLUE, **big),
         dict(paras=["of the backbone is trained. The checkpoint is about 36 MB (31 MB of it is the head), against 374 MB for "
                     "full fine-tuning."], **body)],
        [dict(paras=["Does not save: training time or memory"], **tit), dict(paras=["885 img/s"], color=ORANGE_TEXT, **big),
         dict(paras=["for LoRA on all linear layers, against 1,131 img/s for full fine-tuning; peak memory 19.9 GB against "
                     "18.9 GB."], **body)],
        [dict(paras=["Merging matters at inference"], **tit), dict(paras=["+35%"], color=ORANGE_TEXT, **big),
         dict(paras=["batch-1 latency for AdaptFormer (9.89 ms), which cannot be merged. Merged LoRA adds nothing (about "
                     "7.3 ms)."], **body)],
    ], [CARD] * 3)
    footer(el, 13, "All measured on an A100-80GB, bf16")
    slides.append(dict(bg=PAGE, el=el, notes=(
        "A practical lesson about cost. What PEFT clearly saves is trainable parameters and storage: 1.375% of the backbone, and a "
        "checkpoint of about 36 MB, most of which is the 10,000-class head, against 374 MB for full fine-tuning.\n\nWhat it did not "
        "save in our runs is training time or memory. Gradients still flow back through the whole frozen backbone, and the extra "
        "LoRA branches add work, so LoRA on all linear layers trained at 885 images per second against 1,131 for full fine-tuning, "
        "with slightly more peak memory.\n\nAt inference, merging is what matters. LoRA merges into the weights and costs nothing "
        "extra. AdaptFormer cannot be merged and is about 35% slower at batch size 1.")))

    # 14. Takeaways
    el = []
    el.append(T(LEFT, 64, CONTENT_W, ["Takeaways"], family="Georgia", size=36, bold=True, color=DARK_TEXT, line=1.12, role="title"))
    y = 140
    for i, (text, color) in enumerate([
            ("Start the 10,000-class head from class prototypes, not a linear probe: +2.5 to +3.1 points.", DARK_BLUE),
            ("Give LoRA capacity in the MLP rather than a different learning rate: +0.88 points at 1.375% of the parameters.", DARK_BLUE),
            ("Spend the whole budget on full fine-tuning; stop LoRA at about 3 epochs.", DARK_BLUE),
            ("Result: 76.22% with full fine-tuning and 71.90% with LoRA on CodaBench val, both inside the budget.", DARK_ORANGE)], 1):
        el.append(T(LEFT, y - 6, 40, [str(i)], family="Georgia", size=30, bold=True, color=color, line=1.1))
        t = T(LEFT + 56, y, CONTENT_W - 56, [text], size=20, color=DARK_TEXT, line=1.3)
        el.append(t)
        y += t["h"] + 26
    footer(el, 14, dark=True)
    slides.append(dict(bg=DARK, el=el, notes=(
        "To sum up in four points. One: initialize the 10,000-class head from class prototypes, not from a linear probe; that alone "
        "is worth 2.5 to 3.1 points. Two: for LoRA, add capacity in the MLP rather than tuning the learning rate. Three: full "
        "fine-tuning should use the whole budget, while LoRA should stop at about 3 epochs. And four, the outcome: 76.22% with full "
        "fine-tuning and 71.90% with LoRA on CodaBench val, both well inside the compute budget.")))

    # 15. Thank you
    el = []
    el.append(T(LEFT, 64, 700, ["CS604 · Team 06"], size=14, bold=True, color=EYEBROW, spacing=2, caps=True))
    for i, c in enumerate([GRAY, BLUE, ORANGE]):
        el.append(oval(852 + i * 16, 66, 10, c))
    t = T(LEFT, 196, CONTENT_W, ["Thank you"], family="Georgia", size=60, bold=True, color=INK, line=1.05, role="title")
    el.append(t)
    el.append(T(LEFT, t["y"] + t["h"] + 18, CONTENT_W, ["Questions are welcome."], size=22, color=INK2, line=1.3))
    el.append(T(LEFT, 468, CONTENT_W, ["[Member names]"], size=14, color=INK2, pinned=True))
    slides.append(dict(bg=CARD, el=el, notes=(
        "Thank you. We are happy to take questions. Likely questions: why prototypes beat the linear probe, whether the gains hold "
        "on the test set, and why LoRA was not faster to train. The backup slide has the exact hyperparameters and how we counted "
        "GPU time.")))

    # 16. Backup: implementation details
    el = []
    ct = header(el, "Backup", "Implementation details and budget accounting")
    colw = (CONTENT_W - 32) / 2
    lists = [("Training", [
        "Backbone: timm vit_base_patch16_224.augreg2_in21k_ft_in1k (85.8M parameters); new 10,000-class head (7.69M)",
        "AdamW, warmup then cosine decay, bf16, batch 256, label smoothing 0.1, gradient clipping at 1.0",
        "Augmentation: RandomResizedCrop (area 0.25–1) and horizontal flip",
        "Full fine-tuning: LR 1e-4, weight decay 0.05, layer-wise LR decay 0.75, drop path 0.1",
        "LoRA: r = 8, α = 16, LR 1e-3, weight decay 1e-4, merged after training",
        "AdaptFormer: bottleneck 64, scale 0.1, parallel to the MLP"]),
        ("Measurement", [
        "Budget: 96 GB·h ÷ 80 GB = 1.2 h per run on an A100-80GB (±10% allowed)",
        "Counted: head initialization with feature extraction, training, per-epoch holdout evaluation",
        "Not counted: unpacking data, saving checkpoints, final prediction",
        "Latency: random input, forward pass only, bf16, batch 1; about 3,850 img/s at batch 256",
        "Software: torch 2.11.0 (CUDA 13.0), timm 1.0.30, peft 0.21.1",
        "Hardware: A100-SXM4-80GB with 12 CPUs; one ablation batch ran on an RTX PRO 6000 and is used for accuracy only"])]
    for i, (head, items) in enumerate(lists):
        x = LEFT + i * (colw + 32)
        h3 = T(x, ct + 2, colw, [head], size=16, bold=True, line=1.2)
        el.append(h3)
        el.append(T(x, h3["y"] + h3["h"] + 10, colw, items, size=14, color=INK2, line=1.25, after=6, bullet=14))
    footer(el, 16)
    slides.append(dict(bg=PAGE, el=el, notes=(
        "Backup slide for questions about implementation. Left: the training recipe for both methods. Right: how we measured. GPU "
        "time counts everything that uses the GPU for training, including the feature extraction for the prototype head, but not "
        "unpacking data, saving checkpoints or the final prediction. Latency is measured with random input, forward pass only, in "
        "bf16 at batch size 1. The software versions were logged from the later runs onward.")))
    return slides


# ---------------------------------------------------------------- checks

_cmaps = {}


def missing_glyphs(text, family, bold):
    """Characters the font file has no glyph for (PowerPoint would substitute another font or show a box)."""
    if (family, bold) not in _cmaps:
        from reportlab.pdfbase.ttfonts import TTFont as ReportLabFont
        _cmaps[(family, bold)] = ReportLabFont(f"{family}-{bold}", FONT_FILES[(family, bold)]).face.charToGlyph
    return sorted({ch for ch in text if ord(ch) > 32 and ord(ch) not in _cmaps[(family, bold)]})


def check(slides):
    problems = []
    for n, s in enumerate(slides, 1):
        for e in s["el"]:
            right, bottom = e["x"] + e["w"], e["y"] + e["h"]
            if right > W - 60 + 0.5 or e["x"] < 60 - 0.5:
                problems.append(f"slide {n}: {e['kind']} outside the side margins (x {e['x']:.0f}–{right:.0f})")
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
        if title and n not in (1, 15):
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
    if e.get("label"):
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = e["label"]
        r.font.name, r.font.size, r.font.bold = "Arial", Pt(13), True
        r.font.color.rgb = rgb("FFFFFF")


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
            if k == "rect":
                d.rounded_rectangle(box, radius=e["radius"] * SC, fill="#" + e["fill"],
                                    outline=("#" + e["outline"]) if e.get("outline") else None, width=int(e["outline_w"] * SC))
            elif k == "oval":
                d.ellipse(box, fill="#" + e["fill"])
                if e.get("label"):
                    f = font("Arial", True, 13 * SC)
                    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), e["label"], font=f, fill="#FFFFFF", anchor="mm")
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

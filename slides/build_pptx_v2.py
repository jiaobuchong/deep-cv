#!/usr/bin/env python3
"""Build the 15-minute version of the CS604 Team 06 deck (15 slides, then 5 backup slides) as cs604_team06_slides_v2.pptx.

Speaker plan:  A  slides 1–3   opening; the problem and what it implies; baselines and the seven methods   ~3:10
               B  slides 4–6   how we iterated; the method comparison; the problem with the linear-probe head ~3:30
               C  slides 7–9   our method, why it works (loss evidence), each other change with its reason ~3:55
               D  slides 10–15 the two final models (one architecture slide each), results, resolution, close
Backup slides 16–20, for questions: the other five methods, rebuilt from the method slides of build_pptx.py.
It follows the instructor's guidance: same backbone as recommended, focus on adaptation methods, and show the process
(analysis, iterations, failed ideas, lessons), not only the final accuracy.
The layout helpers, checks, writer and preview renderer come from build_pptx.py.

Usage:
    python3 slides/build_pptx_v2.py                 # writes cs604_team06_slides_v2.pptx next to the notebooks
    python3 slides/build_pptx_v2.py --preview DIR   # also renders slide-NN.png previews into DIR
"""
import argparse
import os


from build_pptx import (BLUE, CARD, CARD_BLUE, CONTENT_W, DARK, DARK_BLUE, DARK_MUTED, DARK_ORANGE, DARK_SUB, DARK_TEXT,
                        DIAGRAM_ALT,
                        EYEBROW, GRAY, INK, INK2, LEFT, MUTED, NAMES, ORANGE, PAGE, ROOT, TEST_FULL, TEST_LORA,
                        W, T, arrow, card_row, check, figure_with_caption, footer, header, image, oval, rect, render, table,
                        write_pptx)
from build_pptx import build_slides as v1_slides

OUT = os.path.join(ROOT, "cs604_team06_slides_v2.pptx")
WPM = 130                 # speaking rate used for the timing estimate


def lead_paras(items):
    """Bulleted paragraphs that start with a bold lead-in."""
    return [[(lead + " ", {"bold": True, "color": INK}), (text, {})] for lead, text in items]


def flow(el, y, steps, gap=34, pad=16, body_size=14):
    """Numbered cards in a row joined by arrows. steps: (number, title, body, fill, badge color). Returns the bottom."""
    bw = (CONTENT_W - (len(steps) - 1) * gap) / len(steps)
    built = [(T(0, 0, bw - 2 * pad, [title], size=16, bold=True, line=1.2),
              T(0, 0, bw - 2 * pad, [body], size=body_size, color=INK2, line=1.3)) for _, title, body, _, _ in steps]
    bh = max(pad + 26 + 10 + tt["h"] + 6 + tb["h"] + pad for tt, tb in built)
    for i, ((n, _, _, fill, badge), (tt, tb)) in enumerate(zip(steps, built)):
        x = LEFT + i * (bw + gap)
        el.append(rect(x, y, bw, bh, fill, radius=10))
        el.append(oval(x + pad, y + pad, 26, badge, label=n))
        tt["x"], tt["y"] = x + pad, y + pad + 26 + 10
        tb["x"], tb["y"] = x + pad, tt["y"] + tt["h"] + 6
        el += [tt, tb]
        if i < len(steps) - 1:
            el.append(arrow(x + bw + 8, y + bh / 2 - 8, gap - 16, 16, GRAY))
    return y + bh


def build_slides():
    A, B, C, D = NAMES
    slides = []

    # 1. Cover (A)
    el = []
    el.append(T(LEFT, 64, 700, ["CS604 course project · Team 06"], size=14, bold=True, color=DARK_BLUE, spacing=2, caps=True))
    for i, c in enumerate([GRAY, BLUE, ORANGE]):
        el.append(oval(852 + i * 16, 66, 10, c))
    t = T(LEFT, 150, CONTENT_W, ["Prototype-initialized adaptation of ViT-B/16 for 10,000-class fine-grained classification"],
          family="Georgia", size=44, bold=True, color=DARK_TEXT, line=1.1, role="title")
    el.append(t)
    el.append(T(LEFT, t["y"] + t["h"] + 26, 780, [
        "Seven adaptation methods, one budget, and the one choice that mattered most: how the new head starts.",
        f"Test top-1: {TEST_FULL} with full fine-tuning, {TEST_LORA} with LoRA."], size=18, color=DARK_SUB, line=1.35, after=8))
    el.append(T(LEFT, 468, CONTENT_W, [" · ".join(NAMES) + " · 7 November 2026"], size=14, color=DARK_SUB, pinned=True))
    slides.append(dict(bg=DARK, el=el, big_title=True, speaker="A", notes=(
        "Good afternoon, we are Team 06. Our task was to adapt one pretrained Vision Transformer, ViT-B/16, to 10,000 "
        "fine-grained classes within a fixed compute budget. We compared seven adaptation methods. Along the way, one design "
        "choice mattered more than the choice of method: how the new 10,000-class head starts. On the hidden test set we "
        f"reach {TEST_FULL} with full fine-tuning and {TEST_LORA} with LoRA.")))

    # 2. The problem, and what it implies for our design (A)
    el = []
    ct = header(el, "Part 1 · The problem", "Many classes, few photos, one budget")
    side = 300
    el.append(image("data_examples.jpg", LEFT, ct + 2, side,
                    "Sixteen training photos in a 4 by 4 grid, one class per row: a bird seen through a scope, as a flock, at "
                    "a feeder and on reeds; a snake on rock, pavement, gravel and leaves; a moth on a wall, a leaf, close up "
                    "and on fabric; a plant as flowers, fruit, a whole shrub and a branch."))
    x0, cw = LEFT + side + 32, CONTENT_W - side - 32
    y = ct + 2
    for num, label, so in [("10,000", "fine-grained classes; the class names are hidden",
                            "the features must adapt, not only the head"),
                           ("45", "training photos per class, 450,000 in total",
                            "easy to memorize; judge on 20,000 held-out photos, not training loss"),
                           ("1.2 h", "per run on one A100-80GB: 2 h × 48 GB of GPU memory, ±10%",
                            "every idea must fit one budget, and we count all GPU time"),
                           ("ViT-B/16", "pretrained on ImageNet-21k, fine-tuned on ImageNet-1k; 85.8M parameters",
                            "the backbone the brief recommends; we keep it in every experiment")]:
        n = T(x0, y, cw, [num], family="Georgia", size=28, bold=True, color=BLUE, line=1.1)
        lab = T(x0, y + n["h"] + 1, cw, [label], size=14, color=INK2, line=1.3)
        imp = T(x0, lab["y"] + lab["h"] + 3, cw, [[("So: ", {"bold": True}), (so, {})]], size=14, color=EYEBROW, line=1.3)
        el += [n, lab, imp]
        y = imp["y"] + imp["h"] + 12
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="A", notes=(
        "Before choosing any method, we asked what makes this problem hard, because each difficulty points to a design "
        "choice. First, 10,000 fine-grained classes with hidden names: the pretrained features were not made for these "
        "distinctions, so the backbone itself has to adapt, not only the classifier. Second, only 45 photos per class, and "
        "very different ones: on the left, each row is one class, from a single bird to a whole flock, from a flower to a "
        "whole shrub. A model can simply memorize them, so we never trust the training loss and always judge on held-out photos. That "
        "turned out to be the key risk. Validation, 50,000 photos, and test, 100,000, have hidden labels. Third, a fixed "
        "budget of 2 hours on 48 GB, which is 1.2 hours per run on our 80 GB A100. We count all GPU time: feature extraction "
        "for the head, training and evaluation during training.\n\nThe backbone is the one the brief recommends: ViT-B/16 pretrained on ImageNet-21k "
        "and fine-tuned on ImageNet-1k. We did not change it, and we checked that the ImageNet-21k-only versions give no "
        "better features.\n\nOur yardstick: we hold out 2 training photos per class, 20,000 in total. It tracks CodaBench "
        "within about 0.3 points, so every comparison in this talk uses it.")))

    # 3. Baselines and the seven methods (A)
    el = []
    ct = header(el, "Part 1 · Starting point", "Baselines first, then seven methods")
    tb = table(LEFT, ct, [400, 144, 144, 144], [
        ["Reference baseline (project brief)", "Trainable", "Reference", "Ours (holdout)"],
        ["Zero-shot: prototype head, no training", "0%", "43.7%", "43.95%"],
        ["Full fine-tuning, 3 epochs", "100%", "70.8%", "70.88%"],
        ["LoRA r = 8 on qkv, 3 epochs", "0.34%", "68.3%", "68.57%"],
    ], ["left", "right", "right", "right"], size=13)
    el.append(tb)
    note = T(LEFT, tb["y"] + tb["h"] + 6, CONTENT_W, [[("Same backbone, same numbers: ", {"bold": True, "color": INK}),
                                                       ("all three within 0.3 points of the reference.", {})]],
             size=14, color=INK2, line=1.3)
    el.append(note)
    name = dict(size=14, bold=True, line=1.2, gap=3)
    desc = dict(size=13, color=INK2, line=1.25, gap=4)
    pct = dict(family="Georgia", size=17, bold=True, color=BLUE, line=1.1)
    y = note["y"] + note["h"] + 8
    for title, cards, fill in [
            ("Train whole blocks, from the top",
             [("Linear probe", "the head only", "0 (0%)"), ("Partial", "the last K of the 12 blocks", "7.1M–42.5M (8–50%)"),
              ("Full fine-tuning", "every weight", "85.8M (100%)")], CARD),
            ("Small changes in every block",
             [("BitFit", "the bias terms only", "102,912 (0.12%)"), ("Adapter", "a bottleneck after each MLP", "1.19M (1.39%)"),
              ("AdaptFormer", "a bottleneck beside each MLP", "1.19M (1.39%)"),
              ("LoRA", "a low-rank update, merged after training", "1.18M (1.38%)")], CARD_BLUE)]:
        lab = T(LEFT, y, CONTENT_W, [title], size=11, bold=True, color=EYEBROW, spacing=1.2, caps=True)
        el.append(lab)
        y = card_row(el, lab["y"] + lab["h"] + 4,
                     [[dict(paras=[n], **name), dict(paras=[d], **desc), dict(paras=[p], **pct)] for n, d, p in cards],
                     [fill] * len(cards), gap=12, pad=10, radius=8) + 8
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="A", notes=(
        "Before trying anything new, we reproduced the three reference numbers in the brief: zero-shot 43.95% against 43.7, "
        "full fine-tuning for 3 epochs 70.88 against 70.8, and LoRA on the attention qkv layer 68.57 against 68.3. So our "
        "pipeline matches the reference.\n\nThen we compared seven ways to adapt the model, in two families. The first family "
        "trains whole blocks from the top: the linear probe trains only the head, partial fine-tuning trains the last K of the "
        "12 blocks, and full fine-tuning trains every weight. The second family makes small changes in every block: BitFit "
        "trains only the biases, the Adapter and AdaptFormer add a small bottleneck after or beside each MLP, and LoRA learns "
        "a low-rank update that is merged back into the weights. They train 0.12 to 1.4% of the backbone.\n\nAll seven share "
        "one protocol: the same head, 3 epochs, the same holdout and the same GPU, so their numbers compare directly.\n\n"
        f"Handoff: {B} will show how we worked through these experiments, and how the seven compare.")))

    # 4. How we iterated: one question per round, failed ideas included (B)
    el = []
    ct = header(el, "Part 2 · Our process", "How we iterated: one question at a time")
    tb = table(LEFT, ct, [62, 214, 266, 290], [
        ["When", "Question", "What we tried", "What we found (holdout top-1)"],
        ["Sep 28", "Is our pipeline right?", "Reproduce the three reference baselines", "All three within 0.3 points"],
        ["Sep 28", "Do longer runs help?", "Full FT for 8 epochs, LoRA for 10", "Full +1.4; LoRA +0.1, and its loss rose in "
                                                                                  "epoch 1"],
        ["Oct 18", "Is LoRA's learning rate too high?", "Halve it: 1e-3 → 5e-4", "Failed: −0.12, so not the cause"],
        ["Oct 18", "Does LoRA lack capacity?", "Also put LoRA on proj, fc1 and fc2", "Worked: +0.88"],
        ["Oct 19", "Is the head the problem?", "Linear-probe vs random vs prototype head", "Worked: +2.5; the linear-probe "
                                                                                          "head hurts"],
        ["Oct 19", "What is the best final recipe?", "Prototype head; full FT 8 epochs; LoRA 3 vs 6", "Val 76.22% and 71.90%; "
                                                                                                    "6 LoRA epochs: −0.46"],
        ["Nov 2", "Which method is best per weight?", "Seven methods under one protocol", "Where you train beats how much"],
        ["Nov 5", "What limits us now? (after the deadline)", "In21k-only backbones; larger input images",
         "In21k-only backbones probe worse (61.05%, 51.07% vs 61.94%); resolution: +4.4"],
    ], ["left"] * 4, size=13, highlight=(5,))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 14, CONTENT_W, [[("Failed ideas mattered too: ", {"bold": True, "color": INK}),
                                                           ("each one ruled out an explanation and pointed to the next test.",
                                                            {})]], size=15, color=INK2, line=1.3))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="B", notes=(
        "Here is how we got there: one question per round, and each answer chose the next experiment. After reproducing the "
        "baselines, longer training helped full fine-tuning but not LoRA, and LoRA's training loss even rose in the first "
        "epoch. Our first guess, a too-high learning rate, failed: halving it changed nothing. Our second guess, missing "
        "capacity, helped: adding LoRA to the MLP gave 0.9 points. The third question, the head, was the big one: 2.5 "
        "points, as Part 3 explains. Then we fixed the final recipe, compared all seven methods under one protocol, and "
        "after the deadline asked what still limits us: not the backbone, but the input resolution. The failed ideas "
        "mattered as much as the successes: each one ruled out a wrong explanation.")))

    # 5. Comparison table and accuracy against parameters (B)
    el = []
    ct = header(el, "Part 2 · Comparison", "Spread small changes over every block")
    tb = table(LEFT, ct, [140, 62, 78, 76, 82], [
        ["Method", "Trains", "Holdout", "Train memory", "Latency, batch 1"],
        ["Linear probe", "0%", "61.94%", "3.2 GB", "6.96 ms"],
        ["Partial, K = 4", "33%", "71.88%", "7.0 GB", "7.41 ms"],
        ["Full fine-tuning", "100%", "73.38%", "18.9 GB", "7.78 ms"],
        ["BitFit", "0.12%", "69.81%", "12.7 GB", "7.33 ms"],
        ["Adapter", "1.39%", "71.54%", "12.8 GB", "9.74 ms"],
        ["AdaptFormer", "1.39%", "71.17%", "12.8 GB", "10.23 ms"],
        ["LoRA", "1.38%", "71.98%", "19.9 GB", "7.61 ms"],
    ], ["left", "right", "right", "right", "right"], size=14, highlight=(3, 7), bold_rows=(3, 7))
    el.append(tb)
    fx = LEFT + tb["w"] + 24
    img = image("fig7_acc_params.png", fx, ct - 6, LEFT + CONTENT_W - fx,
                "Holdout top-1 against trainable backbone weights on a log axis. Grey line, last K blocks from the top: 67.97% "
                "at 8.3% (K = 1), 70.99% (K = 2), 71.88% (K = 4), 72.89% at 49.6% (K = 6), 73.38% at 100%. Blue points, small "
                "changes in every block: BitFit 69.81% at 0.12%; LoRA 71.98%, Adapter 71.54% and AdaptFormer 71.17% near "
                "1.4%. The linear probe, 61.94%, is below the range.")
    el.append(img)
    el.append(T(LEFT, max(tb["y"] + tb["h"], img["y"] + img["h"]) + 12, CONTENT_W, [
        "Same protocol: LoRA (1.38%) matches partial K=4 (33%); BitFit (0.12%) beats K=1 (8%).",
        "Memory follows how far gradients flow, not how many weights train: LoRA 19.9 GB, partial K=4 only 7.0 GB.",
        "Full fine-tuning is the most accurate; LoRA is the best efficient method and merges, so no latency is added.",
    ], size=15, color=INK2, line=1.3, after=6, bullet=16))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="B", notes=(
        "Here are all seven under the same protocol; the plot shows accuracy against the share of "
        "trained weights, on a log scale.\n\nThe grey line trains whole blocks from the top, and it climbs slowly: one block, "
        "8% of the weights, gives 68%; six blocks give 72.9%; everything gives 73.4%. The blue points make small changes in "
        "every block, and they sit far to the left of the line. LoRA, with 1.38% of the weights, matches training the last "
        "four blocks, which is 33%. BitFit, with only the biases, beats training the last block. So where the trainable "
        "weights sit matters more than how many there are: a little capacity in every block beats a lot in a few.\n\nMemory "
        "tells the same story. LoRA trains 1.38% of the weights yet needs 19.9 GB, more than full fine-tuning, while the last "
        "four blocks need only 7 GB. Memory follows how far back the gradients must flow, not how many weights are trained."
        "\n\nFull fine-tuning is still the most accurate. Among the efficient methods LoRA is best, and after training it merges into "
        "the weights, so it adds no latency, while the two adapters add 2 to 3 milliseconds per photo.")))

    # 6. The problem: the linear-probe head starts high and ends low (B)
    el = []
    ct = header(el, "Part 2 · A problem we found", "Linear-probe heads start high, finish low")
    img = image("fig2_head_init_holdout.png", (W - 600) / 2, ct - 4, 600,
                "Line charts of holdout top-1 accuracy by epoch. Full fine-tuning: the linear-probe head goes from 62.1% "
                "before training to 70.9% after 3 epochs, the prototype head from 44.0% to 73.4%. LoRA on all linear layers: "
                "linear-probe head 62.1% to 69.6%, prototype head 44.0% to 72.0%.")
    el.append(img)
    el.append(T(LEFT, img["y"] + img["h"] + 10, CONTENT_W, lead_paras([
        ("Recommended recipe, LP-FT (Kumar et al., ICLR 2022):", "fit a linear-probe head on frozen features, then fine-tune."),
        ("Starts high, ends low:", "18 points ahead before training, 2.5 points behind after it."),
        ("First suspect:", "LoRA's learning rate. Halving it changed nothing (−0.12), so we tested the head."),
    ]), size=15, color=INK2, line=1.3, after=6, bullet=16))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="B", notes=(
        "Now a closer look at the head question from the timeline. Our first runs used a recommended two-step recipe, "
        "LP-FT, from Kumar and colleagues at ICLR 2022: first fit a "
        "linear-probe head on frozen features, then fine-tune. The leftmost point of each panel is the head alone: the "
        "linear-probe head starts at 62%, far above the 44% of a prototype head. But after three epochs it is behind: 70.9 "
        "against 73.4 for full fine-tuning, and 69.6 against 72.0 for LoRA.\n\nThe first symptom was LoRA's training loss "
        "going up in the first epoch, so our first suspect was the learning rate. Halving it changed nothing, minus 0.12, "
        "which is noise. So we tested the head instead: a prototype head, and even a random head, beat the linear-probe head "
        f"by about 2.5 points.\n\nHandoff: {C} will explain our fix and why it works.")))

    # 7. Our method: the prototype head (C)
    el = []
    ct = header(el, "Part 3 · Our method", "Our fix: start from class prototypes")
    bottom = flow(el, ct + 4, [
        ("1", "Extract features", "Run the frozen ViT once over all 450,000 training photos (about 6 min).", CARD, INK),
        ("2", "Prototype head", "Class weight = normalized mean feature × 40, as in weight imprinting (Qi et al., CVPR 2018).",
         CARD_BLUE, BLUE),
        ("3", "Fine-tune", "Full fine-tuning or LoRA, starting from this head.", CARD, INK),
        ("4", "Predict", "Full model: add the logits of each photo and its mirror image.", CARD, INK)])
    big = dict(family="Georgia", size=32, bold=True, color=BLUE, line=1.05, gap=6)
    lab = dict(size=14, color=INK2, line=1.3)
    card_row(el, bottom + 18, [
        [dict(paras=["0"], **big), dict(paras=["gradient steps to build the head: it never fits a training photo, so "
                                                "it has nothing to memorize"], **lab)],
        [dict(paras=["44%"], **big), dict(paras=["holdout top-1 before any training, because each logit = 40 × "
                                                  "cosine(photo, class mean)"], **lab)],
        [dict(paras=["4.39"], **big), dict(paras=["starting loss, vs 2.29 for the linear-probe head: a strong signal "
                                                   "that makes the backbone adapt"], **lab)],
    ], [CARD_BLUE, CARD, CARD], gap=20, pad=18)
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="C", notes=(
        "Our fix replaces the linear probe with class prototypes. Step one: run the frozen backbone once over all training "
        "photos and keep each 768-dimensional feature; this takes about six minutes and counts toward our GPU budget. Step "
        "two, the key step: the weight of each class is the normalized mean feature of its photos, times a temperature of "
        "40. The idea itself is weight imprinting, from Qi and colleagues at CVPR 2018; what is new is our finding that it "
        "beats the LP-FT start here. There are no gradient steps, so the head never fits any training photo. Step three: fine-tune with any method; "
        "we keep full fine-tuning and LoRA. Step four: the full model adds the logits of each photo and its mirror image.\n\n"
        "Why is this a good start? Each logit is about 40 times the cosine between a photo and its class mean, so the head "
        "already reaches 44% before any training, with calibrated scores. And its starting loss is 4.39, against 2.29 for the "
        "linear-probe head, so the backbone gets a strong learning signal from the first step. The next slide shows what "
        "that difference does during training.")))

    # 8. Why it works: the loss evidence (C)
    el = []
    ct = header(el, "Part 3 · Why it works", "Lower training loss, yet lower accuracy")
    img = image("fig10_loss_slide.png", (W - 640) / 2, ct - 6, 640,
                "Training loss of full fine-tuning on all data for 8 epochs. The linear-probe head (orange) starts at 2.8 and "
                "ends at 1.77; the prototype head (blue) starts above 5 and ends at 1.85. CodaBench val: 73.10% with the "
                "linear-probe head, 76.22% with the prototype head.")
    el.append(img)
    bottom = flow(el, img["y"] + img["h"] + 10, [
        ("1", "Fit on the same photos", "The linear-probe head is trained on frozen features of the training photos.",
         CARD, ORANGE),
        ("2", "Weak, then wrong signal", "Low starting loss; as features move, the head stops fitting (LoRA loss 2.2 → 2.7).",
         CARD, ORANGE),
        ("3", "Memorization", "Lower training loss (1.77 vs 1.85), lower val (73.10% vs 76.22%).", CARD, ORANGE)],
         gap=30, pad=14, body_size=14)
    el.append(T(LEFT, bottom + 8, CONTENT_W, [[("Control: ", {"bold": True, "color": INK}),
                                                ("a random head ties the prototype head (71.95% vs 72.00%), so fitting the head "
                                                 "on the training photos is what hurts.", {})]], size=14, color=INK2, line=1.3))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="C", notes=(
        "Here is the evidence from two final-recipe runs: full fine-tuning on all data for 8 epochs, identical except for the "
        "head. The orange linear-probe run has the lower training loss the whole time and ends at 1.77, against 1.85 for the "
        "blue run. Yet on CodaBench val it scores 73.10, against 76.22. Lower training loss but lower accuracy: that is "
        "memorization.\n\nOur explanation has three steps. One: the linear-probe head was fitted to frozen features of the very "
        "same training photos. Two: its starting loss is low, so the backbone gets a weak signal; and as the features start to "
        "move, the head no longer fits them, which is why LoRA's loss rose in the first epoch. Three: the model ends up "
        "fitting the 45 photos per class instead of features that generalize.\n\nA control supports this: a random head ends "
        "level with the prototype head, 71.95 against 72.00. So prototypes are not magic; the harm comes from fitting the head "
        "on the training photos.")))

    # 9. Every change, one at a time, and why (D)
    el = []
    ct = header(el, "Part 4 · Ablations", "Each change, tested one at a time")
    el.append(image("fig9_changes_why.png", LEFT, ct - 4, CONTENT_W,
                    "Horizontal bars of the change in top-1 points, one setting at a time, with the reason. Head: linear-probe to "
                    "prototype head, full fine-tuning 8 epochs, val 73.10 to 76.22, +3.12, the head no longer memorizes the "
                    "training photos. LoRA, holdout, 3 epochs: learning rate 1e-3 to 5e-4, 68.57 to 68.45, −0.12, within noise; "
                    "qkv only to all four linear layers, 68.57 to 69.45, +0.88, the MLP holds 2/3 of each block's weights; 3 to "
                    "6 epochs, 71.98 to 71.52, −0.46, training loss 1.95 to 1.70 yet holdout falls. Full fine-tuning, CodaBench val: 3 to 8 epochs, "
                    "74.01 to 76.22, +2.21, still improving at epoch 3 and 8 epochs fit the budget; flip TTA, 72.57 to 72.84, "
                    "+0.27; 43 to 45 photos per class, 72.84 to 73.10, +0.26."))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="C", notes=(
        "Each row changes one setting of our final recipe and keeps the rest "
        "fixed; the right column gives the reason. The head is the biggest change, plus 3.1 points on val, as we just saw. "
        "For LoRA, halving the learning rate did nothing, while adding LoRA to all four linear layers gave plus 0.9: two "
        "thirds of each block's weights sit in the MLP. Training LoRA longer hurt: its training loss fell from 1.95 to 1.70 "
        "while the holdout fell, so it overfits, and we stop at 3 epochs. Full fine-tuning behaves the other way: 3 to 8 "
        "epochs gained 2.2 points, and 8 epochs fit our budget. Flip test-time augmentation and all 45 photos per class add "
        "about a quarter point each.\n\n"
        f"Handoff: {D} will show our two final models and their results.")))

    # 10–11. The two final models, one slide each: the architecture, then the exact setup in three cards (D)
    # settings from the runs' config.json and summary.json: 1019_1319_1_full and 1019_1319_3_lora
    for eyebrow, title, diagram, cards, notes in [
            ("Part 4 · Final model 1 of 2", "Full fine-tuning trains every weight", "full", [
                ("Model", ["ViT-B/16: 85.8M weights, all trained", "Head: 768 → 10,000, 7.69M weights",
                           "Head starts from class prototypes × 40", "All 45 photos per class, 224 px",
                           "Random crop (25–100%) + flip"]),
                ("Training", ["Loss: cross-entropy, label smoothing 0.1", "AdamW, weight decay 0.05, clip 1.0",
                              "LR 1e-4, × 0.75 per block to the input", "Drop path 0.1, bf16, batch 256",
                              "8 epochs: 1 warmup, then cosine to 0"]),
                ("Cost and result", ["1.03 GPU-hours on one A100", "18.9 GB peak, 14,056 steps",
                                     "Flip TTA: 14.95 ms, 1,919 photos/s", "Val 76.22%, test 76.09%"]),
            ], "Before the numbers, here are our two final models; blue means trained, grey frozen. The first is full "
               "fine-tuning of ViT-B/16: all 85.8 million weights plus the new 7.69 million head, which starts from the class "
               "prototypes. We train with cross-entropy and label smoothing 0.1, and AdamW at a learning rate of 1e-4 that "
               "shrinks by 0.75 per block toward the input, so the first block gets only 3% of it. Drop path 0.1, 8 epochs on "
               "all 45 photos per class, 1.03 GPU-hours."),
            ("Part 4 · Final model 2 of 2", "LoRA trains 1.38%, then merges", "lora", [
                ("Model", ["ViT-B/16: all 85.8M weights frozen", "LoRA r = 8, α = 16 on qkv, proj, fc1, fc2",
                           "1,179,648 LoRA weights (1.375%)", "Head: 7.69M, from prototypes × 40",
                           "43 photos per class (2 held out), 224 px"]),
                ("Training", ["Loss: cross-entropy, label smoothing 0.1", "AdamW, weight decay 1e-4, clip 1.0",
                              "LR 1e-3, no layer decay, no drop path", "bf16, batch 256, random crop + flip",
                              "3 epochs: 0.3 warmup, then cosine to 0"]),
                ("Cost and result", ["0.54 GPU-hours on one A100", "19.9 GB peak, 5,037 steps",
                                     "Merged: 7.61 ms, 3,869 photos/s", "Val 71.90%, test 71.67%"]),
            ], "The second is LoRA, shown inside one block. The backbone stays frozen. Next to qkv, proj, fc1 and fc2 in every "
               "block, rank-8 matrices A and B learn the update, 1.18 million weights in total, and the head is trained too. "
               "Same loss; the learning rate is 1e-3 without layer decay, for 3 epochs, because longer training overfits. It "
               "took 0.54 GPU-hours, and after training the update is merged, so inference is as fast as the base model.")]:
        el = []
        ct = header(el, eyebrow, title)
        img = image(f"diagrams/method_{diagram}.png", (W - 780) / 2, ct - 6, 780, DIAGRAM_ALT[diagram])
        el.append(img)
        card_row(el, img["y"] + img["h"] + 8,
                 [[dict(paras=[head], size=14, bold=True, color=BLUE, line=1.2, gap=6),
                   dict(paras=items, size=13, color=INK2, line=1.25, after=3)] for head, items in cards],
                 [CARD_BLUE, CARD, CARD], gap=14, pad=11, radius=8)
        footer(el, len(slides) + 1)
        slides.append(dict(bg=PAGE, el=el, speaker="D", notes=notes))

    # 12. Final results on every metric (D)
    el = []
    ct = header(el, "Part 4 · Final models", "Final results on every metric")
    tb = table(LEFT, ct, [262, 104, 70, 70, 80, 92, 154], [
        ["Model", "Trainable", "Val", "Test", "GPU-hours", "Peak memory", "Latency, batch 1, A100"],
        ["Project brief reference: full fine-tuning", "100%", "70.8%", "–", "–", "–", "–"],
        ["Ours: full FT, prototype head, 8 epochs, all data, flip TTA", "85.8M (100%)", "76.22%", TEST_FULL, "1.03",
         "18.9 GB", "14.95 ms (2 passes)"],
        ["Project brief reference: LoRA r = 8 on qkv", "0.34%", "68.3%", "–", "–", "–", "–"],
        ["Ours: LoRA on all linear layers, prototype head, 3 epochs", "1.18M (1.375%)", "71.90%", TEST_LORA, "0.54",
         "19.9 GB", "7.61 ms (merged)"],
    ], ["left"] + ["right"] * 6, size=13, highlight=(2, 4), bold_rows=(2, 4))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 22, CONTENT_W, lead_paras([
        ("Held up on test:", "within 0.25 points of val for both models."),
        ("Inside the rules:", "both fit the 1.2-hour budget; both test files come from our Phase 1 models."),
        ("Recipe:", "AdamW, cosine, bf16, batch 256; full FT at LR 1e-4, layer decay 0.75; LoRA r = 8 at 1e-3."),
        ("Fair comparison:", "the like-for-like test is slide 5; here each model got its best setting."),
    ]), size=15, color=INK2, line=1.35, after=10, bullet=16))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="D", notes=(
        "Here are their results on every metric the brief asks for. Full fine-tuning reaches 76.22% on val and "
        f"{TEST_FULL} on the hidden test set, 5.4 points above the reference, in 1.03 GPU-hours with 18.9 GB. LoRA trains "
        f"1.375% of the backbone and reaches 71.90% on val and {TEST_LORA} on test, 3.6 points above the reference LoRA, in "
        "0.54 GPU-hours. For both, test is within 0.25 points of val, both "
        "fit the budget, and both test files come from our Phase 1 models. The like-for-like comparison is the one "
        f"{B} showed, 73.38 against 71.98 under one protocol; here each model got its best setting.")))

    # 13. After the deadline: input resolution, as one table (D). Runs: 1019_1120_1_full (baseline and its 288 px test),
    # 1105_1031_1_full_288 (trained at 288 px), 1019_1319_1_full (final model), 1105_1032_1_full_288 (final recipe at
    # 288 px)
    el = []
    ct = header(el, "Part 4 · After the deadline", "Input resolution: the largest gain we found")
    intro = T(LEFT, ct - 4, CONTENT_W, [[("Optimized model: ", {"bold": True, "color": INK}),
                                         ("full fine-tuning of ViT-B/16 with the prototype head. The trained weights stay the "
                                          "same (85.8M backbone + 7.69M head); only the image size changes, and the position "
                                          "embeddings are interpolated, so no parameter is added.", {})]],
              size=14, color=INK2, line=1.3)
    el.append(intro)
    tb = table(LEFT, intro["y"] + intro["h"] + 10, [290, 80, 80, 190, 100, 92], [
        ["Run", "Train size", "Test size", "Top-1", "GPU-hours", "Memory"],
        ["3 epochs · holdout", "", "", "", "", ""],
        ["Baseline", "224 px", "224 px", "73.38%", "0.48", "18.9 GB"],
        ["Same model, larger test images", "224 px", "288 px", "75.50% (+2.1)", "0.48", "18.9 GB"],
        ["Trained at 288 px", "288 px", "288 px", "77.23% (+3.9)", "0.68", "30.2 GB"],
        ["Trained at 288 px, best test size", "288 px", "320 px", "77.78% (+4.4)", "0.68", "30.2 GB"],
        ["8 epochs · all data · flip TTA", "", "", "", "", ""],
        ["Our final model", "224 px", "224 px", "val 76.22%, test 76.09%", "1.03", "18.9 GB"],
        ["Same recipe at 288 px", "288 px", "320 px", "no test score", "1.61*", "30.2 GB"],
    ], ["left", "right", "right", "right", "right", "right"], size=13, highlight=(5,), bold_rows=(1, 6))
    el.append(tb)
    el.append(T(LEFT, tb["y"] + tb["h"] + 12, CONTENT_W, lead_paras([
        ("Why it helps:", "random crops enlarge objects in training, so larger test images match them (FixRes)."),
        ("Not in our results:", "found after the deadline, and the 8-epoch run (1.61*) is over the 1.2-hour budget."),
    ]), size=15, color=INK2, line=1.3, after=6, bullet=16))
    footer(el, len(slides) + 1)
    slides.append(dict(bg=PAGE, el=el, speaker="D", notes=(
        "One more lesson from after the deadline, on the same full fine-tuning model with the prototype head: only the image "
        "size changes. With the 3-epoch protocol, simply testing our "
        "224-pixel model at 288 pixels adds 2.1 points. Training at 288 as well and testing at 320 gives 77.78%, 4.4 points "
        "above the baseline, in 0.68 GPU-hours, inside the budget, but it needs 30 GB of memory instead of 19. The same "
        "final recipe at 288 pixels would take 1.61 GPU-hours, over the budget, so it "
        "is not part of our results and has no test score.")))

    # 14. Takeaways and close (D)
    el = []
    el.append(T(LEFT, 56, CONTENT_W, ["Takeaways"], family="Georgia", size=36, bold=True, color=DARK_TEXT, line=1.12,
                role="title"))
    y = 120
    for i, (text, color) in enumerate([
            ("New finding: the usual linear-probe start (LP-FT) hurt all three methods we tested; class prototypes added "
             "2.1 to 3.1 points.", DARK_BLUE),
            ("Where you train beats how much: LoRA at 1.38% matches partial fine-tuning at 33%.", DARK_BLUE),
            ("Failed ideas were useful: a lower learning rate changed nothing, and longer LoRA training hurt.", DARK_BLUE),
            ("Limits: one run per setting (holdout noise about ±0.3 points); the efficient methods share one untuned "
             "learning rate.", DARK_ORANGE),
            ("After the deadline: larger input images added 4.4 holdout points. Next: trade epochs for image size within "
             "the budget.", DARK_ORANGE)], 1):
        el.append(T(LEFT, y - 4, 40, [str(i)], family="Georgia", size=26, bold=True, color=color, line=1.1))
        t = T(LEFT + 50, y, CONTENT_W - 50, [text], size=18, color=DARK_TEXT, line=1.3, role="point")
        el.append(t)
        y += t["h"] + 18
    footer(el, len(slides) + 1, dark=True)
    slides.append(dict(bg=DARK, el=el, speaker="D", notes=(
        "To sum up. One, our main new finding: the usual recipe of starting from a linear probe hurt all three methods we "
        "tried, "
        "while a head built from class prototypes added 2 to 3 points. Two: where the trainable weights sit matters more "
        "than how many; LoRA at 1.4% matches training a third of the network. Three: the failed ideas were useful; each one "
        "ruled out an explanation. Four, our limits: each setting was trained once, and differences below about 0.3 points "
        "are within holdout noise; and the efficient methods share one learning rate that we did not tune per method. And "
        "five: after the deadline we found that larger input images add 4.4 points on the holdout; next, we would trade some "
        "epochs for larger images within the budget.")))
    slides[-1]["qa"] = (
        "Q&A routing (not spoken):\n"
        f"- Why prototype beats lp, and the evidence -> {C}. Slides 6 to 8: lp starts 62% but ends 2.5 lower; lower train loss "
        "(1.77 vs 1.85) but lower val (73.10 vs 76.22); random head ties proto (71.95 vs 72.00), so fitting the head on the "
        "training photos is what hurts. Is the prototype head new? No: it is weight imprinting (Qi et al., CVPR 2018); what "
        "is new is the finding that it beats the recommended LP-FT start (Kumar et al., ICLR 2022) with 10,000 classes and "
        "45 photos each. Why 40? Chosen from 5, 10, 15, 20, 30, 40, 60, 80 by the cross-entropy on 10,000 training "
        "features; no gradient steps. Doesn't LP-FT say a random head distorts the features? It does, to protect "
        "robustness on new domains; here the features must change, and for LoRA a random head ties the prototype head. "
        "Slide 5 says 71.98 and slide 8 says 72.00: the same setting run twice, 72.00 on an RTX PRO 6000 and 71.98 on the "
        "A100.\n"
        f"- Why PEFT lost to full fine-tuning; fairness of the final comparison -> {D}. 450,000 photos in total is plenty for "
        "full fine-tuning (few per class, many overall); 10,000 fine-grained classes need changes in every layer; same "
        "protocol gap is 73.38 vs 71.98. LoRA got worse with more epochs (-0.46), full got better (+2.21), so each final model "
        "has its best setting; LR 1e-3 was not tuned per efficient method (limitation). LoRA final model: no TTA, 43 photos "
        "per class. Why does LoRA get worse after 3 epochs while full FT keeps improving? Our hypothesis, not yet tested: "
        "7.69M of LoRA's 8.87M trainable weights are the head, and the LoRA recipe gives it LR 1e-3 (full FT: 1e-4), weight "
        "decay 1e-4 (full FT: 0.05) and no drop path (full FT: 0.1), so the head memorizes the 45 photos per class, as in "
        "Part 3. Next test: 6 LoRA epochs with a head LR of 1e-4 (notebook methods/10).\n"
        f"- Method details, parameters, latency, LoRA vs adapters -> {B}. Slides 3, 5, 10 and 11; backup slides 16 to 20 show "
        "the other five methods. LoRA 12 x 8 x (3,072 + 1,536 + 3,840 + "
        "3,840) = 1,179,648; adapters 1,189,632. LoRA merges (no latency) but trains slowest (888 img/s, 19.9 GB); adapters "
        "train 1.6-1.7x faster with 36% less memory but add 2.4-2.9 ms at batch 1. Why LoRA needs more memory than full FT "
        "(19.9 vs 18.9 GB): to train its A matrices it must keep every linear layer's input, as full FT does, plus the "
        "activations of its 96 extra small matrix multiplications; the optimizer state that full FT adds is only about 1 GB. "
        "Partial K = 4 keeps activations for its 4 blocks only (7.0 GB). Other LoRA ranks? Not tried; r = 8 as in the brief, "
        "and we added capacity by placement instead (qkv only to all four linear layers: +0.88). Is LoRA really better than "
        "the Adapter (71.98 vs 71.54)? Only by 0.44, close to the noise; its clear advantage is merging (7.61 vs 9.74 ms at "
        "batch 1).\n"
        f"- Budget, GPU-hours, base model, data and holdout -> {A}. Slides 2, 4 and 12. 96 GB-h / 80 GB = 1.2 h (1.32 with 10%); "
        "counted: head features, training, holdout evaluation. Base model from page 1 of the brief; in21k-only checkpoints "
        "have a weaker linear probe (61.05% and 51.07% vs 61.94%), and our baselines match the reference within 0.3, so "
        "we kept the recommended backbone. Holdout: 2 per class, seed 0.\n"
        f"- Resolution over budget? -> {D}. Slide 13: the 3-epoch 288 px run (+4.4 holdout) used 0.68 GPU-hours, inside the "
        "budget; only the full 8-epoch recipe at 288 px (1.61 h) is over, and it has no test score.\n"
        f"- Limits on slide 14 -> {D}. One seed per setting. 0.3 points is about the standard error of accuracy on 20,000 "
        "holdout photos (0.32 at 72%), and the holdout matched CodaBench (72.31 vs 72.57 full, 71.98 vs 71.90 LoRA). The "
        "efficient methods all used LR 1e-3; only LoRA's was checked (5e-4: -0.12).")

    # 15. Thank you; D leads the questions (the Q&A routing is in its notes)
    el = []
    el.append(T(LEFT, 64, 700, ["CS604 · Team 06"], size=14, bold=True, color=EYEBROW, spacing=2, caps=True))
    for i, c in enumerate([GRAY, BLUE, ORANGE]):
        el.append(oval(852 + i * 16, 66, 10, c))
    t = T(LEFT, 196, CONTENT_W, ["Thank you"], family="Georgia", size=60, bold=True, color=INK, line=1.05, role="title")
    el.append(t)
    el.append(T(LEFT, t["y"] + t["h"] + 18, CONTENT_W, ["Questions are welcome."], size=22, color=INK2, line=1.3))
    el.append(T(LEFT, 468, CONTENT_W, [" · ".join(NAMES)], size=14, color=INK2, pinned=True))
    slides.append(dict(bg=CARD, el=el, big_title=True, speaker="D", notes="Thank you; we are happy to take questions.",
                       qa=slides[-1].pop("qa")))

    # 16–20. Backup for questions: the other five methods, rebuilt from method slides 1, 2, 4, 5 and 6 of the first deck
    # (full fine-tuning and LoRA are slides 10 and 11). Same content; this deck's footer, and no handoff lines in the notes.
    backup = {f"METHOD {i} OF 7" for i in (1, 2, 4, 5, 6)}
    for s in v1_slides():
        eyebrow = next(("".join(t for t, _ in e["paras"][0]) for e in s["el"] if e["kind"] == "text"), "")
        if eyebrow in backup:
            el = [e for e in s["el"] if not e.get("footer")]
            footer(el, len(slides) + 1)
            notes = "\n\n".join(p for p in s["notes"].split("\n\n") if not p.startswith("Handoff:"))
            slides.append(dict(bg=s["bg"], el=el, speaker="Backup", notes=notes))

    return slides


def timing(slides):
    """Speaking time per speaker from the notes (handoffs included, Q&A routing excluded), at WPM words per minute."""
    totals = {}
    for n, s in enumerate(slides, 1):
        words = len(s["notes"].split())
        totals.setdefault(s["speaker"], [0, []])
        totals[s["speaker"]][0] += words
        totals[s["speaker"]][1].append(n)
    for who, (words, nums) in totals.items():
        if not who.startswith("Backup"):
            m = words / WPM
            print(f"  speaker {who}: slides {nums[0]}–{nums[-1]}, {words} words ≈ {int(m)}:{round(m % 1 * 60):02d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="directory for slide-NN.png previews")
    args = ap.parse_args()
    slides = build_slides()
    problems = check(slides)
    print(f"{len(slides)} slides; layout problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    timing(slides)
    for s in slides:   # speaker label first, Q&A routing last
        s["notes"] = (f"Speaker {s['speaker']}.\n\n" if not s["speaker"].startswith("Backup") else "") + s["notes"] + (
            "\n\n" + s["qa"] if s.get("qa") else "")
    write_pptx(slides, OUT)
    print("wrote", OUT)
    if args.preview:
        for p in render(slides, args.preview):
            print(p)


if __name__ == "__main__":
    main()

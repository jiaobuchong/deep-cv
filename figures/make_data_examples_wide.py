#!/usr/bin/env python3
"""Wide version of the training-photo grid for the ICLR report (single column).

Same 4 classes and photos as make_data_examples.py, laid out as 2 rows x (4 + 4) photos,
with a wider gap between classes. Reads cs604-course-data/train/; writes
report_iclr/figures/data_examples_wide.jpg."""
import os

from PIL import Image, ImageOps

from make_data_examples import PICKS, TRAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_iclr", "figures", "data_examples_wide.jpg")
TILE, GAP, CLASS_GAP, BG = 230, 6, 26, (255, 255, 255)


def main():
    classes = list(PICKS.items())
    width = 8 * TILE + 6 * GAP + CLASS_GAP
    height = 2 * TILE + CLASS_GAP
    sheet = Image.new("RGB", (width, height), BG)
    for k, (cls, picks) in enumerate(classes):
        row, half = divmod(k, 2)
        x0 = half * (4 * TILE + 3 * GAP + CLASS_GAP)
        y0 = row * (TILE + CLASS_GAP)
        files = sorted(os.listdir(os.path.join(TRAIN, cls)))
        for c, i in enumerate(picks):
            im = Image.open(os.path.join(TRAIN, cls, files[i])).convert("RGB")
            im = ImageOps.fit(im, (TILE, TILE), Image.LANCZOS)
            sheet.paste(im, (x0 + c * (TILE + GAP), y0))
    sheet.save(OUT, quality=90)
    print("wrote", OUT, sheet.size)


if __name__ == "__main__":
    main()

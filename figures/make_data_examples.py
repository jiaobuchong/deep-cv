#!/usr/bin/env python3
"""Example training photos for the task slide: 4 of the 10,000 classes (rows), 4 of their 45 photos each.
The photos are picked to show how much one class varies (distance, background, whole plant vs flower).

Reads cs604-course-data/train/ (the course data, unpacked locally); writes figures/data_examples.jpg."""
import os

from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(os.path.dirname(HERE), "cs604-course-data", "train")
OUT = os.path.join(HERE, "data_examples.jpg")

PICKS = {  # class folder -> photos, by position in the sorted file list
    "class_3867": (0, 2, 4, 5),   # a bird: through a scope, a flock, at a feeder, on reeds
    "class_4969": (0, 2, 4, 5),   # a snake: on rock, on pavement, on gravel, in leaves
    "class_1690": (0, 2, 3, 4),   # a moth: tiny on a wall, on a leaf, close-up, on fabric
    "class_7845": (1, 2, 4, 5),   # a plant: flowers, fruit, the whole shrub, a branch
}
TILE, GAP, BG = 240, 8, (250, 249, 246)   # BG = the slide background (PAGE)


def main():
    rows = len(PICKS)
    side = 4 * TILE + 3 * GAP
    sheet = Image.new("RGB", (side, rows * TILE + (rows - 1) * GAP), BG)
    for r, (cls, picks) in enumerate(PICKS.items()):
        files = sorted(os.listdir(os.path.join(TRAIN, cls)))
        for c, i in enumerate(picks):
            im = Image.open(os.path.join(TRAIN, cls, files[i])).convert("RGB")
            im = ImageOps.fit(im, (TILE, TILE), Image.LANCZOS)          # center crop to a square
            sheet.paste(im, (c * (TILE + GAP), r * (TILE + GAP)))
    sheet.save(OUT, quality=88)
    print("wrote", OUT, sheet.size)


if __name__ == "__main__":
    main()

"""
Generate overlay images for the paintings where no important figure made the
top 3 (Hit@3 = False). These are the genuine failure cases for the locked-in
pipeline and the source material for the qualitative discussion in §6.

Outputs one PNG per painting under diagnostic/hit3_misses/, showing every
detected figure with its mask, its rank by combo_saliency_heavy, and whether
it was annotated as important.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HOME = Path.home()
RUN_DIR = HOME / "thesis_models" / "full-run"
OUTPUTS = RUN_DIR / "outputs"
IMAGES_DIR = OUTPUTS / "images_subset"
SAM3_ROOT = OUTPUTS / "sam3"
OUT_DIR = RUN_DIR / "diagnostic" / "hit3_misses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(OUTPUTS / "all_features_with_composite.csv")
features["painting_id"] = features["painting_id"].astype(str)

# Find paintings where no important figure landed in the top 3
miss_paintings = []
for pid, g in features.groupby("painting_id"):
    if not g["is_important"].any():
        continue
    g_sorted = g.sort_values("combo_saliency_heavy", ascending=False).reset_index(drop=True)
    top3_important = g_sorted.head(3)["is_important"].any()
    if not top3_important:
        miss_paintings.append(pid)

print(f"Hit@3 misses: {len(miss_paintings)} paintings")
print(f"Saving overlays to {OUT_DIR}\n")


def load_image(painting_id):
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMAGES_DIR / f"{painting_id}{ext}"
        if p.exists():
            return np.array(Image.open(p).convert("RGB"))
    raise FileNotFoundError(f"No image found for {painting_id}")


def overlay(painting_id):
    img = load_image(painting_id)
    H, W = img.shape[:2]
    pf = features[features["painting_id"] == painting_id].copy()
    pf = pf.sort_values("combo_saliency_heavy", ascending=False).reset_index(drop=True)
    pf["rank"] = range(1, len(pf) + 1)

    fig, ax = plt.subplots(figsize=(14, 14 * H / W))
    ax.imshow(img)

    for _, row in pf.iterrows():
        x1, y1, x2, y2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
        is_imp = bool(row["is_important"])
        color = (1.0, 0.2, 0.2) if is_imp else (0.2, 0.5, 1.0)

        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label = f"#{int(row['rank'])}  score={row['combo_saliency_heavy']:.2f}"
        if is_imp:
            label += "  [IMP]"

        ax.text(row["centroid_x"], y1 - 10, label,
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=color, edgecolor='none', pad=2))

    best_imp_rank = int(pf[pf["is_important"]]["rank"].min())
    n_imp = int(pf["is_important"].sum())
    ax.set_title(
        f"{painting_id}\n"
        f"{n_imp} important figure(s), best ranked at #{best_imp_rank}  |  "
        f"red = important, blue = non-important",
        fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    out_path = OUT_DIR / f"{painting_id}.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


for i, pid in enumerate(miss_paintings, 1):
    try:
        overlay(pid)
        print(f"  [{i:3d}/{len(miss_paintings)}] {pid}")
    except Exception as exc:
        print(f"  [{i:3d}/{len(miss_paintings)}] {pid} -- FAILED: {exc}")

print(f"\nDone. Open: {OUT_DIR}")
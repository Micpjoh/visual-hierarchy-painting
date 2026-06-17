from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

# FORCE Python to use sam-hq's segment_anything first
SAM_HQ_REPO = Path.home() / "thesis_models/repos/sam-hq"
sys.path.insert(0, str(SAM_HQ_REPO))

from segment_anything import sam_model_registry, SamPredictor

IMAGE_DIR = Path.home() / "thesis_models/pilot/images/clean"
LABEL_DIR = Path.home() / "thesis_models/pilot/outputs/yolov7/persons_only_025/labels"
OUTPUT_ROOT = Path.home() / "thesis_models/pilot/outputs/hqsam"
CHECKPOINT = Path.home() / "thesis_models/repos/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth"
MODEL_TYPE = "vit_b"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
sam.to(device=device)
predictor = SamPredictor(sam)

def load_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            if cls != 0:
                continue

            xc, yc, w, h = map(float, parts[1:5])

            x1 = max(0, (xc - w / 2) * img_w)
            y1 = max(0, (yc - h / 2) * img_h)
            x2 = min(img_w, (xc + w / 2) * img_w)
            y2 = min(img_h, (yc + h / 2) * img_h)

            boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    return boxes

image_paths = sorted(IMAGE_DIR.glob("*.jpg"), key=lambda p: int(p.stem))

for image_path in image_paths:
    stem = image_path.stem
    label_path = LABEL_DIR / f"{stem}.txt"
    out_dir = OUTPUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    img_h, img_w = image.shape[:2]

    boxes = load_yolo_boxes(label_path, img_w, img_h)
    print(f"{stem}: boxes found = {len(boxes)}")

    if not boxes:
        continue

    predictor.set_image(image)

    for i, box in enumerate(boxes, start=1):
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False
        )

        mask = (masks[0].astype(np.uint8) * 255)
        Image.fromarray(mask).save(out_dir / f"mask_{i}.png")

print(f"Saved HQ-SAM masks under: {OUTPUT_ROOT}")

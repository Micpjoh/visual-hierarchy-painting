from pathlib import Path
import numpy as np
from PIL import Image
import torch

IMAGE_DIR = Path.home() / "thesis_models/pilot/images/clean"
LABEL_DIR = Path.home() / "thesis_models/pilot/outputs/yolov7/persons_only_025/labels"
OUTPUT_ROOT = Path.home() / "thesis_models/pilot/outputs/sam3"
CHECKPOINT = Path.home() / "thesis_models/repos/sam3/checkpoints/sam3.1_multiplex.pt"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path=str(CHECKPOINT),
    device=device,
    eval_mode=True,
    enable_inst_interactivity=True,
    load_from_HF=False,
)
processor = Sam3Processor(model, device=device)

def load_yolo_boxes_normalized(label_path):
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
            boxes.append([xc, yc, w, h])

    return boxes

image_paths = sorted(IMAGE_DIR.glob("*.jpg"), key=lambda p: int(p.stem))

for image_path in image_paths:
    stem = image_path.stem
    label_path = LABEL_DIR / f"{stem}.txt"
    out_dir = OUTPUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    boxes = load_yolo_boxes_normalized(label_path)
    print(f"{stem}: boxes found = {len(boxes)}")

    if not boxes:
        continue

    state = {}
    processor.set_image(image, state=state)

    for i, box in enumerate(boxes, start=1):
        processor.reset_all_prompts(state)
        state = processor.add_geometric_prompt(box=box, label=True, state=state)

        masks = state["masks"]
        scores = state["scores"]

        if len(masks) == 0:
            print(f"{stem} box {i}: no mask returned")
            continue

        if torch.is_tensor(scores):
            best_idx = int(torch.argmax(scores).item())
        else:
            best_idx = int(np.argmax(scores))

        mask = masks[best_idx]
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()

        mask = np.squeeze(mask)

        if mask.ndim == 3:
            mask = mask[0]

        if mask.ndim != 2:
            raise ValueError(f"{stem} box {i}: unexpected mask shape {mask.shape}")

        mask = (mask > 0).astype(np.uint8) * 255
        Image.fromarray(mask).save(out_dir / f"mask_{i}.png")

print(f"Saved SAM3 masks under: {OUTPUT_ROOT}")

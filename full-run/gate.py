"""
IoU validation gate.

Shared by full_run_orchestrator.py (uses the passers list to decide which
paintings get sent to SAM3/DIS) and full_run.ipynb (re-runs and displays
the gate for transparency). Single source of truth for the gate logic.
"""
import json
from pathlib import Path

import cv2
import pandas as pd


def _load_yolo_person_boxes(painting_id, yolo_labels_dir, images_dir):
    """Read YOLOv7 label file -> list of (x1,y1,x2,y2) person boxes in pixels."""
    yolo_labels_dir = Path(yolo_labels_dir)
    images_dir = Path(images_dir)

    label_path = yolo_labels_dir / f"{painting_id}.txt"
    image_path = images_dir / f"{painting_id}.jpg"
    if not label_path.exists() or not image_path.exists():
        return [], None

    img = cv2.imread(str(image_path))
    if img is None:
        return [], None
    H, W = img.shape[:2]

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            if int(parts[0]) != 0:           # person class only
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            boxes.append((x1, y1, x2, y2))
    return boxes, (W, H)


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_via_important_boxes(via_json_path):
    """VIA export -> {painting_id: [(x1,y1,x2,y2), ...]}, first box = main figure.

    Handles both the simple VIA region export (top-level dict of entries) and
    the full VIA project export (entries nested under '_via_img_metadata').
    painting_id is the filename without extension.
    """
    with open(via_json_path, "r") as f:
        via_raw = json.load(f)

    # Full project export -> drill into _via_img_metadata
    if isinstance(via_raw, dict) and "_via_img_metadata" in via_raw:
        entries = via_raw["_via_img_metadata"]
    else:
        entries = via_raw

    out = {}
    for entry in entries.values():
        filename = entry["filename"]
        pid = filename.rsplit(".", 1)[0]   # strip any extension, not just .jpg
        boxes = []
        for region in entry.get("regions", []):
            a = region.get("shape_attributes", {})
            if a.get("name") != "rect":
                continue
            x1, y1 = float(a["x"]), float(a["y"])
            x2, y2 = x1 + float(a["width"]), y1 + float(a["height"])
            boxes.append((x1, y1, x2, y2))
        out[pid] = boxes
    return out


def compute_gate(via_json_path, yolo_labels_dir, images_dir, iou_threshold=0.20):
    """
    Apply the IoU validation gate.

    Returns
    -------
    gate_df : pd.DataFrame   one row per painting, with `passed_gate` boolean
    passed  : list[str]      painting_ids that passed
    failed  : list[str]      painting_ids that failed
    """
    via_boxes = load_via_important_boxes(via_json_path)

    rows = []
    for pid, vboxes in via_boxes.items():
        yolo_boxes, _ = _load_yolo_person_boxes(pid, yolo_labels_dir, images_dir)
        best = [max((_iou(v, y) for y in yolo_boxes), default=0.0) for v in vboxes]
        rows.append({
            "painting_id":       pid,
            "n_important_boxes": len(vboxes),
            "n_yolo_detections": len(yolo_boxes),
            "main_iou":          best[0] if best else 0.0,
            "any_important_iou": max(best) if best else 0.0,
        })

    gate_df = pd.DataFrame(rows).sort_values("painting_id").reset_index(drop=True)
    gate_df["passed_gate"] = gate_df["any_important_iou"] >= iou_threshold

    passed = sorted(gate_df.loc[gate_df["passed_gate"], "painting_id"])
    failed = sorted(gate_df.loc[~gate_df["passed_gate"], "painting_id"])

    return gate_df, passed, failed

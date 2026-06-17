#!/usr/bin/env python3
"""
feature_pipeline.py  —  Stages 5-6 of the auto-run pipeline.

Reads YOLO, SAM3, and DIS outputs. Computes the three primary features:
    iso_nearest_centroid_norm, elev_bbox_top, sal_sum.
Per-painting min-max normalises each. Builds combo_saliency_heavy.
Ranks detections per painting. Writes features.json and rankings.json.

NO annotation dependency. This script can be run on any painting dataset
without ground-truth boxes. The output rankings.json is the input to
evaluate.py, which is the only file that needs annotations.

Environment variables (set by orchestrator):
    PIPELINE_MANIFEST       path to manifest.json (image_id, image_path, width, height)
    PIPELINE_YOLO_OUT       path to yolo_detections.json
    PIPELINE_SAM3_OUT_DIR   directory of per-painting .npy mask files
    PIPELINE_DIS_OUT_DIR    directory of per-painting .npy saliency files
    PIPELINE_FEATURES_OUT   output path for features.json
    PIPELINE_RANKING_OUT    output path for rankings.json
    PIPELINE_W_SAL          composite weight for salience  (default 0.50)
    PIPELINE_W_ISO          composite weight for isolation (default 0.25)
    PIPELINE_W_ELEV         composite weight for elevation (default 0.25)
"""

import os
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ---------- Env helpers ----------

def env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var {key} not set by orchestrator")
    return val


def env_float(key: str, default: float = None) -> float:
    return float(env(key, str(default) if default is not None else None))


# ---------- Feature computation ----------

def compute_isolation_nearest_centroid(centroids: np.ndarray, diag: float) -> np.ndarray:
    """
    iso_nearest_centroid_norm: distance from each centroid to its nearest
    neighbour, normalised by the image diagonal. Larger = more isolated.
    Solo detections receive a placeholder of 1.0 (maximum isolation).
    """
    n = len(centroids)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.full(1, fill_value=1.0)
    tree = cKDTree(centroids)
    dists, _ = tree.query(centroids, k=2)   # col 0 = self (0), col 1 = nearest
    return dists[:, 1] / diag


def compute_elevation_bbox_top(boxes: np.ndarray, height: int) -> np.ndarray:
    """
    elev_bbox_top: 1 - (ymin / H).
    Top of painting -> ~1.0, bottom -> ~0.0.
    """
    ymin = boxes[:, 1]
    return 1.0 - (ymin / float(height))


def compute_salience_sum(masks: np.ndarray, saliency_map: np.ndarray) -> np.ndarray:
    """
    sal_sum: total DIS saliency contained within each figure's SAM3 mask.
    masks:        (N, H, W) boolean array
    saliency_map: (H, W)    float32 in [0, 1]
    """
    n = masks.shape[0]
    if n == 0:
        return np.zeros(0)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(saliency_map[masks[i]].sum())
    return out


def min_max_normalise(values: np.ndarray) -> np.ndarray:
    """
    Per-painting min-max normalisation to [0, 1].
    Returns all-zeros when all values are equal (no within-painting contrast),
    handling the division-by-zero edge case.
    """
    if len(values) == 0:
        return values
    mn, mx = float(values.min()), float(values.max())
    if mx <= mn:
        return np.zeros_like(values)
    return (values - mn) / (mx - mn)


# ---------- Per-painting pipeline ----------

def process_painting(rec, yolo_entry, masks, saliency_map, w_sal, w_iso, w_elev):
    """
    For one painting, compute features, composite score, and rank for every
    detected figure.

    Returns a list of detection dicts, each containing:
        detection_id, bbox, confidence, centroid,
        iso_raw, elev_raw, sal_raw,
        iso_norm, elev_norm, sal_norm,
        composite, rank
    Returns an empty list if no detections are available.
    """
    detections = yolo_entry.get("detections", [])
    if not detections:
        return []

    H    = int(rec["height"])
    W    = int(rec["width"])
    diag = float(np.hypot(H, W))

    boxes = np.array(
        [[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in detections],
        dtype=np.float64,
    )

    # Prefer mask centroids; fall back to bbox centroids when mask is absent
    centroids = np.zeros((len(detections), 2))
    for i in range(len(detections)):
        if masks is not None and i < masks.shape[0] and masks[i].any():
            ys, xs = np.where(masks[i])
            centroids[i] = [xs.mean(), ys.mean()]
        else:
            centroids[i] = [
                (boxes[i, 0] + boxes[i, 2]) / 2.0,
                (boxes[i, 1] + boxes[i, 3]) / 2.0,
            ]

    iso_raw  = compute_isolation_nearest_centroid(centroids, diag)
    elev_raw = compute_elevation_bbox_top(boxes, H)
    sal_raw  = (
        compute_salience_sum(masks, saliency_map)
        if masks is not None and saliency_map is not None
        else np.zeros(len(detections))
    )

    iso_norm  = min_max_normalise(iso_raw)
    elev_norm = min_max_normalise(elev_raw)
    sal_norm  = min_max_normalise(sal_raw)

    composite = w_sal * sal_norm + w_iso * iso_norm + w_elev * elev_norm

    # Mean-rank tie handling (Section 4.5): tied figures share the average rank
    order = np.argsort(-composite, kind="stable")
    ranks = np.empty(len(composite), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and composite[order[j + 1]] == composite[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    out = []
    for i, d in enumerate(detections):
        out.append({
            "detection_id": i,
            "bbox":        [d["xmin"], d["ymin"], d["xmax"], d["ymax"]],
            "confidence":  d["confidence"],
            "centroid":    centroids[i].tolist(),
            "iso_raw":     float(iso_raw[i]),
            "elev_raw":    float(elev_raw[i]),
            "sal_raw":     float(sal_raw[i]),
            "iso_norm":    float(iso_norm[i]),
            "elev_norm":   float(elev_norm[i]),
            "sal_norm":    float(sal_norm[i]),
            "composite":   float(composite[i]),
            "rank":        float(ranks[i]),
        })
    return out


# ---------- Main ----------

def main() -> None:
    manifest_path = Path(env("PIPELINE_MANIFEST"))
    yolo_path     = Path(env("PIPELINE_YOLO_OUT"))
    sam_dir       = Path(env("PIPELINE_SAM3_OUT_DIR"))
    dis_dir       = Path(env("PIPELINE_DIS_OUT_DIR"))
    features_out  = Path(env("PIPELINE_FEATURES_OUT"))
    ranking_out   = Path(env("PIPELINE_RANKING_OUT"))

    w_sal  = env_float("PIPELINE_W_SAL",  0.50)
    w_iso  = env_float("PIPELINE_W_ISO",  0.25)
    w_elev = env_float("PIPELINE_W_ELEV", 0.25)

    records = json.loads(manifest_path.read_text())
    yolo    = json.loads(yolo_path.read_text())

    all_features = {}
    all_rankings = {}
    n_no_detections = 0

    for rec in records:
        image_id    = rec["image_id"]
        yolo_entry  = yolo.get(image_id, {})

        sam_path     = sam_dir / f"{image_id}.npy"
        dis_path     = dis_dir / f"{image_id}.npy"
        masks        = np.load(sam_path)  if sam_path.exists()  else None
        saliency_map = np.load(dis_path)  if dis_path.exists()  else None

        ranked = process_painting(
            rec, yolo_entry, masks, saliency_map, w_sal, w_iso, w_elev
        )
        all_features[image_id] = ranked

        if not ranked:
            n_no_detections += 1
            all_rankings[image_id] = {
                "ranked_detections": [],
                "status":            "no_detections",
            }
        else:
            all_rankings[image_id] = {
                "ranked_detections": ranked,
                "status":            "ranked",
            }

    features_out.write_text(json.dumps(all_features, indent=2))
    ranking_out.write_text(json.dumps(all_rankings,  indent=2))

    n_total  = len(records)
    n_ranked = n_total - n_no_detections
    print("=" * 60)
    print("Feature pipeline complete")
    print(f"  Paintings in manifest:  {n_total}")
    print(f"  Successfully ranked:    {n_ranked}")
    print(f"  No detections:          {n_no_detections}")
    print(f"  Features written to:    {features_out}")
    print(f"  Rankings written to:    {ranking_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
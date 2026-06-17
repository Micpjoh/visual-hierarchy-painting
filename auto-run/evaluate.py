#!/usr/bin/env python3
"""
evaluate.py  —  Stage 7 of the auto-run pipeline.

Reads rankings.json (output of feature_pipeline.py) and ArtDL ground-truth
annotations. Runs IoU matching at three thresholds (0.20, 0.50, 0.70) and
reports IR metrics at each threshold.

This file is the ONLY file in the pipeline that requires ground-truth
annotations. feature_pipeline.py can be run without any annotations;
this file is run separately, after the pipeline, purely for evaluation.

Separation rationale:
    In application, users run feature_pipeline.py on their own painting
    dataset and receive ranked figure lists without needing any annotations.
    Evaluation against ground truth is a separate, optional step.

Environment variables (set by orchestrator):
    PIPELINE_RANKING_OUT    path to rankings.json from feature_pipeline.py
    PIPELINE_ARTDL_JSON     path to ArtDL test_saints.json (COCO format)
    PIPELINE_RESULTS_OUT    output path for results.json
    PIPELINE_TOP_K          comma-separated K values, e.g. "1,3"
    PIPELINE_GATE_IOU       primary IoU threshold (default 0.20);
                            sweep always includes 0.20, 0.50, 0.70

Metrics reported per threshold:
    P@1, P@3, R@1, R@3, MRR, Hit@3
    plus: n_scorable_paintings, n_paintings_with_match
"""

import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------- Env helpers ----------

def env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var {key} not set by orchestrator")
    return val


def env_float(key: str, default: float = None) -> float:
    return float(env(key, str(default) if default is not None else None))


def env_int_list(key: str) -> list[int]:
    return [int(x) for x in env(key).split(",")]


# ---------- ArtDL annotation loader ----------

def load_artdl_boxes(artdl_coco_path: Path) -> dict[str, list[dict]]:
    """
    Parse ArtDL test_saints.json (COCO format) into a dict:
        {image_id_stem: [{xmin, ymin, xmax, ymax, class_label}, ...]}

    COCO bbox format [x, y, w, h] is converted to (xmin, ymin, xmax, ymax).
    image_id is the JPEG filename stem, matching the keys used in rankings.json.
    """
    with open(artdl_coco_path) as f:
        coco = json.load(f)

    cat_name     = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img  = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    out = {}
    for img in coco["images"]:
        stem = Path(img["file_name"]).stem
        anns = anns_by_img.get(img["id"], [])
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append({
                "xmin":        float(x),
                "ymin":        float(y),
                "xmax":        float(x + w),
                "ymax":        float(y + h),
                "class_label": cat_name.get(ann["category_id"], "unknown"),
            })
        if boxes:
            out[stem] = boxes
    return out


# ---------- IoU computation ----------

def _iou(box_a, box_b) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_all_ious(ranked_detections, artdl_boxes) -> list[dict]:
    """
    Compute IoU for every (gt, detection) pair.
    Stores raw values so any threshold can be applied in post-hoc analysis
    without rerunning the pipeline.
    Only pairs with IoU > 0 are stored to keep the JSON compact.
    """
    detection_boxes = [tuple(d["bbox"]) for d in ranked_detections]
    pairs = []
    for k, gt in enumerate(artdl_boxes):
        gt_box = (gt["xmin"], gt["ymin"], gt["xmax"], gt["ymax"])
        for i, det_box in enumerate(detection_boxes):
            iou = _iou(gt_box, det_box)
            if iou > 0:
                pairs.append({
                    "gt_idx":      k,
                    "det_idx":     i,
                    "iou":         round(iou, 4),
                    "artdl_class": gt.get("class_label", "unknown"),
                    "det_rank":    ranked_detections[i]["rank"],
                })
    return pairs


def match_at_threshold(iou_pairs, artdl_boxes, ranked_detections,
                       iou_thresh) -> list[float]:
    """
    One-to-one greedy IoU matching at a given threshold.
    Returns a sorted list of ranks at which ArtDL boxes were matched.
    Strongest match committed first to prevent one large detection from
    claiming multiple small saint boxes.
    """
    candidates = [
        (p["iou"], p["gt_idx"], p["det_idx"], p["det_rank"])
        for p in iou_pairs
        if p["iou"] >= iou_thresh
    ]
    candidates.sort(key=lambda c: -c[0])

    used_gt, used_det = set(), set()
    matched_ranks = []
    for iou, k, i, rank in candidates:
        if k in used_gt or i in used_det:
            continue
        matched_ranks.append(rank)
        used_gt.add(k)
        used_det.add(i)
    return sorted(matched_ranks)


# ---------- Metrics (nDCG not included — binary relevance, dropped for
#            consistency with full run where importance is a single class) ----------

def precision_at_k(matched_ranks, k) -> float:
    hits = sum(1 for r in matched_ranks if r <= k)
    return min(hits, k) / float(k)


def recall_at_k(matched_ranks, n_total, k) -> float:
    if n_total == 0:
        return 0.0
    hits = sum(1 for r in matched_ranks if r <= k)
    return min(hits, n_total) / float(n_total)


def mrr(matched_ranks) -> float:
    return 1.0 / matched_ranks[0] if matched_ranks else 0.0


def hit_at_k(matched_ranks, k) -> float:
    return 1.0 if any(r <= k for r in matched_ranks) else 0.0


def compute_metrics(matched_ranks, n_total, top_k_values) -> dict:
    result = {"mrr": mrr(matched_ranks)}
    for k in top_k_values:
        result[f"p_at_{k}"]   = precision_at_k(matched_ranks, k)
        result[f"r_at_{k}"]   = recall_at_k(matched_ranks, n_total, k)
        result[f"hit_at_{k}"] = hit_at_k(matched_ranks, k)
    return result


# ---------- Main ----------

def main() -> None:
    ranking_path  = Path(env("PIPELINE_RANKING_OUT"))
    artdl_path    = Path(env("PIPELINE_ARTDL_JSON"))
    results_path  = Path(env("PIPELINE_RESULTS_OUT"))
    top_k_values  = env_int_list("PIPELINE_TOP_K")

    primary_thresh = env_float("PIPELINE_GATE_IOU", 0.20)
    iou_thresholds = sorted({primary_thresh, 0.20, 0.50, 0.70})

    rankings    = json.loads(ranking_path.read_text())
    artdl_boxes = load_artdl_boxes(artdl_path)

    # Accumulators per threshold
    metric_sums      = {t: defaultdict(float) for t in iou_thresholds}
    n_scorable       = {t: 0 for t in iou_thresholds}
    n_with_match     = {t: 0 for t in iou_thresholds}

    per_painting     = []
    n_no_detections  = 0
    n_no_artdl       = 0
    iou_pairs_store  = {}   # stored for post-hoc analysis

    for image_id, entry in rankings.items():
        ranked = entry.get("ranked_detections", [])
        gt_boxes = artdl_boxes.get(image_id, [])

        if not ranked:
            n_no_detections += 1
            continue

        if not gt_boxes:
            n_no_artdl += 1
            continue

        # Compute all pairwise IoUs once — reused across all thresholds
        iou_pairs = compute_all_ious(ranked, gt_boxes)
        iou_pairs_store[image_id] = iou_pairs

        n_total = len(gt_boxes)
        painting_entry = {
            "image_id":      image_id,
            "n_detections":  len(ranked),
            "n_artdl_boxes": n_total,
        }

        for thresh in iou_thresholds:
            matched_ranks = match_at_threshold(iou_pairs, gt_boxes, ranked, thresh)
            n_scorable[thresh]   += 1
            if matched_ranks:
                n_with_match[thresh] += 1

            m = compute_metrics(matched_ranks, n_total, top_k_values)
            for key, val in m.items():
                metric_sums[thresh][key]              += val
                painting_entry[f"t{thresh}_{key}"]    = val

        per_painting.append(painting_entry)

    # Build aggregate per threshold
    threshold_results = {}
    for thresh in iou_thresholds:
        n = n_scorable[thresh]
        agg = {
            "iou_threshold":           thresh,
            "n_scorable_paintings":    n,
            "n_paintings_with_match":  n_with_match[thresh],
            "n_no_detections":         n_no_detections,
            "n_no_artdl_annotations":  n_no_artdl,
        }
        if n > 0:
            agg["mean_mrr"] = metric_sums[thresh]["mrr"] / n
            for k in top_k_values:
                agg[f"mean_p_at_{k}"]   = metric_sums[thresh][f"p_at_{k}"]   / n
                agg[f"mean_r_at_{k}"]   = metric_sums[thresh][f"r_at_{k}"]   / n
                agg[f"mean_hit_at_{k}"] = metric_sums[thresh][f"hit_at_{k}"] / n
        threshold_results[str(thresh)] = agg

    results = {
        "threshold_results": threshold_results,
        "per_painting":      per_painting,
        "iou_pairs":         iou_pairs_store,
    }
    results_path.write_text(json.dumps(results, indent=2))

    # Print sweep table to stdout
    print("=" * 75)
    print("EVALUATION: IoU threshold sweep")
    print(f"{'Threshold':>12} {'Scorable':>10} {'Matched':>10} "
          f"{'P@1':>7} {'P@3':>7} {'R@1':>7} {'R@3':>7} "
          f"{'MRR':>7} {'Hit@3':>7}")
    print("-" * 75)
    for thresh in iou_thresholds:
        agg = threshold_results[str(thresh)]
        n   = agg["n_scorable_paintings"]
        if n == 0:
            continue
        print(
            f"{thresh:>12.2f} {n:>10} {agg['n_paintings_with_match']:>10} "
            f"{agg.get('mean_p_at_1', 0):>7.3f} "
            f"{agg.get('mean_p_at_3', 0):>7.3f} "
            f"{agg.get('mean_r_at_1', 0):>7.3f} "
            f"{agg.get('mean_r_at_3', 0):>7.3f} "
            f"{agg.get('mean_mrr',    0):>7.3f} "
            f"{agg.get('mean_hit_at_3', 0):>7.3f}"
        )
    print("=" * 75)
    print(f"Results written to: {results_path}")


if __name__ == "__main__":
    main()
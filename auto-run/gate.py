"""
Dual-criterion gate for the auto-run.

The full-run used IoU >= 0.20 against hand-drawn boxes, which were drawn around
the full visible extent of the figure (head to feet). That convention made IoU
a fair criterion.

ArtDL 2.0 does not follow that convention. Some boxes are full-figure; others
are tight crops around heads, gestures, or attributes. A small iconographic box
fully contained inside a YOLO full-figure detection can have IoU well below 0.20
while clearly referring to the same figure.

The gate criterion is therefore disjunctive: a detection matches an ArtDL box
if EITHER
   - IoU >= IOU_THRESH (handles full-figure annotations), OR
   - containment(artdl_box, detection) >= CONTAINMENT_THRESH (handles small
     iconographic annotations fully inside a larger detection).

Containment is asymmetric: it is the fraction of the SMALLER box (the ArtDL
annotation) that falls inside the LARGER box (the detection).
"""
from __future__ import annotations

from typing import Iterable


Box = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def _area(box: Box) -> float:
    xmin, ymin, xmax, ymax = box
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return w * h


def _intersection_area(a: Box, b: Box) -> float:
    xmin = max(a[0], b[0])
    ymin = max(a[1], b[1])
    xmax = min(a[2], b[2])
    ymax = min(a[3], b[3])
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return (xmax - xmin) * (ymax - ymin)


def compute_iou(a: Box, b: Box) -> float:
    """Standard intersection-over-union."""
    inter = _intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_containment(small_box: Box, large_box: Box) -> float:
    """
    Fraction of small_box that falls inside large_box.
    Returns 0.0 if small_box has zero area.
    """
    small_area = _area(small_box)
    if small_area <= 0.0:
        return 0.0
    return _intersection_area(small_box, large_box) / small_area


def is_match(
    artdl_box: Box,
    detection_box: Box,
    iou_thresh: float = 0.20,
    containment_thresh: float = 0.70,
) -> bool:
    """
    Returns True if an ArtDL ground-truth box matches a YOLO detection under
    EITHER IoU or containment.
    """
    if compute_iou(artdl_box, detection_box) >= iou_thresh:
        return True
    if compute_containment(artdl_box, detection_box) >= containment_thresh:
        return True
    return False


def best_match_index(
    artdl_box: Box,
    detections: Iterable[Box],
    iou_thresh: float = 0.20,
    containment_thresh: float = 0.70,
) -> "int | None":
    """
    Given one ArtDL box and a list of detections, return the index of the best
    matching detection or None if no detection matches under either criterion.
    Best match = highest combined score (max of IoU and containment), used only
    to break ties when multiple detections match.
    """
    best_idx = None
    best_score = -1.0
    for i, det in enumerate(detections):
        iou = compute_iou(artdl_box, det)
        cont = compute_containment(artdl_box, det)
        if iou >= iou_thresh or cont >= containment_thresh:
            score = max(iou, cont)
            if score > best_score:
                best_score = score
                best_idx = i
    return best_idx

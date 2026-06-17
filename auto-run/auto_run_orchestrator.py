#!/usr/bin/env python3
"""
Auto-run orchestrator: applies the fixed pipeline (YOLOv7 + SAM3 + DIS +
combo_saliency_heavy) to the ArtDL 2.0 bounding-box-annotated subset.

Structurally similar to full_run_orchestrator.py, with three changes:
  1. No early validation gate. Every painting that yields detections passes
     through segmentation and saliency.
  2. Only combo_saliency_heavy is computed. No sal_high_coverage_50 baseline,
     no Monte Carlo random baseline.
  3. The gate happens at the END, against ArtDL ground truth, using a dual
     criterion (IoU OR containment) to handle ArtDL's iconographic-style
     annotations.

Stages:
  1. Load ArtDL COCO annotations (test_saints.json) into a JSON manifest.
  2. YOLOv7 detection on all images (no gate filtering).
  3. SAM3 segmentation on every detection.
  4. DIS saliency on every image.
  5. Feature extraction: iso_nearest_centroid_norm, elev_bbox_top, sal_sum.
     Per-painting min-max normalisation. Compute combo_saliency_heavy.
  6. Rank detections per painting. Keep top-1 and top-3.
  7. Match top-K detections to ArtDL boxes using dual criterion.
     Compute and report IR metrics.

Run via SLURM: sbatch ~/thesis_models/auto-run/submit.sh
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# ---------- Paths and configuration ----------

THESIS_ROOT = Path(os.environ.get("THESIS_ROOT", "/home/mjohn/thesis_models"))
AUTO_RUN_DIR = THESIS_ROOT / "auto-run"
ENVS = THESIS_ROOT / "envs"
REPOS = THESIS_ROOT / "repos"

# ArtDL dataset locations.
# The dataset is the COCO-format saint-bbox release from Milani & Fraternali (2021):
#   test_saints.json contains 823 paintings with saint-figure bounding boxes.
# Only the JPEGs referenced by this JSON are uploaded to Snellius, not the full
# 42K-image classification set. test_symbols.json (symbolic attributes, e.g.
# crosses, lilies) is excluded by design: the pipeline detects human figures
# and cannot recover attribute boxes.
ARTDL_ROOT = AUTO_RUN_DIR / "artdl"
ARTDL_IMAGES = ARTDL_ROOT / "JPEGImages"
ARTDL_COCO_JSON = ARTDL_ROOT / "test_saints.json"

# Output locations
OUT = AUTO_RUN_DIR / "outputs"
MANIFEST = OUT / "manifest.json"
YOLO_OUT = OUT / "yolo_detections.json"
SAM_OUT = OUT / "sam3_masks"          # directory of .npy files, one per painting
DIS_OUT = OUT / "dis_saliency"        # directory of .npy files, one per painting
FEATURES_OUT = OUT / "features.json"
RANKING_OUT = OUT / "rankings.json"
RESULTS_OUT = OUT / "results.json"
LOG_DIR = OUT / "logs"

# Model checkpoint locations - same as full run
YOLO_WEIGHTS = REPOS / "yolov7" / "yolov7.pt"
SAM3_WEIGHTS = REPOS / "sam3" / "checkpoints" / "sam3.1_multiplex.pt"
DIS_WEIGHTS = REPOS / "DIS" / "saved_models" / "IS-Net" / "isnet-general-use.pth"

# Pipeline parameters (locked in by the full-run results, see Section 5.2)
YOLO_CONF = 0.25
YOLO_IOU_NMS = 0.45
YOLO_IMGSZ = 640
DIS_IMGSZ = 1024

# Composite weights (see Section 4.4)
W_SAL, W_ISO, W_ELEV = 0.50, 0.25, 0.25

# Top-K values reported (see Section 4.5)
TOP_K_VALUES = [1, 3]

# Gate thresholds for ArtDL matching (see auto-run setup in Section 5.3)
GATE_IOU_THRESH = 0.20

# ---------- Utilities ----------

def log(msg: str) -> None:
    """Print a timestamped log line and flush, so SLURM stdout is informative."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_dirs() -> None:
    for d in [OUT, SAM_OUT, DIS_OUT, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run_in_env(env_name: str, script: Path, cwd: Path, env_extra=None) -> int:
    """
    Activate one of the project's virtual environments and run a Python script
    inside it. Mirrors the convention used by full_run_orchestrator.py.
    Returns the subprocess return code.
    """
    venv_python = ENVS / env_name / "bin" / "python"
    env = os.environ.copy()
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    log(f"Running {script.name} in env '{env_name}' (cwd={cwd})")
    proc = subprocess.run(
        [str(venv_python), str(script)],
        cwd=str(cwd),
        env=env,
    )
    return proc.returncode


# ---------- Stage 1: Build manifest from ArtDL COCO JSON ----------

def build_manifest() -> None:
    """
    Parse the ArtDL COCO-format annotation file (test_saints.json) into a
    record per painting of:
        {image_id, image_path, width, height, important_boxes}
    where important_boxes is a list of {xmin, ymin, xmax, ymax, class_label}.

    The source file is COCO format:
      - 'images' is a list of {id, file_name, width, height}
      - 'annotations' is a list of {image_id, category_id, bbox: [x, y, w, h]}
      - 'categories' is a list of {id, name}

    Conversions applied here:
      - COCO bbox [xmin, ymin, w, h] -> (xmin, ymin, xmax, ymax) for the
        downstream gate, which expects the latter (see gate.py).
      - category_id is resolved to the saint name (e.g. '11H(JOHN THE BAPTIST)')
        and stored as class_label for qualitative analysis later.

    Paintings whose JPEG is missing on disk are skipped and logged.
    Paintings with no saint annotations are skipped (cannot be evaluated).
    """
    log("Stage 1: building manifest from ArtDL test_saints.json")

    if not ARTDL_COCO_JSON.exists():
        raise FileNotFoundError(f"ArtDL COCO JSON not found at {ARTDL_COCO_JSON}")
    if not ARTDL_IMAGES.exists():
        raise FileNotFoundError(f"ArtDL JPEGImages dir not found at {ARTDL_IMAGES}")

    with open(ARTDL_COCO_JSON) as f:
        coco = json.load(f)

    # Build category_id -> human-readable name lookup
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image_id for efficient assembly
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    records = []
    skipped_missing_jpeg = 0
    skipped_no_anns = 0

    for img in coco["images"]:
        coco_image_id = img["id"]
        file_name = img["file_name"]

        image_path = ARTDL_IMAGES / file_name
        if not image_path.exists():
            skipped_missing_jpeg += 1
            continue

        anns = anns_by_image.get(coco_image_id, [])
        if not anns:
            skipped_no_anns += 1
            continue

        # Convert each COCO box [x, y, w, h] to (xmin, ymin, xmax, ymax)
        boxes = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append({
                "class_label": cat_name.get(ann["category_id"], "unknown"),
                "xmin": float(x),
                "ymin": float(y),
                "xmax": float(x + w),
                "ymax": float(y + h),
            })

        # Use the JPEG filename stem as the image_id used throughout the pipeline.
        # The downstream stages (YOLO, SAM3, DIS) key all their outputs on this,
        # so it has to be stable and filesystem-safe.
        image_id = Path(file_name).stem

        records.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "width": int(img["width"]),
            "height": int(img["height"]),
            "important_boxes": boxes,
        })

    log(f"  parsed {len(records)} paintings from test_saints.json")
    log(f"  skipped {skipped_missing_jpeg} (JPEG not on disk)")
    log(f"  skipped {skipped_no_anns} (no saint annotations)")
    MANIFEST.write_text(json.dumps(records, indent=2))


# ---------- Stage 2: YOLOv7 detection ----------

def run_yolo() -> None:
    """
    Run YOLOv7 on every painting in the manifest. No gate filtering. Output
    is a JSON keyed by image_id with detection boxes and confidence scores.
    """
    log("Stage 2: YOLOv7 person detection on all manifest paintings")
    rc = run_in_env(
        env_name="yolo",
        script=Path(__file__).parent / "models" / "YOLOautorun.py",
        cwd=REPOS / "yolov7",
        env_extra={
            "PIPELINE_MANIFEST": str(MANIFEST),
            "PIPELINE_YOLO_WEIGHTS": str(YOLO_WEIGHTS),
            "PIPELINE_YOLO_CONF": str(YOLO_CONF),
            "PIPELINE_YOLO_IOU_NMS": str(YOLO_IOU_NMS),
            "PIPELINE_YOLO_IMGSZ": str(YOLO_IMGSZ),
            "PIPELINE_YOLO_OUT": str(YOLO_OUT),
        },
    )
    if rc != 0:
        raise RuntimeError(f"YOLOv7 stage failed with code {rc}")


# ---------- Stage 3: SAM3 segmentation ----------

def run_sam3() -> None:
    """
    Run SAM3 on every detection from every painting. Output is one .npy per
    painting in SAM_OUT, containing a stacked boolean mask array (N_detections,
    H, W).
    """
    log("Stage 3: SAM3 segmentation on all detections (no gate filtering)")
    rc = run_in_env(
        env_name="samfamily",
        script=Path(__file__).parent / "models" / "sam3autorun.py",
        cwd=REPOS / "sam3",
        env_extra={
            "PIPELINE_MANIFEST": str(MANIFEST),
            "PIPELINE_YOLO_OUT": str(YOLO_OUT),
            "PIPELINE_SAM3_WEIGHTS": str(SAM3_WEIGHTS),
            "PIPELINE_SAM3_OUT_DIR": str(SAM_OUT),
        },
    )
    if rc != 0:
        raise RuntimeError(f"SAM3 stage failed with code {rc}")


# ---------- Stage 4: DIS saliency ----------

def run_dis() -> None:
    """
    Run DIS on every painting. Output is one .npy per painting in DIS_OUT
    containing a float saliency map at original image resolution.
    """
    log("Stage 4: DIS saliency estimation on all manifest paintings")
    rc = run_in_env(
        env_name="samfamily",
        script=Path(__file__).parent / "models" / "DISautorun.py",
        cwd=REPOS / "DIS" / "IS-Net",
        env_extra={
            "PIPELINE_MANIFEST": str(MANIFEST),
            "PIPELINE_DIS_WEIGHTS": str(DIS_WEIGHTS),
            "PIPELINE_DIS_IMGSZ": str(DIS_IMGSZ),
            "PIPELINE_DIS_OUT_DIR": str(DIS_OUT),
            # Required because the DIS scripts assume their own repo root on sys.path
            "PYTHONPATH": str(REPOS / "DIS" / "IS-Net"),
        },
    )
    if rc != 0:
        raise RuntimeError(f"DIS stage failed with code {rc}")


# ---------- Stages 5-7: feature extraction, ranking, gate matching, metrics ----------
# Done in pure Python (CPU) inside this same script, since these stages don't
# need a GPU and don't depend on any of the per-model environments.

def run_feature_pipeline() -> None:
    """
    Reads YOLO detections, SAM3 masks, DIS saliency maps.
    Computes iso_nearest_centroid_norm, elev_bbox_top, sal_sum.
    Builds combo_saliency_heavy. Ranks figures per painting.
    Writes features.json and rankings.json.
 
    No annotation dependency — can be run on any painting dataset.
    """
    log("Stages 5-6: feature extraction + ranking (feature_pipeline.py)")
    rc = run_in_env(
        env_name="samfamily",
        script=Path(__file__).parent / "feature_pipeline.py",
        cwd=AUTO_RUN_DIR,
        env_extra={
            "PIPELINE_MANIFEST":      str(MANIFEST),
            "PIPELINE_YOLO_OUT":      str(YOLO_OUT),
            "PIPELINE_SAM3_OUT_DIR":  str(SAM_OUT),
            "PIPELINE_DIS_OUT_DIR":   str(DIS_OUT),
            "PIPELINE_FEATURES_OUT":  str(FEATURES_OUT),
            "PIPELINE_RANKING_OUT":   str(RANKING_OUT),
            "PIPELINE_W_SAL":         str(W_SAL),
            "PIPELINE_W_ISO":         str(W_ISO),
            "PIPELINE_W_ELEV":        str(W_ELEV),
        },
    )
    if rc != 0:
        raise RuntimeError(f"Feature pipeline stage failed with code {rc}")
 
 
# ---------- Stage 7: evaluation against ArtDL annotations ----------
 
def run_evaluation() -> None:
    """
    Reads rankings.json (output of feature_pipeline.py) and ArtDL annotations.
    Runs IoU matching at thresholds 0.20, 0.50, 0.70.
    Writes results.json with per-threshold aggregate metrics.
 
    This is the ONLY stage that requires ground-truth annotations.
    Separated from the pipeline so users without annotations can still run
    stages 1-6 and obtain ranked figure lists.
    """
    log("Stage 7: evaluation against ArtDL ground truth (evaluate.py)")
    rc = run_in_env(
        env_name="samfamily",
        script=Path(__file__).parent / "evaluate.py",
        cwd=AUTO_RUN_DIR,
        env_extra={
            "PIPELINE_RANKING_OUT":  str(RANKING_OUT),
            "PIPELINE_ARTDL_JSON":   str(ARTDL_COCO_JSON),
            "PIPELINE_RESULTS_OUT":  str(RESULTS_OUT),
            "PIPELINE_TOP_K":        ",".join(str(k) for k in TOP_K_VALUES),
            "PIPELINE_GATE_IOU":     str(GATE_IOU_THRESH),
        },
    )
    if rc != 0:
        raise RuntimeError(f"Evaluation stage failed with code {rc}")
 
 
# ---------- Main ----------
 
def main() -> None:
    ensure_dirs()
    t0 = time.time()
    log(f"Auto-run started. Output dir: {OUT}")
 
    build_manifest()
    run_yolo()            
    run_sam3()            
    run_dis()              
    run_feature_pipeline()   
    run_evaluation()        
 
    log(f"Auto-run finished in {(time.time() - t0) / 60:.1f} minutes")
    log(f"Results: {RESULTS_OUT}")
 
 
if __name__ == "__main__":
    main()

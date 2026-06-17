#!/usr/bin/env python3
"""
Full-run orchestrator.

Stages:
  1. Symlink first 200 images into outputs/images_subset/
  2. YOLOv7 person detection (yolo env)
  3. IoU validation gate -> writes paintings_passed.txt
  4. SAM3 segmentation on gate-passers only (samfamily env)
  5. DIS saliency on gate-passers only (samfamily env)
  6. Execute full_run.ipynb -> full_run_executed.ipynb (gate display +
     feature extraction with the locked-in techniques + per-painting summary)

Run interactively from a GPU node, or via submit.sh under SLURM.
"""
import os
import subprocess
import sys
from pathlib import Path

# Local import: gate computation shared with the notebook.
sys.path.insert(0, str(Path(__file__).parent))
import gate as gate_module


# ============================================================
# Paths
# ============================================================
HOME = Path.home()

# Source data
IMAGE_SRC   = HOME / "thesis_models" / "full-run" / "images"
ANNOTATIONS = HOME / "thesis_models" / "full-run" / "annotations.json"
N_IMAGES    = 250

# This run lives here
RUN_DIR = HOME / "thesis_models" / "full-run"
OUTPUTS = RUN_DIR / "outputs"
LOGS    = RUN_DIR / "logs"

SUBSET_DIR  = OUTPUTS / "images_subset"
YOLO_OUT    = OUTPUTS / "yolov7"
YOLO_LABELS = YOLO_OUT / "persons_only_025" / "labels"
SAM3_OUT    = OUTPUTS / "sam3"
DIS_OUT     = OUTPUTS / "dis"
PASS_LIST   = OUTPUTS / "paintings_passed.txt"
GATE_CSV    = OUTPUTS / "iou_per_painting.csv"

# Environments and repos.
MODELS = HOME / "thesis_models"
ENVS   = MODELS / "envs"
REPOS  = MODELS / "repos"

YOLO_PY = ENVS / "yolo" / "bin" / "python"
SAM_PY  = ENVS / "samfamily" / "bin" / "python"
DIS_PY  = ENVS / "samfamily" / "bin" / "python"      
NB_PY   = ENVS / "samfamily" / "bin" / "python"

IOU_THRESHOLD = 0.20


# ============================================================
# Helpers
# ============================================================
def banner(label):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}", flush=True)


def run(label, cmd, cwd=None, env=None):
    banner(label)
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd, cwd=cwd, env=env).returncode
    if rc != 0:
        sys.exit(f"!! {label} failed (exit {rc})")


def prepare_subset():
    """Symlink up to N images from IMAGE_SRC into a clean subset dir."""
    if not IMAGE_SRC.exists():
        sys.exit(f"!! image source not found: {IMAGE_SRC}")
    if not ANNOTATIONS.exists():
        sys.exit(f"!! annotations not found: {ANNOTATIONS}")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    for old in SUBSET_DIR.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = sorted(p for p in IMAGE_SRC.iterdir() if p.suffix in exts)
    chosen = images[:N_IMAGES]

    for src in chosen:
        (SUBSET_DIR / src.name).symlink_to(src.resolve())
    print(f"Subset: linked {len(chosen)} of {len(images)} images into {SUBSET_DIR}")
    return len(chosen)


def apply_gate():
    """Run the gate, write paintings_passed.txt + iou_per_painting.csv."""
    gate_df, passed, failed = gate_module.compute_gate(
        via_json_path    = ANNOTATIONS,
        yolo_labels_dir  = YOLO_LABELS,
        images_dir       = SUBSET_DIR,
        iou_threshold    = IOU_THRESHOLD,
    )
    gate_df.to_csv(GATE_CSV, index=False)
    PASS_LIST.write_text("\n".join(passed) + ("\n" if passed else ""))
    print(f"Gate (IoU >= {IOU_THRESHOLD}): {len(passed)} passed, {len(failed)} failed")
    print(f"Wrote: {GATE_CSV}")
    print(f"Wrote: {PASS_LIST}")


# ============================================================
# Pipeline
# ============================================================
def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DIS_OUT.mkdir(parents=True, exist_ok=True)
    SAM3_OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Image subset ----
    banner("1. Prepare image subset")
    n = prepare_subset()
    if n == 0:
        sys.exit("!! no images in subset")

   # ---- 2. YOLOv7 ----
    yolo_env = os.environ.copy()
    yolo_env.update({
        "PIPELINE_IMAGES":    str(SUBSET_DIR),
        "PIPELINE_YOLO_OUT":  str(YOLO_OUT / "persons_only_025"),
        "PIPELINE_YOLO_REPO": str(REPOS / "yolov7"),
    })
    run("2. YOLOv7 (person class, conf 0.25)",
        [str(YOLO_PY), str(RUN_DIR / "models" / "YOLOfullrun.py")],
        env=yolo_env)

    # ---- 3. Validation gate ----
    banner("3. Validation gate")
    apply_gate()

    # ---- 4. SAM3 (passers only) ----
    sam_env = os.environ.copy()
    sam_env.update({
        "PIPELINE_IMAGES":    str(SUBSET_DIR),
        "PIPELINE_LABELS":    str(YOLO_LABELS),
        "PIPELINE_SAM_OUT":   str(SAM3_OUT),
        "PIPELINE_PASS_LIST": str(PASS_LIST),
    })
    run("4. SAM3 (passers only)",
        [str(SAM_PY), str(RUN_DIR / "models" / "sam3fullrun.py")],
        env=sam_env)

    # ---- 5. DIS (passers only) ----
    dis_env = os.environ.copy()
    dis_env.update({
        "PIPELINE_IMAGES":    str(SUBSET_DIR),
        "PIPELINE_DIS_OUT":   str(DIS_OUT),
        "PIPELINE_PASS_LIST": str(PASS_LIST),
    })
    run("5. DIS (passers only)",
        [str(DIS_PY), str(RUN_DIR / "models" / "DISfullrun.py")],
        cwd=REPOS / "DIS" / "IS-Net",
        env=dis_env)

    # ---- 6. Execute the analysis notebook ----
    # Notebook reads paths from these env vars (with sensible fallbacks).
    nb_env = os.environ.copy()
    nb_env.update({
        "PIPELINE_IMAGES":      str(SUBSET_DIR),
        "PIPELINE_ANNOTATIONS": str(ANNOTATIONS),
        "PIPELINE_YOLO_LABELS": str(YOLO_LABELS),
        "PIPELINE_SAM_OUT":     str(SAM3_OUT),
        "PIPELINE_DIS_OUT":     str(DIS_OUT),
        "PIPELINE_GATE_CSV":    str(GATE_CSV),
        "PIPELINE_PASS_LIST":   str(PASS_LIST),
        "PIPELINE_OUTPUTS":     str(OUTPUTS),
        "PIPELINE_IOU_THRESHOLD": str(IOU_THRESHOLD),
    })
    notebook_in  = RUN_DIR / "full_run.ipynb"
    notebook_out = OUTPUTS / "full_run_executed.ipynb"
    run("6. Execute full_run.ipynb", [
        str(NB_PY), "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", str(notebook_in),
        "--output", str(notebook_out),
        "--ExecutePreprocessor.timeout=3600",
    ], env=nb_env)

    banner("Pipeline complete")
    print(f"Executed notebook: {notebook_out}")
    print(f"Outputs:           {OUTPUTS}")


if __name__ == "__main__":
    main()

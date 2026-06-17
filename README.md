# Computationally Measuring Visual Hierarchy in Group Paintings

Code, annotations, and results for the bachelor thesis:  
**"Computationally Measuring Visual Hierarchy in Group Paintings"**  
by Micah Philippe John, University of Amsterdam, 2026.  
Supervisor: Nanne van Noord.

## Research Question

Whether figures of importance in group paintings are visually emphasised through
measurable compositional cues — spatial isolation, compositional elevation, and
visual salience — and whether existing computer vision tools can recover that emphasis.

## Pipeline

```
YOLOv7 (person detection, conf 0.25)
  → IoU validation gate (≥0.20)
  → Centroid matching (YOLO ↔ hand-drawn annotation)
  → SAM3 (figure segmentation)
  → DIS (saliency estimation)
  → Feature extraction (isolation, elevation, salience)
  → Salience-heavy composite ranking:
      score_i = 0.50·S̃ + 0.25·Ĩ + 0.25·Ẽ
```

## Repository Structure

```
├── pilot/
│   └── calculations.ipynb           # Pilot experiment (25 Renaissance paintings)
│
├── full-run/
│   ├── full_run.ipynb               # Full-run analysis notebook
│   ├── full_run_orchestrator.py     # End-to-end pipeline orchestrator
│   ├── gate.py                      # IoU validation gate
│   ├── annotations.json             # Hand-drawn VIA importance annotations
│   ├── final_combined_dataset.csv   # Painting metadata
│   ├── diagnostic/
│   │   └── spot_check_misses.py     # Qualitative failure-case overlays
│   └── outputs/
│       ├── all_features_*.csv       # Feature tables
│       ├── detection_failures.csv   # Gate failures
│       ├── iou_per_painting.csv     # Per-painting IoU scores
│       └── summary/                 # Result tables and figures
│
├── auto-run/
│   ├── auto_run_orchestrator.py     # Auto-run pipeline orchestrator
│   ├── feature_pipeline.py          # Feature extraction (annotation-free)
│   ├── evaluate.py                  # Evaluation with IoU threshold sweep
│   ├── autorun_analysis.ipynb       # Post-hoc analysis and figures
│   └── outputs/
│       ├── features.json            # Per-painting features
│       ├── rankings.json            # Pipeline rankings (no annotations needed)
│       ├── results.json             # Evaluation results per IoU threshold
│       └── analysis/                # Figures and tables
│
├── scripts/
│   ├── run_sam3.py                  # SAM3 inference from YOLO boxes
│   ├── run_dis.py                   # DIS saliency map generation
│   └── generate_overlays.py         # Diagnostic overlay generation
│
├── patches/
│   ├── DIS_inference.patch          # Path + output squeeze fix
│   └── yolov7_experimental.patch    # PyTorch 2.x compatibility fix
│
├── requirements_samfamily.txt       # SAM3, DIS, analysis environment
├── requirements_yolo.txt            # YOLOv7 environment
└── submit.sh                        # SLURM job submission script
```

## Three Experiments

| Experiment | Paintings | Purpose |
|------------|-----------|---------|
| **Pilot** | 25 (Renaissance) | Component selection and calibration |
| **Full run** | 250 (1500–1930) | Main evaluation against hand-drawn annotations |
| **Auto-run** | 808 (ArtDL 2.0) | External transfer test against independent annotations |

## Key Results

| Experiment | P@1 | MRR | Hit@3 |
|------------|-----|-----|-------|
| Full run — composite | 0.569 | 0.728 | 0.873 |
| Full run — salience only | 0.402 | 0.614 | 0.789 |
| Full run — random | 0.269 | 0.500 | 0.665 |
| Auto-run (IoU ≥0.20) | 0.878 | 0.930 | 0.986 |
| Auto-run (IoU ≥0.50) | 0.851 | 0.903 | 0.960 |
| Auto-run (IoU ≥0.70) | 0.741 | 0.799 | 0.864 |

## External Repositories

Two separate Python environments are required due to dependency conflicts
between YOLOv7 and the SAM family models.

| Repo | URL | Commit | Modified? |
|------|-----|--------|-----------|
| YOLOv7 | https://github.com/WongKinYiu/yolov7 | `a207844` | Yes — `weights_only=False` for PyTorch 2.x |
| SAM3 | https://github.com/facebookresearch/sam3 | `2e0009e` | No |
| SAM-HQ | https://github.com/SysCV/sam-hq | `e696978` | No |
| SAM | https://github.com/facebookresearch/segment-anything | `dca509f` | No |
| DIS | https://github.com/xuebinqin/DIS | `b6764e2` | Yes — path config + output squeeze fix |
| U-2-Net | https://github.com/xuebinqin/U-2-Net | `ac7e1c8` | No |

Patches for modified repos are in `patches/`. Apply with:
```bash
cd <repo_directory>
git apply <patch_file>
```

## Model Weights (not included)

Download and place in the appropriate repo directory:

- **YOLOv7**: `yolov7.pt` from [YOLOv7 releases](https://github.com/WongKinYiu/yolov7/releases)
- **SAM3**: `sam3.1_multiplex.pt` from [SAM3 checkpoints](https://github.com/facebookresearch/sam3)
- **DIS**: `isnet-general-use.pth` from [DIS releases](https://github.com/xuebinqin/DIS)

## Reproduction

1. Clone this repository
2. Set up the two Python environments from `requirements_*.txt`
3. Clone and patch external repos as listed above
4. Download model weights
5. For the full run: `python full-run/full_run_orchestrator.py`
6. For the auto-run: `python auto-run/auto_run_orchestrator.py`

The pipeline runs on NVIDIA GPU (tested on SURF Snellius, A100).
Analysis notebooks can be run on CPU after the pipeline stages complete.

## Dataset

- **Full run**: 250 paintings sourced from the Metropolitan Museum of Art and
  Cleveland Museum of Art Open Access collections.
- **Auto-run**: ArtDL 2.0 test split (`test_saints.json`, 808 paintings).
  See [Milani & Fraternali, 2021](https://doi.org/10.1145/3458885).

## License

Code is provided for academic use. Painting images are subject to the
respective museum open-access policies.

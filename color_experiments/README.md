# Assignment 5: Color Thresholding Experiments

This is the single reproduction guide for Assignment 5 in the ALINA fork. It covers setup, annotation, training, evaluation, outputs, and demonstrations. This folder contains scripts to reproduce the Assignment 5 comparison of five color-based methods against ALINA's original HSV threshold. The normal `alina label` command and `alina/pipeline.py` are unchanged. The experiment runner supplies alternative color masks to ALINA's histogram and CIRCLEDAT steps.

## 1. Setup

Run all commands **from the ALINA repository root**. Python and a virtual environment are required. For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install scikit-learn joblib pandas matplotlib
```

Input frames should be JPG images in `data/Raw_Data/vidd_1/`, `vidd_2/`, and `vidd_3/`. If the virtual environment does not exist yet, create it with `python -m venv .venv` before activation. No Gemini API key is needed for Assignment 5.

## 2. Annotate the frames

```powershell
python color_experiments/annotate.py
```

The tool chooses three training and two test frames per video by default (15 total), using seed 42. Select a fixed ROI for each video by clicking **bottom-left, top-left, top-right, bottom-right**, then press `N`. On the warped images, use `Y` for yellow paint, `W` for white paint, `B` for background, `U` to undo, `+`/`-` for brush size, `N` to save/continue, and `Q` to save/quit. Only paint actual markings; do not draw across gaps between dashed lines. Background strokes should cover varied pavement and lighting.

Files are written to `outputs/color_annotations/train/<video>/<frame>/` and `outputs/color_annotations/test/<video>/<frame>/`, with `warped.png` and `labels.png` in each frame folder. Label IDs are `0` unlabeled, `1` background, `2` yellow, `3` white. Per-video `<video>_roi.json` and `<video>_selection.json` files are also written at the annotation root. Existing labels are skipped when the tool is rerun.

**For exact reproduction of our experiment:** the original automatic split was adjusted by moving the one frame with white markings into training and exchanging it with a test frame. Preserve the final annotation folders and update the corresponding selection JSON/manifest when doing this; otherwise a fresh run with seed 42 may produce a different split. No frame should occur in both splits. All three labeled classes must be present in training.

## 3. Train the models

```powershell
python color_experiments/train_models.py
```

Training uses labeled pixels from `train/` only, balanced sampling and seed 42. The models are saved to `outputs/color_experiments/models.joblib`.

The methods are:

| Method name | Description |
|---|---|
| `baseline` | Original fixed normalized-HSV threshold; reference method, not ML |
| `kmeans` | K-Means clusters normalized HSV; labeled training pixels assign cluster classes |
| `gmm` | Gaussian mixture clustering with the same class-mapping approach |
| `random_forest` | Supervised classifier using normalized HSV and BGR values |
| `svm` | Scaled features with **LinearSVC** (not the earlier RBF SVM) |
| `adaptive_kmeans` | Refit K-Means on each frame and match cluster colors to training prototypes |

Yellow and white predictions are combined into one binary candidate-marking mask.

## 4. Evaluate the color masks

```powershell
python color_experiments/evaluate.py
python color_experiments/analyze_results.py
```

`evaluate.py` compares each mask to the six held-out annotated test frames and writes `outputs/color_experiments/test_metrics.csv` and images under `outputs/color_experiments/previews/`. `analyze_results.py` creates `summary.csv` and `comparison_f1.png`, `comparison_iou.png`, and `comparison_mask_ms.png` in `outputs/color_experiments/`.

Precision, recall, F1, and IoU are calculated **only on manually labeled pixels**, not the full image. Timing measures full-frame mask generation. The evaluation combines yellow and white into one positive class.

## 5. Run the complete ALINA comparison

The end-to-end script reads the fixed ROIs saved during annotation, applies the perspective transform, generates a color mask, removes predictions outside the valid transformed ROI, and runs ALINA's histogram and CIRCLEDAT stages. To reproduce the completed experiment (five evenly spaced frames per video, 15 frames per method):

```powershell
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods baseline svm --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_roi_fixed
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods kmeans gmm random_forest adaptive_kmeans --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_remaining
```

The flag is **`--output`**, not `--output-dir`. Separate folders are used because the script records run settings and rejects mismatched settings when reusing an output directory. To start a clean run, choose a new output directory. Other options include `--fps` (sampled-frame MP4 playback rate), `--skip-videos`, `--models`, `--annotations`, and `--data-dir`.

Each output directory contains `results.csv`, `summary.csv`, per-frame masks, valid-ROI masks, annotated JPGs, coordinate TXT files, and MP4 demonstrations named `<video>_<method>.mp4`. The MP4s show **sampled frames**, not continuous video. Completed results are saved incrementally for restart.

The 100,000-pixel limit avoids extremely slow traversals. A `skipped_excessive` frame exceeded the selected-component limit and was **not** traversed; `no_peak` means the histogram did not accept a starting point; `ok` means traversal ran, but may still yield zero final pixels. The end-to-end detection rate is the fraction of frames with nonempty output under these safeguards, **not ground-truth line-detection accuracy**. Excessive skips must not be interpreted as correct negative detections.

## 6. Where to find evidence

| Output | Location |
|---|---|
| Saved models | `outputs/color_experiments/models.joblib` |
| Pixel-level per-frame metrics | `outputs/color_experiments/test_metrics.csv` |
| Pixel-level summary | `outputs/color_experiments/summary.csv` |
| Three comparison plots | `outputs/color_experiments/comparison_*.png` |
| Predicted masks and overlays | `outputs/color_experiments/previews/` |
| End-to-end results | `outputs/color_experiments/end_to_end_roi_fixed/` and `end_to_end_remaining/` |
| Sampled-frame videos | `<end-to-end output>/<video>_<method>.mp4` |

Generated `outputs/` content may be ignored by Git. For a repository-only submission, include representative plots, CSV summaries, and MP4 demonstrations in a tracked results folder or link to them explicitly. The README by itself does not supply those generated files.

## Reproducibility notes

The experiment uses seed 42 for training and frame selection; runtime varies with hardware. The six-frame test uses sparse manually drawn labels and does not independently measure white-only accuracy because the identified white-marking frame was moved into training. Cluster class assignments use training labels, so K-Means and GMM are not completely annotation-free. The original ALINA CLI remains unchanged; learned methods are evaluated through `run_end_to_end.py` rather than being installed as its default thresholding step. See the separate Assignment 5 report for results, interpretation, and limitations.

## Quick command reference

Run these from the repository root after setup and after the annotation frames have been checked:

```powershell
python color_experiments/train_models.py
python color_experiments/evaluate.py
python color_experiments/analyze_results.py
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods baseline svm --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_roi_fixed
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods kmeans gmm random_forest adaptive_kmeans --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_remaining
```

The Assignment 5 report is separate from this README. The earlier Assignment 4 experiment READMEs remain in their own folders.

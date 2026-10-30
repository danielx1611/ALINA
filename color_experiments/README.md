# Assignment 5: Color Thresholding Experiments

This folder contains my Assignment 5 work on replacing ALINA's fixed color threshold with methods that learn from image colors. I compared K-Means, Gaussian Mixture Models (GMM), Random Forest, Linear SVM, and adaptive K-Means against ALINA's original HSV threshold. The original `alina/pipeline.py` and `alina label` command are unchanged. The experiment scripts generate alternative masks and pass them through ALINA's histogram and CIRCLEDAT steps.

## Setup

Run these commands from the repository root. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install scikit-learn joblib pandas matplotlib
```

If you already have a virtual environment, you can skip creating it. The scripts expect JPG frames in `data/Raw_Data/vidd_1/`, `vidd_2/`, and `vidd_3/`.

## Annotate the frames

```powershell
python color_experiments/annotate.py
```

The annotation tool selects three training frames and two test frames per video using seed 42. For each video, click the ROI corners in this order: bottom-left, top-left, top-right, bottom-right, then press `N`. The tool then shows the transformed frames for labeling.

Use `Y` for yellow paint, `W` for white paint, and `B` for background. `U` undoes a stroke, `+` and `-` change the brush size, `N` saves and moves on, and `Q` saves and quits. Draw on the painted markings themselves rather than across the spaces between dashed lines. A few strokes across different background areas are enough; the entire image does not need to be filled in.

Annotations are saved under `outputs/color_annotations/train/` and `outputs/color_annotations/test/`. Each frame folder contains `warped.png` and `labels.png`. The label values are 0 for unlabeled, 1 for background, 2 for yellow, and 3 for white. The annotation folder also contains ROI and frame-selection JSON files for each video.

For my final experiment, I swapped one white-marking frame into the training set so all three classes were represented. The original random selection alone does not reproduce that change. Exact reproduction requires the final annotation folders and their matching ROI and selection files. Those annotation files are local outputs and are not included in the committed results folder.

## Train the models

```powershell
python color_experiments/train_models.py
```

This reads labeled training pixels, samples them by class, and fits the models with seed 42. The models are saved to `outputs/color_experiments/models.joblib`.

| Method | How it works |
|---|---|
| `baseline` | ALINA's original fixed HSV threshold; included for comparison |
| `kmeans` | Groups similar HSV colors and assigns classes using the training labels |
| `gmm` | Groups colors using a Gaussian mixture model, then assigns classes from the training labels |
| `random_forest` | Learns the three classes from normalized HSV and BGR color values |
| `svm` | Uses a scaled Linear SVM to classify pixels |
| `adaptive_kmeans` | Finds color groups separately for each frame and matches them to learned class colors |

All methods produce a binary mask where yellow and white predictions count as possible line markings.

## Compare the color masks

```powershell
python color_experiments/evaluate.py
python color_experiments/analyze_results.py
```

The evaluation uses the six held-out annotated frames. It calculates precision, recall, F1, and IoU on the **pixels that were manually labeled**. Unlabeled pixels are excluded from those scores. The reported mask time covers prediction on the full transformed image.

The scripts save per-frame results to `outputs/color_experiments/test_metrics.csv`, averages to `summary.csv`, and graphs to `comparison_f1.png`, `comparison_iou.png`, and `comparison_mask_ms.png`. They also save predicted masks and overlays under `outputs/color_experiments/previews/`.

## Test the methods in ALINA

These commands use five evenly spaced frames from each video, giving 15 frames per method:

```powershell
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods baseline svm --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_roi_fixed
python color_experiments/run_end_to_end.py --frames-per-video 5 --methods kmeans gmm random_forest adaptive_kmeans --max-component-pixels 100000 --output outputs/color_experiments/end_to_end_remaining
```

The script uses the saved ROI for each video, generates the color mask, removes predictions outside the transformed ROI, and runs ALINA's histogram and CIRCLEDAT stages. The 100,000-pixel component limit prevents very large regions from causing long traversal times. A `skipped_excessive` result means the limit was reached and traversal was skipped. A `no_peak` result means ALINA did not accept a starting point. An `ok` result means traversal ran, but it can still produce zero final pixels.

The end-to-end detection rate is the share of frames with nonempty output after these checks. It does **not** tell us whether the detected pixels match the actual painted lines. The script also saves mask coverage and timing measurements. Results are written to `results.csv` and `summary.csv` in each selected output folder, along with masks and annotated frames. The script can generate sampled-frame MP4 files locally, but no videos are included in the committed evidence.

Use `--output` to choose a results folder. Runs with different settings should use different folders, since the script checks its saved settings before resuming.

## Results available on GitHub

The following files are committed in [`results/`](results/), so they can be viewed without rerunning the experiments:

| File | Contents |
|---|---|
| [test_metrics.csv](results/test_metrics.csv) | Results for each annotated test frame and method |
| [summary.csv](results/summary.csv) | Average pixel-level scores and mask time |
| [F1 graph](results/comparison_f1.png) | Average F1 by method |
| [IoU graph](results/comparison_iou.png) | Average IoU by method |
| [Processing-time graph](results/comparison_mask_ms.png) | Average mask generation time |
| [Baseline and SVM end-to-end summary](results/end_to_end_baseline_svm.csv) | Full-pipeline results for those two methods |
| [Other methods' end-to-end summary](results/end_to_end_other_methods.csv) | Full-pipeline results for the other four methods |

The full per-frame masks, annotated images, trained models, and annotation data remain under the local `outputs/` directory, which is excluded from Git. The CSVs and graphs above are the evidence included with this repository.

## Notes on the experiment

The test annotations cover selected strokes rather than every pixel in each image. We combined yellow and white into one marking class for scoring, and the test set does not independently measure white-only performance after the white-marking example was moved into training. Some learned methods also labeled large pavement regions as markings. The end-to-end component limit made those runs manageable, but skipped frames should not be counted as correct detections. The separate assignment report discusses these results and limitations in more detail.

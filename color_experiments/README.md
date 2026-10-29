# Assignment 5 — ALINA Color Experiments

Run commands from the ALINA repository root, after activating `.venv`.

Install any missing dependencies:

```powershell
python -m pip install scikit-learn joblib pandas matplotlib
```

Keep the 15 completed annotations in `outputs/color_annotations/`. Nine frames under `train/` are used for fitting; six frames under `test/` are reserved for evaluation. The original numeric label masks are unchanged: 0=unlabeled, 1=background, 2=yellow, 3=white.

```powershell
python color_experiments/train_models.py
python color_experiments/evaluate.py
python color_experiments/analyze_results.py
```

The six compared approaches are original ALINA HSV thresholds (baseline), K-Means, Gaussian mixture model, Random Forest, SVM, and adaptive per-frame K-Means. Seeds are fixed to 42. Models are saved to `outputs/color_experiments/models.joblib`; results are written to `test_metrics.csv`, `summary.csv`, and three comparison figures. Method masks and overlays are saved under `previews/`.

**Evaluation note:** Pixel-level scores use only the hand-marked pixels in the six test frames, not the entire frame. They are binary marking-vs-background scores; yellow and white are combined for scoring. This initial evaluation measures the color mask only, not the downstream CIRCLEDAT result. A separate full-pipeline experiment and video demonstration are still required for the final submission.

**Method limitations:** Cluster-to-class mapping is learned from training labels. Adaptive K-Means matches per-frame cluster centers to class prototypes and may be unstable under lighting changes. The HSV baseline uses the current default ALINA thresholds. All models use normalized HSV features; supervised models also use original BGR channels. The normalization itself depends on the full frame and can change with lighting.

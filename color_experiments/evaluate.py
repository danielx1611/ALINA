"""Compare all color methods on held-out annotated ALINA frames."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from color_methods import METHODS, predict_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("outputs/color_annotations"))
    parser.add_argument("--models", type=Path, default=Path("outputs/color_experiments/models.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/color_experiments"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()

    # Keep test labels separate from the data used to fit the models.

    saved = joblib.load(args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    label_paths = sorted((args.annotations / "test").glob("*/*/labels.png"))
    if not label_paths:
        raise ValueError("No test annotations found")

    for label_path in label_paths:
        image = cv2.imread(str(label_path.parent / "warped.png"))
        labels = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or labels is None:
            raise ValueError(f"Missing image or labels: {label_path}")
        if image.shape[:2] != labels.shape:
            raise ValueError(f"Image and labels have different sizes: {label_path}")

        known = labels > 0
        if not np.any(known):
            print(f"Skipping empty labels: {label_path}")
            continue
        truth = (labels[known] >= 2).astype(np.uint8)
        video, frame = label_path.parent.parent.name, label_path.parent.name

        for method in args.methods:
            start = time.perf_counter()
            mask = predict_mask(image, method, saved["models"], seed=saved["seed"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            predicted = (mask[known] > 0).astype(np.uint8)
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, predicted, average="binary", zero_division=0)
            iou = jaccard_score(truth, predicted, zero_division=0)
            rows.append({"video": video, "frame": frame, "method": method,
                         "precision": round(float(precision), 5), "recall": round(float(recall), 5),
                         "f1": round(float(f1), 5), "iou": round(float(iou), 5),
                         "mask_ms": round(elapsed_ms, 2), "labeled_pixels": int(known.sum()),
                         "positive_pixels": int(truth.sum())})

            # Save previews for checking false positives and missed markings.
            preview_dir = args.output / "previews" / video / frame
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"{method}_mask.png"), mask)
            overlay = image.copy()
            overlay[mask > 0] = (0, 0, 255)
            cv2.imwrite(str(preview_dir / f"{method}_overlay.jpg"),
                        cv2.addWeighted(image, 0.65, overlay, 0.35, 0))
            print(f"{video}/{frame} {method}: F1={f1:.3f}, IoU={iou:.3f}, {elapsed_ms:.0f} ms", flush=True)

    if not rows:
        raise ValueError("No test results were produced. Check the annotations and selected methods.")

    output_path = args.output / "test_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

"""Train ALINA's color models from the saved annotation strokes."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
from color_methods import fit_models, sample_training_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("outputs/color_annotations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/color_experiments/models.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Read the labeled strokes and keep the test frames untouched.
    features, labels = sample_training_pixels(args.annotations, seed=args.seed)
    print("Training samples:", {name: int(np.sum(labels == number))
                                for number, name in ((1, "background"), (2, "yellow"), (3, "white"))})

    # Save every fitted method together for consistent evaluation.
    models = fit_models(features, labels, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "seed": args.seed}, args.output)
    print(f"Saved models to {args.output}")


if __name__ == "__main__":
    main()

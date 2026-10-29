"""Create a summary table and comparison graphs from the held-out test data."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Summarize the held-out frames before drawing the comparison charts.
root = Path("outputs/color_experiments")
results = pd.read_csv(root / "test_metrics.csv")
summary = (
    results.groupby("method")[["precision", "recall", "f1", "iou", "mask_ms"]]
    .mean()
    .sort_values("f1", ascending=False)
)
summary.to_csv(root / "summary.csv", float_format="%.4f")
print(summary.round(4).to_string())

for metric, title in (("f1", "Pixel F1 on annotated test pixels"),
                      ("iou", "Pixel IoU on annotated test pixels"),
                      ("mask_ms", "Average mask generation time (ms)")):
    ax = summary[metric].sort_values(ascending=False).plot.bar(figsize=(10, 5))
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(root / f"comparison_{metric}.png", dpi=160)
    plt.close()

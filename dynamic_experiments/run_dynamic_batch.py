"""Dynamic ROI evaluation for ALINA. Run from the repository root.

Each evaluated frame gets its own ROI. The original ALINA pipeline is unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import process_image
from alina.roi import compute_display_scale
from roi_methods import METHODS, select_roi

FIELDS = ["video", "frame", "method", "status", "roi_ms", "alina_ms",
          "total_ms", "detected_pixels", "had_lines", "cache_hit", "error"]


def alina_roi(quad):
    bl, tl, tr, br = np.asarray(quad, dtype=np.int32)
    return np.array([bl, tl, tl, tr, tr, br], dtype=np.int32).reshape(1, 6, 2)


def read_rows(path):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["video"], r["frame"], r["method"]): r for r in csv.DictReader(f)}


def save_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    tmp.replace(path)


def frame_subset(paths, count):
    # Evenly spaced indices including the first and last frames; no duplicates.
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic per-frame ROI selection")
    parser.add_argument("--data-dir", default="data/Raw_Data")
    parser.add_argument("--output-dir", default="outputs/dynamic_experiments")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--gemini-per-video", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemini-model", default="gemini-flash-lite-latest")
    parser.add_argument("--max-selection-dim", type=int, default=1080)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.gemini_per_video < 1 or args.max_selection_dim < 1:
        parser.error("gemini-per-video and max-selection-dim must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    cv2.setRNGSeed(args.seed)

    data_dir, out = Path(args.data_dir), Path(args.output_dir)
    videos = sorted(p for p in data_dir.iterdir() if p.is_dir() and list(p.glob("*.jpg"))) if data_dir.exists() else []
    if not videos:
        parser.error(f"No video folders containing JPG images under {data_dir}")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    rows = read_rows(csv_path)
    manifest = {"seed": args.seed, "gemini_model": args.gemini_model,
                "max_selection_dim": args.max_selection_dim,
                "gemini_per_video": args.gemini_per_video,
                "methods": list(args.methods),
                "videos": {p.name: len(list(p.glob('*.jpg'))) for p in videos}}
    manifest_path = out / "experiment_config.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k in ("seed", "gemini_model", "max_selection_dim", "gemini_per_video"):
            if old.get(k) != manifest[k]:
                parser.error(f"Existing results use a different {k}; choose another --output-dir")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for video in videos:
        paths = sorted(video.glob("*.jpg"))
        gemini_paths = set(frame_subset(paths, args.gemini_per_video))
        print(f"\n{video.name}: {len(paths)} frames, {len(gemini_paths)} Gemini samples", flush=True)
        for path in paths:
            selected_methods = [m for m in args.methods if m != "gemini" or path in gemini_paths]
            if not selected_methods:
                continue
            if all((video.name, path.name, m) in rows and
                   (rows[(video.name, path.name, m)]["status"] == "ok" or not args.retry_failed)
                   for m in selected_methods):
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"WARNING: Cannot read {path}", flush=True)
                continue
            scale = compute_display_scale(img, max_display_dim=args.max_selection_dim)
            selection = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
            sh, sw = selection.shape[:2]
            config = PipelineConfig(input_dir=video, output_images_dir=out, output_coords_dir=out)
            for method in selected_methods:
                key = (video.name, path.name, method)
                if key in rows and (rows[key]["status"] == "ok" or not args.retry_failed):
                    continue
                folder = out / video.name / method
                folder.mkdir(parents=True, exist_ok=True)
                cache = folder / f"{path.stem}_gemini_{sw}x{sh}_{args.gemini_model}.json"
                actual_cache = cache.with_name(cache.stem + "_bbox1000_v1" + cache.suffix)
                timing_file = folder / f"{path.stem}_gemini_inference_ms.json"
                cache_hit = method == "gemini" and actual_cache.exists()
                roi_ms = alina_ms = None
                try:
                    start = time.perf_counter()
                    quad_small = select_roi(selection, method, args.seed, cache, args.gemini_model)
                    roi_ms = (time.perf_counter() - start) * 1000
                    if method == "gemini" and not cache_hit:
                        timing_file.write_text(json.dumps({"inference_ms": roi_ms}), encoding="utf-8")
                    quad = np.rint(np.asarray(quad_small, dtype=np.float64) / scale).astype(np.int32)
                    quad[:, 0] = np.clip(quad[:, 0], 0, img.shape[1] - 1)
                    quad[:, 1] = np.clip(quad[:, 1], 0, img.shape[0] - 1)
                    (folder / f"{path.stem}_roi_selection.json").write_text(json.dumps(np.asarray(quad_small).astype(int).tolist(), indent=2), encoding="utf-8")
                    (folder / f"{path.stem}_roi.json").write_text(json.dumps(quad.tolist(), indent=2), encoding="utf-8")
                    preview = selection.copy()
                    cv2.polylines(preview, [np.asarray(quad_small, dtype=np.int32)], True, (0, 0, 255), 2)
                    cv2.imwrite(str(folder / f"{path.stem}_roi_selection.jpg"), preview)
                    preview_full = img.copy()
                    cv2.polylines(preview_full, [quad], True, (0, 0, 255), 3)
                    cv2.imwrite(str(folder / f"{path.stem}_roi.jpg"), preview_full)
                    start = time.perf_counter()
                    annotated, coords, had_lines = process_image(img, alina_roi(quad), config)
                    alina_ms = (time.perf_counter() - start) * 1000
                    cv2.imwrite(str(folder / f"{path.stem}_annotated.jpg"), annotated)
                    np.savetxt(folder / f"{path.stem}_coords.txt", coords, fmt="%d")
                    row = dict(video=video.name, frame=path.name, method=method, status="ok",
                               roi_ms=round(roi_ms, 2), alina_ms=round(alina_ms, 2),
                               total_ms=round(roi_ms + alina_ms, 2),
                               detected_pixels=len(coords), had_lines=bool(had_lines),
                               cache_hit=bool(cache_hit), error="")
                except Exception as exc:
                    row = dict(video=video.name, frame=path.name, method=method, status="failed",
                               roi_ms="" if roi_ms is None else round(roi_ms, 2),
                               alina_ms="" if alina_ms is None else round(alina_ms, 2),
                               total_ms="", detected_pixels="", had_lines="",
                               cache_hit=bool(cache_hit), error=f"{type(exc).__name__}: {exc}")
                rows[key] = row
                save_rows(csv_path, rows)
                print(f"[{video.name}/{path.name}/{method}] {row['status']} "
                      f"pixels={row['detected_pixels']} roi_ms={row['roi_ms']} "
                      f"alina_ms={row['alina_ms']} {row['error']}", flush=True)

    print(f"\nFinished. Results: {csv_path.resolve()}")
    for method in args.methods:
        selected = [r for r in rows.values() if r["method"] == method]
        success = sum(r["status"] == "ok" and r["had_lines"] in (True, "True") for r in selected)
        failed = sum(r["status"] == "failed" for r in selected)
        print(f"{method}: {len(selected)} evaluated; {success} with lines; {failed} errors")
    print("Detected pixel count is NOT an accuracy metric. CBEM ground truth is needed for precision/recall/F1.")


if __name__ == "__main__":
    main()

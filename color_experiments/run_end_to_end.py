"""Compare color methods inside ALINA's full line-detection pipeline.

Run from the repository root:
    python color_experiments/run_end_to_end.py --frames-per-video 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from alina.config import PipelineConfig
from alina.histogram import calc_histogram
from alina.traversal import circular_threshold_pixel_discovery_and_traversal
from color_methods import METHODS, predict_mask


RESULT_COLUMNS = (
    "video", "frame", "method", "had_lines", "detected_pixels",
    "mask_ms", "downstream_ms", "total_ms", "mask_coverage",
    "candidate_pixels", "component_pixels", "valid_pixels",
    "valid_mask_coverage", "status", "error",
)


def choose_frames(video_folder: Path, frame_limit: int) -> list[Path]:
    """Use evenly spaced frames to cover different parts of each video."""
    frames = sorted(video_folder.glob("*.jpg"))
    if frame_limit <= 0 or frame_limit >= len(frames):
        return frames

    selected_indices = np.linspace(0, len(frames) - 1, frame_limit, dtype=int)
    return [frames[index] for index in np.unique(selected_indices)]


def load_roi(annotation_folder: Path, video_name: str) -> np.ndarray:
    """Read the four corners selected during the annotation stage."""
    roi_file = annotation_folder / f"{video_name}_roi.json"
    if not roi_file.exists():
        raise FileNotFoundError(f"Missing ROI file: {roi_file}")

    corners = np.asarray(json.loads(roi_file.read_text(encoding="utf-8")), dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError(f"Expected four ROI corners in {roi_file}")
    return corners


def warp_image(image: np.ndarray, corners: np.ndarray, config: PipelineConfig):
    """Apply the same perspective transform used by ALINA."""
    destination = np.asarray([
        config.roi.dst_bottom_left,
        config.roi.dst_top_left,
        config.roi.dst_top_right,
        config.roi.dst_bottom_right,
    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(corners, destination)
    warped_image = cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))

    # Warp a filled ROI as well, so pixels outside the actual pavement
    # selection cannot be mistaken for line markings.
    source_region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(source_region, np.rint(corners).astype(np.int32), 255)

    valid_region = cv2.warpPerspective(
        source_region,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
    )

    # Keep the valid region inside the destination rectangle.
    destination_region = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(
        destination_region, np.rint(destination).astype(np.int32), 255
    )
    valid_region = cv2.bitwise_and(valid_region, destination_region)

    return warped_image, destination, valid_region


def detect_from_mask(image, mask, corners, destination, config, max_component_pixels):
    """Check the selected region before running ALINA's slower traversal."""
    candidate_mask = mask.copy()
    if config.mask_ignore_left_columns > 0:
        candidate_mask[:, :config.mask_ignore_left_columns] = 0

    candidate_pixels = int(cv2.countNonZero(candidate_mask))
    peak_pixel, peak_value = calc_histogram(
        candidate_mask, min_white_pixels=config.min_white_pixels
    )

    empty_coordinates = np.empty((0, 2), dtype=np.int32)
    if not peak_pixel or peak_value <= config.peak_pixel_threshold:
        return image.copy(), empty_coordinates, False, "no_peak", candidate_pixels, 0

    # ALINA visits pixels within a square around each discovered pixel.
    # Dilating once with that same square lets us estimate the full region
    # before the slower Python traversal starts.
    radius = config.circular_threshold
    neighborhood = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1)
    )
    expanded_mask = cv2.dilate(candidate_mask, neighborhood)
    component_count, component_labels = cv2.connectedComponents(
        expanded_mask, connectivity=8
    )

    seed_x, seed_y = peak_pixel
    selected_component = component_labels[seed_y, seed_x]
    if selected_component == 0:
        return image.copy(), empty_coordinates, False, "no_peak", candidate_pixels, 0

    # Count original candidate pixels, not the extra pixels from dilation.
    selected_pixels = int(np.count_nonzero(
        (component_labels == selected_component) & (candidate_mask > 0)
    ))

    if selected_pixels > max_component_pixels:
        return (image.copy(), empty_coordinates, False,
                "skipped_excessive", candidate_pixels, selected_pixels)

    marking_pixels = circular_threshold_pixel_discovery_and_traversal(
        candidate_mask, seed_x, seed_y, config.circular_threshold
    )

    detected_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
    for x_position, y_position in marking_pixels:
        cv2.circle(detected_mask, (x_position, y_position), 1, 255, -1)

    reverse_transform = cv2.getPerspectiveTransform(destination, corners)
    original_mask = cv2.warpPerspective(
        detected_mask, reverse_transform, (image.shape[1], image.shape[0])
    )

    y_positions, x_positions = np.where(original_mask > 0)
    coordinates = np.column_stack((x_positions, y_positions))
    annotated_image = image.copy()
    annotated_image[y_positions, x_positions] = (0, 0, 255)

    return (annotated_image, coordinates, bool(len(marking_pixels)),
            "ok", candidate_pixels, selected_pixels)


def create_video(image_folder: Path, video_file: Path, fps: int) -> bool:
    """Combine the saved frames into an MP4 demonstration."""
    images = sorted(image_folder.glob("*.jpg"))
    if not images:
        return False

    first_image = cv2.imread(str(images[0]))
    if first_image is None:
        return False

    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(video_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print(f"WARNING: Could not create video: {video_file}")
        return False

    try:
        for image_file in images:
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate color methods through ALINA")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Raw_Data"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/color_annotations"))
    parser.add_argument("--models", type=Path, default=Path("outputs/color_experiments/models.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/color_experiments/end_to_end"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--frames-per-video", type=int, default=15)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--max-component-pixels", type=int, default=100000,
                        help="Skip traversal when the selected region exceeds this many pixels")
    arguments = parser.parse_args()

    if arguments.fps < 1 or arguments.max_component_pixels < 1:
        parser.error("FPS and max-component-pixels must be positive")

    saved_models = joblib.load(arguments.models)
    trained_models = saved_models["models"]
    training_seed = saved_models.get("seed", arguments.seed)
    arguments.output.mkdir(parents=True, exist_ok=True)

    # Keep the limit and ROI masking setting with the results for reproducibility.
    settings_file = arguments.output / "run_settings.json"
    current_settings = {
        "max_component_pixels": arguments.max_component_pixels,
        "valid_roi_masking": True,
        "method_names": list(arguments.methods),
        "frames_per_video": arguments.frames_per_video,
        "seed": training_seed,
        "annotations": str(arguments.annotations),
        "models": str(arguments.models),
    }
    if settings_file.exists():
        old_settings = json.loads(settings_file.read_text(encoding="utf-8"))
        if old_settings != current_settings:
            parser.error("Output folder has different run settings. Use a fresh --output folder.")
    else:
        settings_file.write_text(json.dumps(current_settings, indent=2), encoding="utf-8")

    # Store each result immediately so interrupted runs still leave useful data.
    results_file = arguments.output / "results.csv"
    existing_results = {}
    if results_file.exists():
        with results_file.open(newline="", encoding="utf-8") as saved_file:
            for row in csv.DictReader(saved_file):
                existing_results[(row["video"], row["frame"], row["method"])] = row

    video_folders = sorted(
        folder for folder in arguments.data_dir.iterdir()
        if folder.is_dir() and any(folder.glob("*.jpg"))
    ) if arguments.data_dir.exists() else []
    if not video_folders:
        parser.error(f"No video frames found in {arguments.data_dir}")

    for video_folder in video_folders:
        corners = load_roi(arguments.annotations, video_folder.name)
        config = PipelineConfig(
            input_dir=video_folder,
            output_images_dir=arguments.output,
            output_coords_dir=arguments.output,
        )

        for frame_file in choose_frames(video_folder, arguments.frames_per_video):
            original_image = cv2.imread(str(frame_file))
            if original_image is None:
                print(f"Skipping unreadable frame: {frame_file}")
                continue

            warped_image, destination, valid_region = warp_image(
                original_image, corners, config
            )
            valid_pixels = int(cv2.countNonZero(valid_region))
            if valid_pixels == 0:
                raise ValueError(f"Selected ROI has no valid pixels: {video_folder.name}")

            for method in arguments.methods:
                result_key = (video_folder.name, frame_file.stem, method)
                if result_key in existing_results and existing_results[result_key]["status"] in ("ok", "no_peak", "skipped_excessive"):
                    continue

                frame_folder = arguments.output / video_folder.name / method
                frame_folder.mkdir(parents=True, exist_ok=True)

                mask_time = 0.0
                downstream_time = 0.0
                mask_coverage = None
                candidate_pixels = None
                component_pixels = None
                try:
                    start_time = time.perf_counter()
                    mask = predict_mask(warped_image, method, trained_models, seed=training_seed)
                    mask_time = (time.perf_counter() - start_time) * 1000

                    # Remove predictions from the black padding created by warping.
                    # Keep the unfiltered coverage as a diagnostic.
                    raw_mask_coverage = float(np.count_nonzero(mask) / mask.size)
                    mask = cv2.bitwise_and(mask, valid_region)
                    valid_marking_pixels = int(cv2.countNonZero(mask))
                    valid_mask_coverage = valid_marking_pixels / valid_pixels
                    mask_coverage = float(valid_marking_pixels / mask.size)

                    start_time = time.perf_counter()
                    (annotated, coordinates, found_lines, detection_status,
                     candidate_pixels, component_pixels) = detect_from_mask(
                        original_image, mask, corners, destination, config,
                        arguments.max_component_pixels
                    )
                    downstream_time = (time.perf_counter() - start_time) * 1000

                    # Keep a mask and original-perspective result for later review.
                    cv2.imwrite(str(frame_folder / f"{frame_file.stem}_mask.png"), mask)
                    cv2.imwrite(
                        str(frame_folder / f"{frame_file.stem}_valid_roi.png"),
                        valid_region,
                    )
                    cv2.imwrite(str(frame_folder / f"{frame_file.stem}.jpg"), annotated)
                    np.savetxt(frame_folder / f"{frame_file.stem}_coords.txt", coordinates, fmt="%d")

                    row = {
                        "video": video_folder.name,
                        "frame": frame_file.stem,
                        "method": method,
                        "had_lines": int(found_lines),
                        "detected_pixels": len(coordinates),
                        "mask_ms": round(mask_time, 2),
                        "downstream_ms": round(downstream_time, 2),
                        "total_ms": round(mask_time + downstream_time, 2),
                        "mask_coverage": round(mask_coverage, 6),
                        "candidate_pixels": candidate_pixels,
                        "component_pixels": component_pixels,
                        "valid_pixels": valid_pixels,
                        "valid_mask_coverage": round(valid_mask_coverage, 6),
                        "status": detection_status,
                        "error": "",
                    }
                except Exception as error:
                    row = {
                        "video": video_folder.name,
                        "frame": frame_file.stem,
                        "method": method,
                        "had_lines": "",
                        "detected_pixels": "",
                        "mask_ms": round(mask_time, 2),
                        "downstream_ms": round(downstream_time, 2),
                        "total_ms": "",
                        "mask_coverage": "" if mask_coverage is None else round(mask_coverage, 6),
                        "candidate_pixels": "" if candidate_pixels is None else candidate_pixels,
                        "component_pixels": "" if component_pixels is None else component_pixels,
                        "valid_pixels": valid_pixels,
                        "valid_mask_coverage": "",
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }

                existing_results[result_key] = row
                with results_file.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
                    writer.writeheader()
                    writer.writerows(existing_results[key] for key in sorted(existing_results))

                print(
                    f"{video_folder.name}/{frame_file.stem} {method}: "
                    f"{row['status']}, pixels={row['detected_pixels']}, "
                    f"mask={row['mask_ms']} ms, downstream={row['downstream_ms']} ms, "
                    f"coverage={row['mask_coverage']}, "
                    f"valid_roi_coverage={row['valid_mask_coverage']}, "
                    f"component={row['component_pixels']}, total={row['total_ms']} ms "
                    f"{row['error']}",
                    flush=True,
                )

    if not arguments.skip_videos:
        for video_folder in video_folders:
            for method in arguments.methods:
                frame_folder = arguments.output / video_folder.name / method
                video_file = arguments.output / f"{video_folder.name}_{method}.mp4"
                if create_video(frame_folder, video_file, arguments.fps):
                    print(f"Saved video: {video_file}")

    # Summarize only completed runs; failures are shown separately in the CSV.
    import pandas as pd

    results = pd.read_csv(results_file)
    completed = results[results["status"].isin(["ok", "no_peak", "skipped_excessive"])].copy()
    if completed.empty:
        raise RuntimeError("No successful runs to summarize")

    for column in ("had_lines", "detected_pixels", "mask_ms", "downstream_ms",
                   "total_ms", "mask_coverage", "candidate_pixels", "component_pixels",
                   "valid_pixels", "valid_mask_coverage"):
        completed[column] = pd.to_numeric(completed[column], errors="coerce")

    summary = completed.groupby("method").agg(
        frames=("had_lines", "size"),
        detection_rate=("had_lines", "mean"),
        average_detected_pixels=("detected_pixels", "mean"),
        average_mask_ms=("mask_ms", "mean"),
        average_total_ms=("total_ms", "mean"),
        average_mask_coverage=("mask_coverage", "mean"),
        average_valid_roi_coverage=("valid_mask_coverage", "mean"),
        excessive_frames=("status", lambda values: int((values == "skipped_excessive").sum())),
        no_peak_frames=("status", lambda values: int((values == "no_peak").sum())),
    ).sort_values("detection_rate", ascending=False)

    summary.to_csv(arguments.output / "summary.csv")
    print("\nEnd-to-end summary (skipped excessive regions are NOT successful detections):")
    print(summary.round(3).to_string())
    print(f"\nSaved results in {arguments.output}")


if __name__ == "__main__":
    main()

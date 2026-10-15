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

# Repeat the two upper corners to match ALINA’s six-point input format.
def to_alina_roi(quad):
    """ALINA expects six clicks, though only four form its trapezoid."""
    bottom_left, top_left, top_right, bottom_right = np.asarray(quad, dtype=np.int32)
    return np.array([bottom_left, top_left, top_left, top_right, top_right, bottom_right], dtype=np.int32).reshape(1, 6, 2)

# Read the command-line options and set up the experiment folders.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='data/Raw_Data/vidd_1/00001.jpg')
    parser.add_argument('--output-dir', default='outputs/automated_experiments_v2')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gemini-model', default='gemini-flash-lite-latest')
    parser.add_argument('--max-selection-dim', type=int, default=1080)
    options = parser.parse_args()

    # Keep the random methods consistent across repeated runs.
    random.seed(options.seed)
    np.random.seed(options.seed)
    cv2.setRNGSeed(options.seed)
    image_path = Path(options.image)
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise SystemExit(f'Cannot read {image_path}')

    # Select ROIs on a smaller image, just like the manual ALINA window.
    display_scale = compute_display_scale(original_image, max_display_dim=options.max_selection_dim)
    selection_image = cv2.resize(original_image, None, fx=display_scale, fy=display_scale) if display_scale != 1.0 else original_image.copy()
    selection_height, selection_width = selection_image.shape[:2]
    output_folder = Path(options.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    pipeline_config = PipelineConfig(input_dir=image_path.parent, output_images_dir=output_folder, output_coords_dir=output_folder)
    print(f'Original image: {original_image.shape[1]}x{original_image.shape[0]}; selection image: {selection_width}x{selection_height}; scale: {display_scale:.6f}')
    result_rows = []

    # Run each method on the same input image.
    for method_name in options.methods:
        method_folder = output_folder / method_name
        method_folder.mkdir(parents=True, exist_ok=True)
        gemini_prediction_file = method_folder / f'{image_path.stem}_gemini_{selection_width}x{selection_height}_{options.gemini_model}.json'
        roi_time_ms = None
        alina_time_ms = None
        try:
            start_time = time.perf_counter()

            # Time the ROI selection separately from ALINA processing.
            selection_corners = select_roi(selection_image, method_name, options.seed, gemini_prediction_file, options.gemini_model)
            roi_time_ms = (time.perf_counter() - start_time) * 1000

            # Map the selected corners back to the original image size.
            full_size_corners = np.rint(np.asarray(selection_corners, dtype=np.float64) / display_scale).astype(np.int32)
            full_size_corners[:, 0] = np.clip(full_size_corners[:, 0], 0, original_image.shape[1] - 1)
            full_size_corners[:, 1] = np.clip(full_size_corners[:, 1], 0, original_image.shape[0] - 1)

            # Save the selected coordinates and preview images for inspection.
            (method_folder / f'{image_path.stem}_roi_selection.json').write_text(json.dumps(np.asarray(selection_corners).astype(int).tolist(), indent=2))
            (method_folder / f'{image_path.stem}_roi.json').write_text(json.dumps(full_size_corners.tolist(), indent=2))
            selection_preview = selection_image.copy()
            cv2.polylines(selection_preview, [np.asarray(selection_corners, dtype=np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(str(method_folder / f'{image_path.stem}_roi_selection.jpg'), selection_preview)
            full_size_preview = original_image.copy()
            cv2.polylines(full_size_preview, [full_size_corners], True, (0, 0, 255), 3)
            cv2.imwrite(str(method_folder / f'{image_path.stem}_roi.jpg'), full_size_preview)
            start_time = time.perf_counter()

            # Run the unchanged ALINA detector on the selected region.
            annotated_image, detected_coordinates, found_lines = process_image(original_image, to_alina_roi(full_size_corners), pipeline_config)
            alina_time_ms = (time.perf_counter() - start_time) * 1000
            cv2.imwrite(str(method_folder / f'{image_path.stem}_annotated.jpg'), annotated_image)
            np.savetxt(method_folder / f'{image_path.stem}_coords.txt', detected_coordinates, fmt='%d')
            result_row = dict(method=method_name, status='ok', roi_ms=round(roi_time_ms, 2), alina_ms=round(alina_time_ms, 2), total_ms=round(roi_time_ms + alina_time_ms, 2), detected_pixels=len(detected_coordinates), had_lines=found_lines, error='')
        except Exception as exc:
            result_row = dict(method=method_name, status='failed', roi_ms='' if roi_time_ms is None else round(roi_time_ms, 2), alina_ms='' if alina_time_ms is None else round(alina_time_ms, 2), total_ms='', detected_pixels='', had_lines='', error=str(error))
        result_rows.append(result_row)
        print(result_row, flush=True)

    # Save one summary row per method.
    with (output_folder / 'summary.csv').open('w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(result_rows)
    print(f'Saved outputs to {output_folder.resolve()}')
if __name__ == '__main__':
    main()

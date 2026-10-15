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
FIELDS = ['video', 'frame', 'method', 'status', 'roi_ms', 'alina_ms', 'total_ms', 'detected_pixels', 'had_lines', 'cache_hit', 'error']

# Repeat the two upper corners to match ALINA’s six-point input format.
def alina_roi(quad):
    bottom_left, top_left, top_right, bottom_right = np.asarray(quad, dtype=np.int32)
    return np.array([bottom_left, top_left, top_left, top_right, top_right, bottom_right], dtype=np.int32).reshape(1, 6, 2)

# Load completed runs so restarting the batch does not repeat finished frames.
def read_rows(path):
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as results_file:
        return {(record['video'], record['frame'], record['method']): record for record in csv.DictReader(results_file)}

# Write the results through a temporary file to avoid losing progress.
def save_rows(path, rows):
    temporary_path = path.with_suffix('.tmp')
    with temporary_path.open('w', newline='', encoding='utf-8') as results_file:
        writer = csv.DictWriter(results_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows((rows[record_key] for record_key in sorted(rows)))
    temporary_path.replace(path)

# Choose evenly spaced frames for the smaller Gemini sample.
def frame_subset(paths, count):
    if count >= len(paths):
        return paths
    selected_indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[frame_index] for frame_index in selected_indices]

# Read the command-line options and set up the experiment folders.
def main():
    argument_parser = argparse.ArgumentParser(description='Evaluate dynamic per-frame ROI selection')
    argument_parser.add_argument('--data-dir', default='data/Raw_Data')
    argument_parser.add_argument('--output-dir', default='outputs/dynamic_experiments')
    argument_parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    argument_parser.add_argument('--gemini-per-video', type=int, default=10)
    argument_parser.add_argument('--seed', type=int, default=42)
    argument_parser.add_argument('--gemini-model', default='gemini-flash-lite-latest')
    argument_parser.add_argument('--max-selection-dim', type=int, default=1080)
    argument_parser.add_argument('--retry-failed', action='store_true')
    options = argument_parser.parse_args()
    if options.gemini_per_video < 1 or options.max_selection_dim < 1:
        argument_parser.error('gemini-per-video and max-selection-dim must be positive')

    # Use the same random seed for all traditional methods.
    random.seed(options.seed)
    np.random.seed(options.seed)
    cv2.setRNGSeed(options.seed)
    data_folder, output_folder = (Path(options.data_dir), Path(options.output_dir))

    # Find the image folders included in the experiment.
    video_folders = sorted((folder for folder in data_folder.iterdir() if folder.is_dir() and list(folder.glob('*.jpg')))) if data_folder.exists() else []
    if not video_folders:
        argument_parser.error(f'No video folders containing JPG images under {data_folder}')
    output_folder.mkdir(parents=True, exist_ok=True)
    results_csv = output_folder / 'results.csv'
    saved_results = read_rows(results_csv)

    # Record the settings used for this set of results.
    run_settings = {'seed': options.seed, 'gemini_model': options.gemini_model, 'max_selection_dim': options.max_selection_dim, 'gemini_per_video': options.gemini_per_video, 'methods': list(options.methods), 'videos': {folder.name: len(list(folder.glob('*.jpg'))) for folder in video_folders}}
    settings_file = output_folder / 'experiment_config.json'
    if settings_file.exists():
        previous_settings = json.loads(settings_file.read_text(encoding='utf-8'))
        for setting_name in ('seed', 'gemini_model', 'max_selection_dim', 'gemini_per_video'):
            if previous_settings.get(setting_name) != run_settings[setting_name]:
                argument_parser.error(f'Existing results use a different {setting_name}; choose another --output-dir')
    else:
        settings_file.write_text(json.dumps(run_settings, indent=2), encoding='utf-8')

    # Go through the videos and generate a fresh ROI for each frame.
    for video_folder in video_folders:
        image_paths = sorted(video_folder.glob('*.jpg'))
        gemini_frames = set(frame_subset(image_paths, options.gemini_per_video))
        print(f'\n{video_folder.name}: {len(image_paths)} frames, {len(gemini_frames)} Gemini samples', flush=True)
        for image_path in image_paths:
            methods_for_frame = [method_name for method_name in options.methods if method_name != 'gemini' or image_path in gemini_frames]
            if not methods_for_frame:
                continue

            # Skip frames whose requested methods have already finished.
            if all(((video_folder.name, image_path.name, method_name) in saved_results and (saved_results[video_folder.name, image_path.name, method_name]['status'] == 'ok' or not options.retry_failed) for method_name in methods_for_frame)):
                continue
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                print(f'WARNING: Cannot read {image_path}', flush=True)
                continue

            # Resize only for ROI selection; ALINA still receives the original image.
            display_scale = compute_display_scale(original_image, max_display_dim=options.max_selection_dim)
            selection_image = cv2.resize(original_image, None, fx=display_scale, fy=display_scale) if display_scale != 1.0 else original_image.copy()
            selection_height, selection_width = selection_image.shape[:2]
            pipeline_config = PipelineConfig(input_dir=video_folder, output_images_dir=output_folder, output_coords_dir=output_folder)

            # Save the result after each method so interrupted runs can continue.
            for method_name in methods_for_frame:
                result_key = (video_folder.name, image_path.name, method_name)
                if result_key in saved_results and (saved_results[result_key]['status'] == 'ok' or not options.retry_failed):
                    continue
                method_folder = output_folder / video_folder.name / method_name
                method_folder.mkdir(parents=True, exist_ok=True)
                gemini_prediction_file = method_folder / f'{image_path.stem}_gemini_{selection_width}x{selection_height}_{options.gemini_model}.json'
                stored_prediction_file = gemini_prediction_file.with_name(gemini_prediction_file.stem + '_bbox1000_v1' + gemini_prediction_file.suffix)
                inference_time_file = method_folder / f'{image_path.stem}_gemini_inference_ms.json'
                used_saved_prediction = method_name == 'gemini' and stored_prediction_file.exists()
                roi_time_ms = alina_time_ms = None
                try:
                    start_time = time.perf_counter()

                    # Measure how long the method takes to choose its ROI.
                    selection_corners = select_roi(selection_image, method_name, options.seed, gemini_prediction_file, options.gemini_model)
                    roi_time_ms = (time.perf_counter() - start_time) * 1000
                    if method_name == 'gemini' and (not used_saved_prediction):
                        inference_time_file.write_text(json.dumps({'inference_ms': roi_time_ms}), encoding='utf-8')

                    # Convert the corners back to full-resolution coordinates.
                    full_size_corners = np.rint(np.asarray(selection_corners, dtype=np.float64) / display_scale).astype(np.int32)
                    full_size_corners[:, 0] = np.clip(full_size_corners[:, 0], 0, original_image.shape[1] - 1)
                    full_size_corners[:, 1] = np.clip(full_size_corners[:, 1], 0, original_image.shape[0] - 1)
                    (method_folder / f'{image_path.stem}_roi_selection.json').write_text(json.dumps(np.asarray(selection_corners).astype(int).tolist(), indent=2), encoding='utf-8')
                    (method_folder / f'{image_path.stem}_roi.json').write_text(json.dumps(full_size_corners.tolist(), indent=2), encoding='utf-8')
                    selection_preview = selection_image.copy()
                    cv2.polylines(selection_preview, [np.asarray(selection_corners, dtype=np.int32)], True, (0, 0, 255), 2)
                    cv2.imwrite(str(method_folder / f'{image_path.stem}_roi_selection.jpg'), selection_preview)
                    full_size_preview = original_image.copy()
                    cv2.polylines(full_size_preview, [full_size_corners], True, (0, 0, 255), 3)
                    cv2.imwrite(str(method_folder / f'{image_path.stem}_roi.jpg'), full_size_preview)
                    start_time = time.perf_counter()

                    # Pass the ROI into the original ALINA processing code.
                    annotated_image, detected_coordinates, found_lines = process_image(original_image, alina_roi(full_size_corners), pipeline_config)
                    alina_time_ms = (time.perf_counter() - start_time) * 1000
                    cv2.imwrite(str(method_folder / f'{image_path.stem}_annotated.jpg'), annotated_image)
                    np.savetxt(method_folder / f'{image_path.stem}_coords.txt', detected_coordinates, fmt='%d')
                    result_row = dict(video=video_folder.name, frame=image_path.name, method=method_name, status='ok', roi_ms=round(roi_time_ms, 2), alina_ms=round(alina_time_ms, 2), total_ms=round(roi_time_ms + alina_time_ms, 2), detected_pixels=len(detected_coordinates), had_lines=bool(found_lines), cache_hit=bool(used_saved_prediction), error='')
                except Exception as exc:
                    result_row = dict(video=video_folder.name, frame=image_path.name, method=method_name, status='failed', roi_ms='' if roi_time_ms is None else round(roi_time_ms, 2), alina_ms='' if alina_time_ms is None else round(alina_time_ms, 2), total_ms='', detected_pixels='', had_lines='', cache_hit=bool(used_saved_prediction), error=f'{type(error).__name__}: {error}')

                # Keep the newest result for this video, frame, and method.
                saved_results[result_key] = result_row
                save_rows(results_csv, saved_results)
                print(f"[{video_folder.name}/{image_path.name}/{method_name}] {result_row['status']} pixels={result_row['detected_pixels']} roi_ms={result_row['roi_ms']} alina_ms={result_row['alina_ms']} {result_row['error']}", flush=True)
    print(f'\nFinished. Results: {results_csv.resolve()}')
    for method_name in options.methods:
        method_results = [result for result in saved_results.values() if result['method'] == method_name]
        successful_frames = sum((result['status'] == 'ok' and result['had_lines'] in (True, 'True') for result in method_results))
        failed_frames = sum((result['status'] == 'failed' for result in method_results))
        print(f'{method_name}: {len(method_results)} evaluated; {successful_frames} with lines; {failed_frames} errors')
    print('Detected pixel count is NOT an accuracy metric. CBEM ground truth is needed for precision/recall/F1.')
if __name__ == '__main__':
    main()

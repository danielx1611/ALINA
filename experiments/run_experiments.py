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


def to_alina_roi(quad):
    """ALINA expects six clicks, though only four form its trapezoid."""
    bl, tl, tr, br = np.asarray(quad, dtype=np.int32)
    return np.array([bl, tl, tl, tr, tr, br], dtype=np.int32).reshape(1, 6, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', default='data/Raw_Data/vidd_1/00001.jpg')
    p.add_argument('--output-dir', default='outputs/automated_experiments_v2')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gemini-model', default='gemini-flash-lite-latest')
    p.add_argument('--max-selection-dim', type=int, default=1080)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); cv2.setRNGSeed(args.seed)
    image_path = Path(args.image)
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f'Cannot read {image_path}')
    scale = compute_display_scale(img, max_display_dim=args.max_selection_dim)
    selection_img = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
    sh, sw = selection_img.shape[:2]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(input_dir=image_path.parent, output_images_dir=output, output_coords_dir=output)
    print(f'Original image: {img.shape[1]}x{img.shape[0]}; selection image: {sw}x{sh}; scale: {scale:.6f}')
    rows = []
    for method in args.methods:
        folder = output / method
        folder.mkdir(parents=True, exist_ok=True)
        # New output folder and size-specific cache prevent reusing old full-res Gemini coordinates.
        cache = folder / f'{image_path.stem}_gemini_{sw}x{sh}_{args.gemini_model}.json'
        roi_ms = None
        alina_ms = None
        try:
            start = time.perf_counter()
            selection_quad = select_roi(selection_img, method, args.seed, cache, args.gemini_model)
            roi_ms = (time.perf_counter() - start) * 1000
            quad = np.rint(np.asarray(selection_quad, dtype=np.float64) / scale).astype(np.int32)
            quad[:, 0] = np.clip(quad[:, 0], 0, img.shape[1] - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, img.shape[0] - 1)
            # Save both coordinate systems before running ALINA.
            (folder / f'{image_path.stem}_roi_selection.json').write_text(json.dumps(np.asarray(selection_quad).astype(int).tolist(), indent=2))
            (folder / f'{image_path.stem}_roi.json').write_text(json.dumps(quad.tolist(), indent=2))
            preview = selection_img.copy()
            cv2.polylines(preview, [np.asarray(selection_quad, dtype=np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(str(folder / f'{image_path.stem}_roi_selection.jpg'), preview)
            preview_full = img.copy()
            cv2.polylines(preview_full, [quad], True, (0, 0, 255), 3)
            cv2.imwrite(str(folder / f'{image_path.stem}_roi.jpg'), preview_full)
            start = time.perf_counter()
            annotated, coords, had_lines = process_image(img, to_alina_roi(quad), config)
            alina_ms = (time.perf_counter() - start) * 1000
            cv2.imwrite(str(folder / f'{image_path.stem}_annotated.jpg'), annotated)
            np.savetxt(folder / f'{image_path.stem}_coords.txt', coords, fmt='%d')
            row = dict(method=method, status='ok', roi_ms=round(roi_ms,2), alina_ms=round(alina_ms,2),
                       total_ms=round(roi_ms+alina_ms,2), detected_pixels=len(coords), had_lines=had_lines, error='')
        except Exception as exc:
            row = dict(method=method, status='failed', roi_ms='' if roi_ms is None else round(roi_ms,2),
                       alina_ms='' if alina_ms is None else round(alina_ms,2), total_ms='',
                       detected_pixels='', had_lines='', error=str(exc))
        rows.append(row)
        print(row, flush=True)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Saved outputs to {output.resolve()}')

if __name__ == '__main__':
    main()

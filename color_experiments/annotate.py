"""Draw pixel labels on perspective-warped ALINA frames.

Run from the repository root:
    python color_experiments/annotate.py

The saved masks use 0 for unlabeled pixels, 1 for background,
2 for yellow markings, and 3 for white markings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from alina.config import ROIConfig
from alina.roi import compute_display_scale


WINDOW_NAME = "ALINA Color Annotations"
CLASS_NAMES = {1: "Background", 2: "Yellow", 3: "White"}
CLASS_COLORS = {1: (255, 120, 50), 2: (0, 255, 255), 3: (255, 255, 255)}


def select_roi(image: np.ndarray) -> np.ndarray | None:
    """Ask for the four corners of the runway region."""
    scale = compute_display_scale(image, max_display_dim=1100)
    preview = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    points: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_click)

    while True:
        display = preview.copy()
        for index, point in enumerate(points):
            cv2.circle(display, point, 5, (0, 0, 255), -1)
            cv2.putText(display, str(index + 1), (point[0] + 8, point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], len(points) == 4, (0, 255, 0), 2)
        cv2.putText(display, "Click: bottom-left, top-left, top-right, bottom-right", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(display, "Click: bottom-left, top-left, top-right, bottom-right", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("n") and len(points) == 4:
            return np.asarray(points, dtype=np.float32) / scale
        elif key == 27:
            return None


def warp_frame(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Use the same target rectangle as the normal ALINA pipeline."""
    roi = ROIConfig()
    destination = np.asarray([
        roi.dst_bottom_left, roi.dst_top_left,
        roi.dst_top_right, roi.dst_bottom_right,
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (image.shape[1], image.shape[0]))


class StrokeEditor:
    def __init__(self, image: np.ndarray, labels: np.ndarray, brush_size: int = 7):
        self.image = image
        self.labels = labels
        self.brush_size = brush_size
        self.selected_class = 2
        self.drawing = False
        self.previous_point: tuple[int, int] | None = None
        self.history: list[np.ndarray] = []

        self.scale = compute_display_scale(image, max_display_dim=1100)
        self.display_size = (round(image.shape[1] * self.scale), round(image.shape[0] * self.scale))

    def mouse_callback(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        point = (min(int(x / self.scale), self.image.shape[1] - 1),
                 min(int(y / self.scale), self.image.shape[0] - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.history.append(self.labels.copy())
            self.drawing = True
            self.previous_point = point
            self.paint(point)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.paint(point)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.paint(point)
            self.drawing = False
            self.previous_point = None

    def paint(self, point: tuple[int, int]) -> None:
        if self.previous_point is not None:
            cv2.line(self.labels, self.previous_point, point, self.selected_class,
                     thickness=self.brush_size, lineType=cv2.LINE_8)
        cv2.circle(self.labels, point, self.brush_size // 2, self.selected_class, -1)
        self.previous_point = point

    def show(self, frame_name: str, position: int, total: int) -> str:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            overlay = self.image.copy()
            for class_id, color in CLASS_COLORS.items():
                overlay[self.labels == class_id] = color
            display = cv2.addWeighted(self.image, 0.65, overlay, 0.35, 0)
            display = cv2.resize(display, self.display_size)

            status = (f"{position}/{total} {frame_name} | {CLASS_NAMES[self.selected_class]} "
                      f"| brush {self.brush_size} | Y/W/B classes | U undo | +/- size | N save | Q quit")
            cv2.rectangle(display, (0, 0), (display.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(display, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("y"), ord("w"), ord("b")):
                self.selected_class = {ord("y"): 2, ord("w"): 3, ord("b"): 1}[key]
            elif key == ord("u") and self.history:
                self.labels = self.history.pop()
            elif key in (ord("+"), ord("=")):
                self.brush_size = min(75, self.brush_size + 2)
            elif key in (ord("-"), ord("_")):
                self.brush_size = max(1, self.brush_size - 2)
            elif key == ord("n"):
                return "next"
            elif key in (ord("q"), 27):
                return "quit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label taxiway marking colors with mouse strokes")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Raw_Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/color_annotations"))
    parser.add_argument("--train-per-video", type=int, default=3)
    parser.add_argument("--test-per-video", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brush-size", type=int, default=7)
    args = parser.parse_args()

    if args.train_per_video < 0 or args.test_per_video < 0 or args.brush_size < 1:
        parser.error("Frame counts must be nonnegative and brush size must be positive")

    video_folders = sorted(folder for folder in args.data_dir.iterdir()
                           if folder.is_dir() and list(folder.glob("*.jpg"))) if args.data_dir.exists() else []
    if not video_folders:
        parser.error(f"No video folders with JPG frames found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rng = np.random.default_rng(args.seed)

    for folder in video_folders:
        frames = sorted(folder.glob("*.jpg"))
        count = args.train_per_video + args.test_per_video
        if len(frames) < count:
            print(f"Skipping {folder.name}: needs {count} frames, found {len(frames)}")
            continue

        # Save the selection so restarting does not change the train/test split.
        selection_path = args.output_dir / f"{folder.name}_selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            chosen = rng.choice(len(frames), size=count, replace=False)
            selection = {"train": [frames[i].name for i in chosen[:args.train_per_video]],
                         "test": [frames[i].name for i in chosen[args.train_per_video:]]}
            selection_path.write_text(json.dumps(selection, indent=2))

        roi_path = args.output_dir / f"{folder.name}_roi.json"
        if roi_path.exists():
            corners = np.asarray(json.loads(roi_path.read_text()), dtype=np.float32)
        else:
            reference = cv2.imread(str(frames[0]))
            print(f"Select one fixed ROI for {folder.name}, then press N.")
            corners = select_roi(reference)
            if corners is None:
                break
            roi_path.write_text(json.dumps(corners.tolist(), indent=2))

        for split in ("train", "test"):
            for index, filename in enumerate(selection[split], start=1):
                source = folder / filename
                target = args.output_dir / split / folder.name / source.stem
                target.mkdir(parents=True, exist_ok=True)
                label_path = target / "labels.png"
                image_path = target / "warped.png"

                # Existing labels are left untouched when the script is restarted.
                if label_path.exists():
                    print(f"Already labeled: {split}/{folder.name}/{filename}")
                    continue

                image = cv2.imread(str(source))
                if image is None:
                    print(f"Could not read {source}")
                    continue
                warped = warp_frame(image, corners)
                labels = np.zeros(warped.shape[:2], dtype=np.uint8)
                editor = StrokeEditor(warped, labels, args.brush_size)
                action = editor.show(f"{folder.name}/{filename} [{split}]", index, len(selection[split]))

                cv2.imwrite(str(image_path), warped)
                cv2.imwrite(str(label_path), editor.labels)
                manifest[f"{split}/{folder.name}/{filename}"] = {
                    "split": split, "source": str(source),
                    "warped_image": str(image_path), "labels": str(label_path),
                    "counts": {CLASS_NAMES[class_id]: int(np.sum(editor.labels == class_id))
                               for class_id in CLASS_NAMES},
                }
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"Saved {split}/{folder.name}/{filename}")
                if action == "quit":
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print(f"Finished. Annotations saved in {args.output_dir}")


if __name__ == "__main__":
    main()

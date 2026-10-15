from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import numpy as np
METHODS = ('hsv', 'hough', 'ransac', 'kmeans', 'gemini')

# Find the yellow candidate pixels using the same HSV limits for each method.
def yellow_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_pixels = cv2.inRange(hsv_image, (12, 65, 65), (42, 255, 255))
    yellow_pixels = cv2.morphologyEx(yellow_pixels, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return yellow_pixels

# Remove tiny yellow patches so isolated noise does not set the ROI boundaries.
def candidate_points(image, mask):
    image_height, image_width = image.shape[:2]
    mask = mask.copy()

    # Ignore the upper part of the image where taxiway lines are unlikely.
    mask[:int(image_height * 0.38)] = 0
    region_count, region_labels, region_stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)
    for region_index in range(1, region_count):
        if region_stats[region_index, cv2.CC_STAT_AREA] >= max(18, image_height * image_width // 180000):
            filtered_mask[region_labels == region_index] = 255
    pixel_y, pixel_x = np.nonzero(filtered_mask)
    return np.column_stack([pixel_x, pixel_y]).astype(np.float32)

# Use the upper and lower parts of the candidate markings to place the trapezoid.
def fit_trapezoid(image, points):
    image_height, image_width = image.shape[:2]
    if len(points) < 25:
        raise ValueError('Too few marking candidates for ROI construction')

    # Use percentiles to avoid setting the ROI from a single stray pixel.
    top_y = int(np.clip(np.percentile(points[:, 1], 8), 0.4 * image_height, 0.78 * image_height))
    bottom_y = int(np.clip(np.percentile(points[:, 1], 98) + 0.025 * image_height, top_y + 0.12 * image_height, image_height - 2))
    candidate_y = points[:, 1]

    # Find the left and right marking positions near a given image height.
    def bounds(y):
        nearby_x = points[np.abs(candidate_y - y) < max(18, 0.06 * image_height), 0]
        if len(nearby_x) < 10:
            nearby_x = points[:, 0]
        return np.percentile(nearby_x, [7, 93])
    top_left_x, top_right_x = bounds(top_y)
    bottom_left_x, bottom_right_x = bounds(bottom_y)
    top_left_x -= 0.055 * image_width
    top_right_x += 0.055 * image_width
    bottom_left_x -= 0.085 * image_width
    bottom_right_x += 0.085 * image_width
    return validate_quad(np.array([[bottom_left_x, bottom_y], [top_left_x, top_y], [top_right_x, top_y], [bottom_right_x, bottom_y]]), image_width, image_height)

# Check the four corners before sending the region to ALINA.
def validate_quad(points, width, height):
    corners = np.asarray(points, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('ROI must contain four finite coordinate pairs')
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    bottom_left, top_left, top_right, bottom_right = corners
    if not (top_left[0] < top_right[0] and bottom_left[0] < bottom_right[0]):
        raise ValueError('Invalid left/right corner ordering')
    if not max(top_left[1], top_right[1]) < min(bottom_left[1], bottom_right[1]):
        raise ValueError('Top edge must be above bottom edge')
    roi_outline = corners[[0, 1, 2, 3]].astype(np.float32)
    if not cv2.isContourConvex(roi_outline.astype(np.int32)):
        raise ValueError('ROI is not convex')
    if cv2.contourArea(roi_outline) < width * height * 0.003:
        raise ValueError('ROI area is too small')
    return np.rint(corners).astype(np.float32)

# Build an ROI directly from the filtered yellow pixels.
def hsv_roi(image, seed=42):
    return fit_trapezoid(image, candidate_points(image, yellow_mask(image)))

# Find line segments among edges that also fall inside the yellow mask.
def hough_roi(image, seed=42):

    # Use grayscale edges, then limit them to yellow candidates.
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yellow_edges = cv2.Canny(grayscale_image, 60, 160)
    yellow_edges = cv2.bitwise_and(yellow_edges, yellow_mask(image))
    image_height, image_width = grayscale_image.shape
    detected_lines = cv2.HoughLinesP(yellow_edges, 1, np.pi / 180, threshold=20, minLineLength=max(18, image_width // 55), maxLineGap=35)
    if detected_lines is None:
        raise ValueError('Hough detected no candidate lines')
    line_points = []

    # Sample points along the detected lines to estimate the ROI.
    for start_x, start_y, end_x, end_y in np.asarray(detected_lines).reshape(-1, 4):
        if max(start_y, end_y) < 0.38 * image_height or abs(end_y - start_y) < 8:
            continue
        sample_positions = np.linspace(0, 1, max(5, int(np.hypot(end_x - start_x, end_y - start_y) // 6)))
        line_points.extend(zip(start_x + (end_x - start_x) * sample_positions, start_y + (end_y - start_y) * sample_positions))
    return fit_trapezoid(image, np.asarray(line_points, dtype=np.float32))

# Fit separate left and right trends while ignoring pixels that do not fit.
def ransac_roi(image, seed=42):
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    candidate_pixels = candidate_points(image, yellow_mask(image))
    if len(candidate_pixels) < 40:
        raise ValueError('Insufficient RANSAC candidate points')
    middle_x = np.median(candidate_pixels[:, 0])
    inlier_groups = []

    # Fit the left and right sides independently.
    for pixel_group in (candidate_pixels[candidate_pixels[:, 0] <= middle_x], candidate_pixels[candidate_pixels[:, 0] > middle_x]):
        if len(pixel_group) < 20:
            continue
        line_model = RANSACRegressor(estimator=LinearRegression(), random_state=seed, min_samples=max(2, int(len(pixel_group) * 0.08)), residual_threshold=max(12, image.shape[1] * 0.018))
        line_model.fit(pixel_group[:, 1:2], pixel_group[:, 0])
        matching_pixels = pixel_group[line_model.inlier_mask_]
        inlier_groups.append(matching_pixels)
    if not inlier_groups:
        raise ValueError('RANSAC found no reliable lines')
    return fit_trapezoid(image, np.concatenate(inlier_groups))

# Group similar colors, then keep the cluster with the most yellow candidates.
def kmeans_roi(image, seed=42):
    image_height, image_width = image.shape[:2]
    small_image = cv2.resize(image, (min(480, image_width), min(270, image_height)), interpolation=cv2.INTER_AREA)
    hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
    pixel_colors = hsv_image.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)

    # Use six color groups, matching the experiment settings.
    _, cluster_labels, cluster_centers = cv2.kmeans(pixel_colors, 6, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.3), 3, cv2.KMEANS_PP_CENTERS)
    small_yellow_mask = cv2.resize(yellow_mask(image), (small_image.shape[1], small_image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    cluster_map = cluster_labels.reshape(hsv_image.shape[:2])
    yellow_counts = []
    for cluster_index in range(len(cluster_centers)):
        in_cluster = cluster_map == cluster_index
        matching_yellow_pixels = int(np.count_nonzero(in_cluster & small_yellow_mask))
        yellow_counts.append(matching_yellow_pixels)
    best_cluster = int(np.argmax(yellow_counts))
    if yellow_counts[best_cluster] < 10:
        raise ValueError('K-means clusters contain too few yellow candidate pixels')
    selected_mask = ((cluster_map == best_cluster) & small_yellow_mask).astype(np.uint8) * 255
    selected_mask = cv2.resize(selected_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return fit_trapezoid(image, candidate_points(image, selected_mask))

# Ask Gemini for a normalized box, then reshape it into the trapezoid ALINA needs.
def gemini_roi(image, cache_path, model='gemini-3-flash-preview'):
    """Gemini predicts a normalized [ymin, xmin, ymax, xmax] box.

    Convert the box to a trapezoid for ALINA. Raw predictions are cached
    separately from previous Gemini coordinate experiments.
    """
    cache_path = Path(cache_path)
    cache_path = cache_path.with_name(cache_path.stem + '_bbox1000_v1' + cache_path.suffix)
    image_height, image_width = image.shape[:2]
    if cache_path.exists():
        gemini_data = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        if not os.getenv('GEMINI_API_KEY'):
            raise RuntimeError('Set GEMINI_API_KEY or provide a cached prediction')
        from google import genai
        from google.genai import types
        encoded_success, encoded_image = cv2.imencode('.jpg', image)
        if not encoded_success:
            raise ValueError('Cannot encode image')
        prompt = '\nYou are locating yellow painted taxiway line markings in an aircraft cockpit image.\nIdentify the COMBINED visible region occupied by the yellow painted taxiway\nmarkings, including left, right, and central branches where visible.\n\nThe other ROI-selection methods use these OpenCV HSV yellow candidate ranges:\nHue 12-42, Saturation 65-255, Value 65-255. These are visual guidance only;\ndo not claim to have numerically thresholded the image.\n\nReturn ONE rectangular bounding box tightly enclosing the visible yellow\npainted taxiway markings. Include a small margin of surrounding pavement,\nbut exclude unrelated objects, aircraft nose, propeller, grass, and sky as\nmuch as possible. Do not prioritize the bottom of the image; examine the\nentire frame. Only include visible markings, not extrapolated extensions.\n\nIMPORTANT: Use normalized coordinates from 0 to 1000 on EACH axis,\nindependently of image resolution. Origin (0,0) is the upper left;\nX increases rightward, Y increases downward. The bottom right is (1000,1000).\n\nReturn the box in EXACTLY this order: [ymin, xmin, ymax, xmax].\nymin is the TOP edge, ymax the BOTTOM edge; xmin is the LEFT edge,\nxmax the RIGHT edge. Require 0 <= ymin < ymax <= 1000 and\n0 <= xmin < xmax <= 1000.\n\nReturn ONLY JSON, without explanation:\n{"box": [ymin, xmin, ymax, xmax]}\n'
        gemini_client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        gemini_response = gemini_client.models.generate_content(model=model, contents=[types.Part.from_bytes(data=encoded_image.tobytes(), mime_type='image/jpeg'), prompt], config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0))
        gemini_data = json.loads(gemini_response.text)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(gemini_data, indent=2), encoding='utf-8')
    normalized_box = np.asarray(gemini_data['box'], dtype=np.float64)
    if normalized_box.shape != (4,) or not np.isfinite(normalized_box).all():
        raise ValueError('Gemini must return four finite values [ymin,xmin,ymax,xmax]')
    top_normalized, left_normalized, bottom_normalized, right_normalized = normalized_box
    if not (0 <= top_normalized < bottom_normalized <= 1000 and 0 <= left_normalized < right_normalized <= 1000):
        raise ValueError(f'Invalid normalized Gemini box: {normalized_box.tolist()}')
    left_x = left_normalized * (image_width - 1) / 1000.0
    right_x = right_normalized * (image_width - 1) / 1000.0
    top_y = top_normalized * (image_height - 1) / 1000.0
    bottom_y = bottom_normalized * (image_height - 1) / 1000.0
    middle_x = (left_x + right_x) / 2.0
    half_top_width = (right_x - left_x) * 0.25
    roi_corners = np.array([[left_x, bottom_y], [middle_x - half_top_width, top_y], [middle_x + half_top_width, top_y], [right_x, bottom_y]], dtype=np.float32)
    return validate_quad(roi_corners, image_width, image_height)

# Use the requested method to generate the four ROI corners.
def select_roi(image, method, seed=42, cache_path=None, model='gemini-3-flash-preview'):
    if method == 'gemini':
        if cache_path is None:
            raise ValueError('Gemini requires a cache path')
        return gemini_roi(image, cache_path, model)
    roi_functions = {'hsv': hsv_roi, 'hough': hough_roi, 'ransac': ransac_roi, 'kmeans': kmeans_roi}
    return roi_functions[method](image, seed)

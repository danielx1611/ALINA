from __future__ import annotations
import json
import os
from pathlib import Path
import cv2
import numpy as np

METHODS = ('hsv', 'hough', 'ransac', 'kmeans', 'gemini')


def yellow_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (12, 65, 65), (42, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def candidate_points(image, mask):
    h, w = image.shape[:2]
    mask = mask.copy()
    mask[:int(h * .38)] = 0
    # Ignore the aircraft nose/propeller in the center lower region.
    # This is a heuristic, not a learned mask.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(18, h*w//180000):
            cleaned[labels == i] = 255
    y, x = np.nonzero(cleaned)
    return np.column_stack([x, y]).astype(np.float32)


def fit_trapezoid(image, points):
    h, w = image.shape[:2]
    if len(points) < 25:
        raise ValueError('Too few marking candidates for ROI construction')
    # Focus on the visible vertical range rather than the whole image.
    top = int(np.clip(np.percentile(points[:, 1], 8), .40*h, .78*h))
    bottom = int(np.clip(np.percentile(points[:, 1], 98) + .025*h, top + .12*h, h-2))
    ys = points[:, 1]
    def bounds(y):
        near = points[np.abs(ys-y) < max(18, .06*h), 0]
        if len(near) < 10:
            near = points[:, 0]
        return np.percentile(near, [7, 93])
    lt, rt = bounds(top)
    lb, rb = bounds(bottom)
    # Provide margin for a useful perspective warp.
    lt -= .055*w; rt += .055*w
    lb -= .085*w; rb += .085*w
    return validate_quad(np.array([[lb,bottom],[lt,top],[rt,top],[rb,bottom]]), w, h)


def validate_quad(points, width, height):
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all():
        raise ValueError('ROI must contain four finite coordinate pairs')
    p[:, 0] = np.clip(p[:, 0], 0, width-1)
    p[:, 1] = np.clip(p[:, 1], 0, height-1)
    bl, tl, tr, br = p
    if not (tl[0] < tr[0] and bl[0] < br[0]):
        raise ValueError('Invalid left/right corner ordering')
    if not (max(tl[1], tr[1]) < min(bl[1], br[1])):
        raise ValueError('Top edge must be above bottom edge')
    contour = p[[0,1,2,3]].astype(np.float32)
    if not cv2.isContourConvex(contour.astype(np.int32)):
        raise ValueError('ROI is not convex')
    if cv2.contourArea(contour) <= 0:
        raise ValueError('ROI has zero area')
    return np.rint(p).astype(np.float32)


def hsv_roi(image, seed=42):
    return fit_trapezoid(image, candidate_points(image, yellow_mask(image)))


def hough_roi(image, seed=42):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.bitwise_and(edges, yellow_mask(image))
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                            minLineLength=max(18, w//55), maxLineGap=35)
    if lines is None:
        raise ValueError('Hough detected no candidate lines')
    samples = []
    for x1,y1,x2,y2 in np.asarray(lines).reshape(-1, 4):
        if max(y1,y2) < .38*h or abs(y2-y1) < 8:
            continue
        t = np.linspace(0,1, max(5,int(np.hypot(x2-x1,y2-y1)//6)))
        samples.extend(zip(x1+(x2-x1)*t, y1+(y2-y1)*t))
    return fit_trapezoid(image, np.asarray(samples, dtype=np.float32))


def ransac_roi(image, seed=42):
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    pts = candidate_points(image, yellow_mask(image))
    if len(pts) < 40:
        raise ValueError('Insufficient RANSAC candidate points')
    # Split into left and right groups to fit two independent boundaries.
    median = np.median(pts[:,0])
    sampled = []
    for group in (pts[pts[:,0] <= median], pts[pts[:,0] > median]):
        if len(group) < 20:
            continue
        model = RANSACRegressor(estimator=LinearRegression(), random_state=seed,
                                min_samples=max(2,int(len(group)*.08)),
                                residual_threshold=max(12, image.shape[1]*.018))
        model.fit(group[:,1:2], group[:,0])
        inliers = group[model.inlier_mask_]
        sampled.append(inliers)
    if not sampled:
        raise ValueError('RANSAC found no reliable lines')
    return fit_trapezoid(image, np.concatenate(sampled))


def kmeans_roi(image, seed=42):
    h,w = image.shape[:2]
    small = cv2.resize(image, (min(480,w), min(270,h)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1,3).astype(np.float32)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(pixels, 6, None,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 35, .3),
        3, cv2.KMEANS_PP_CENTERS)
    # K-means centers can average yellow markings with gray pavement.
    # Select clusters by their yellow-pixel enrichment rather than requiring
    # a cluster centroid to be yellow. This keeps K-means meaningfully involved.
    yellow_small = cv2.resize(yellow_mask(image),
                              (small.shape[1], small.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    label_map = labels.reshape(hsv.shape[:2])
    cluster_scores = []
    for k in range(len(centers)):
        membership = label_map == k
        yellow_count = int(np.count_nonzero(membership & yellow_small))
        cluster_scores.append(yellow_count)
    best = int(np.argmax(cluster_scores))
    if cluster_scores[best] < 10:
        raise ValueError('K-means clusters contain too few yellow candidate pixels')
    # Retain only yellow-like pixels within the most enriched cluster.
    mask = ((label_map == best) & yellow_small).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return fit_trapezoid(image, candidate_points(image, mask))


def gemini_roi(image, cache_path, model='gemini-3-flash-preview'):
    """Gemini predicts a normalized [ymin, xmin, ymax, xmax] box.

    Convert the box to a trapezoid for ALINA. Raw predictions are cached
    separately from previous Gemini coordinate experiments.
    """
    cache_path = Path(cache_path)
    cache_path = cache_path.with_name(cache_path.stem + '_bbox1000_v1' + cache_path.suffix)
    h, w = image.shape[:2]

    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        if not os.getenv('GEMINI_API_KEY'):
            raise RuntimeError('Set GEMINI_API_KEY or provide a cached prediction')
        from google import genai
        from google.genai import types
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise ValueError('Cannot encode image')
        prompt = """
You are locating yellow painted taxiway line markings in an aircraft cockpit image.
Identify the COMBINED visible region occupied by the yellow painted taxiway
markings, including left, right, and central branches where visible.

The other ROI-selection methods use these OpenCV HSV yellow candidate ranges:
Hue 12-42, Saturation 65-255, Value 65-255. These are visual guidance only;
do not claim to have numerically thresholded the image.

Return ONE rectangular bounding box tightly enclosing the visible yellow
painted taxiway markings. Include a small margin of surrounding pavement,
but exclude unrelated objects, aircraft nose, propeller, grass, and sky as
much as possible. Do not prioritize the bottom of the image; examine the
entire frame. Only include visible markings, not extrapolated extensions.

IMPORTANT: Use normalized coordinates from 0 to 1000 on EACH axis,
independently of image resolution. Origin (0,0) is the upper left;
X increases rightward, Y increases downward. The bottom right is (1000,1000).

Return the box in EXACTLY this order: [ymin, xmin, ymax, xmax].
ymin is the TOP edge, ymax the BOTTOM edge; xmin is the LEFT edge,
xmax the RIGHT edge. Require 0 <= ymin < ymax <= 1000 and
0 <= xmin < xmax <= 1000.

Return ONLY JSON, without explanation:
{"box": [ymin, xmin, ymax, xmax]}
"""
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=encoded.tobytes(), mime_type='image/jpeg'), prompt],
            config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0),
        )
        data = json.loads(response.text)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    box = np.asarray(data['box'], dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('Gemini must return four finite values [ymin,xmin,ymax,xmax]')
    ymin, xmin, ymax, xmax = box
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        raise ValueError(f'Invalid normalized Gemini box: {box.tolist()}')

    # Convert the rectangular box to a perspective trapezoid.
    # Fixed 50% top-to-bottom width ratio, centered on the box.
    # The geometric conversion is identical for every image and model response.
    x_left = xmin * (w - 1) / 1000.0
    x_right = xmax * (w - 1) / 1000.0
    y_top = ymin * (h - 1) / 1000.0
    y_bottom = ymax * (h - 1) / 1000.0
    center_x = (x_left + x_right) / 2.0
    top_half_width = (x_right - x_left) * 0.25
    quad = np.array([
        [x_left, y_bottom],
        [center_x - top_half_width, y_top],
        [center_x + top_half_width, y_top],
        [x_right, y_bottom],
    ], dtype=np.float32)
    return validate_quad(quad, w, h)


def select_roi(image, method, seed=42, cache_path=None, model='gemini-3-flash-preview'):
    if method == 'gemini':
        if cache_path is None:
            raise ValueError('Gemini requires a cache path')
        return gemini_roi(image, cache_path, model)
    functions = {'hsv':hsv_roi, 'hough':hough_roi, 'ransac':ransac_roi, 'kmeans':kmeans_roi}
    return functions[method](image, seed)

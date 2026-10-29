"""Train color detectors and convert their predictions into ALINA masks."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

METHODS = ("baseline", "kmeans", "gmm", "random_forest", "svm", "adaptive_kmeans")


def color_features(image):
    """Use the same normalized HSV image as ALINA, plus original BGR colors."""
    from alina.color import normalize_color_features
    hsv = normalize_color_features(image)
    return np.concatenate((hsv, image), axis=2).astype(np.float32)


def sample_training_pixels(annotation_dir, seed=42, per_class=6000):
    """Read only training annotations; test labels never enter model fitting."""
    from pathlib import Path

    random_generator = np.random.default_rng(seed)
    samples = {1: [], 2: [], 3: []}
    training_folder = Path(annotation_dir) / "train"

    for label_path in sorted(training_folder.glob("*/*/labels.png")):
        image = cv2.imread(str(label_path.parent / "warped.png"))
        labels = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or labels is None or image.shape[:2] != labels.shape:
            raise ValueError(f"Invalid annotation pair: {label_path}")
        features = color_features(image)

        for class_id in samples:
            pixels = features[labels == class_id]
            if len(pixels):
                # Limit each frame so one heavily annotated frame cannot dominate.
                count = min(len(pixels), max(100, per_class // 3))
                chosen = random_generator.choice(len(pixels), size=count, replace=False)
                samples[class_id].append(pixels[chosen])

    training_features, training_labels = [], []
    for class_id, groups in samples.items():
        if not groups:
            raise ValueError(f"No training pixels for class {class_id}. Check your strokes.")
        pixels = np.concatenate(groups)
        chosen = random_generator.choice(len(pixels), size=min(per_class, len(pixels)), replace=False)
        training_features.append(pixels[chosen])
        training_labels.append(np.full(len(chosen), class_id, dtype=np.uint8))

    return np.concatenate(training_features), np.concatenate(training_labels)


def cluster_labels(model, features, labels):
    """Give each learned cluster its most common annotated class."""
    assignments = model.predict(features)
    mapping = {}
    for cluster_id in range(model.n_clusters if hasattr(model, "n_clusters") else model.n_components):
        matching = labels[assignments == cluster_id]
        # Use the most common annotated class for this cluster.
        if len(matching) > 0:
            class_counts = np.bincount(matching, minlength=4)
            mapping[cluster_id] = int(np.argmax(class_counts))
        else:
            mapping[cluster_id] = 1
    return mapping


def fit_models(features, labels, seed=42):
    """Train the four reusable models and save class color prototypes."""
    hsv_features = features[:, :3]

    # Fit color groups before training the supervised classifiers.
    models = {}

    kmeans = KMeans(n_clusters=8, random_state=seed, n_init=10)
    kmeans.fit(hsv_features)
    models["kmeans"] = (kmeans, cluster_labels(kmeans, hsv_features, labels))

    gmm = GaussianMixture(n_components=8, covariance_type="diag", random_state=seed, max_iter=100)
    gmm.fit(hsv_features)
    models["gmm"] = (gmm, cluster_labels(gmm, hsv_features, labels))

    # The forest learns from all six color features.
    forest = RandomForestClassifier(n_estimators=100, max_depth=18, random_state=seed,
                                    n_jobs=-1, class_weight="balanced_subsample")
    forest.fit(features, labels)
    models["random_forest"] = forest

    # Scale the features so no color channel has an unfair advantage.
    # A linear SVM predicts much faster than the previous RBF SVM.
    svm_model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=5000, random_state=seed)
    )
    svm_model.fit(features, labels)
    models["svm"] = svm_model

    # These colors help match new clusters to the labeled classes.
    models["prototypes"] = np.stack([
        np.median(hsv_features[labels == class_id], axis=0)
        for class_id in (1, 2, 3)
    ])
    return models


def predict_mask(image, method, models=None, seed=42, chunk_size=100000):
    """Return ALINA's usual 0/255 binary mask for yellow and white markings."""
    if method not in METHODS:
        raise ValueError(f"Unknown color method: {method}")

    if method == "baseline":
        from alina.color import normalize_color_features
        normalized = normalize_color_features(image)
        return cv2.inRange(normalized, np.array((0, 70, 170), dtype=np.uint8),
                           np.array((255, 255, 255), dtype=np.uint8))

    if models is None:
        raise ValueError("Trained models are required for this method")

    height, width = image.shape[:2]
    features = color_features(image).reshape(-1, 6)

    if method == "adaptive_kmeans":
        # Learn new color groups for each frame, then match their centers to
        # the representative background, yellow, and white training colors.
        random_generator = np.random.default_rng(seed)
        indices = random_generator.choice(len(features), min(12000, len(features)), replace=False)
        local = KMeans(n_clusters=8, random_state=seed, n_init=3, max_iter=75)
        local.fit(features[indices, :3])
        prototypes = models["prototypes"]
        distances = np.linalg.norm(local.cluster_centers_[:, None, :] - prototypes[None, :, :], axis=2)
        cluster_classes = distances.argmin(axis=1) + 1
        model = local
    elif method in ("kmeans", "gmm"):
        model, mapping = models[method]
        number_of_clusters = model.n_clusters if hasattr(model, "n_clusters") else model.n_components
        cluster_classes = np.array([
            mapping[cluster_index] for cluster_index in range(number_of_clusters)
        ])
    else:
        model = models[method]

    # Predict in batches so large frames do not need one huge model call.
    predictions = np.empty(len(features), dtype=np.uint8)
    for start in range(0, len(features), chunk_size):
        batch = features[start:start + chunk_size]
        if method in ("kmeans", "gmm", "adaptive_kmeans"):
            predictions[start:start + len(batch)] = cluster_classes[model.predict(batch[:, :3])]
        else:
            predictions[start:start + len(batch)] = model.predict(batch)

    return np.where(predictions.reshape(height, width) >= 2, 255, 0).astype(np.uint8)

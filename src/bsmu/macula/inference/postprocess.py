from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize

# CatBoost predicted label -> palette class index
_CATBOOST_LABEL_TO_CLASS = {1: 1, 2: 2, 3: 10, 4: 11, 5: 0}


def refine_tr_mask(tr_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    excluded_seed = (3, 4, 5, 6, 12, 13, 14, 15, 16, 17, 18, 255)
    result = tr_mask.copy()
    for cls in excluded_seed:
        result[gt_mask == cls] = 0
    result[gt_mask >= 15] = 0
    return result


def _filtered_rpe_mask(
    gt_mask: np.ndarray, rpe_class: int, rpe_area_threshold: int,
) -> np.ndarray:
    rpe_mask = gt_mask == rpe_class
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num = label(rpe_mask, structure=structure)
    filtered_rpe = np.zeros_like(rpe_mask, dtype=bool)
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        if comp.sum() >= rpe_area_threshold:
            filtered_rpe |= comp
    return filtered_rpe


def remove_above_rpe(
    tr_mask: np.ndarray, gt_mask: np.ndarray, rpe_class: int = 9, rpe_area_threshold: int = 50,
) -> np.ndarray:
    result = tr_mask.copy()
    H, W = result.shape
    filtered_rpe = _filtered_rpe_mask(gt_mask, rpe_class, rpe_area_threshold)
    result[filtered_rpe] = 0

    top_rpe = np.full(W, -1, dtype=np.int32)
    for col in range(W):
        rows = np.where(filtered_rpe[:, col])[0]
        if rows.size:
            top_rpe[col] = rows[0]

    yy = np.arange(H)[:, None]
    valid_cols = top_rpe >= 0
    result[(yy < top_rpe) & valid_cols] = 0
    return result


def remove_below_choria(
    tr_mask: np.ndarray, gt_mask: np.ndarray, rpe_class: int = 14, rpe_area_threshold: int = 50,
) -> np.ndarray:
    result = tr_mask.copy()
    H, W = result.shape
    filtered_rpe = _filtered_rpe_mask(gt_mask, rpe_class, rpe_area_threshold)
    result[filtered_rpe] = 0

    top_rpe = np.full(W, -1, dtype=np.int32)
    for col in range(W):
        rows = np.where(filtered_rpe[:, col])[0]
        if rows.size:
            top_rpe[col] = rows[0]

    yy = np.arange(H)[:, None]
    valid_cols = top_rpe >= 0
    result[(yy > top_rpe) & valid_cols] = 0
    return result


def filter_by(mask: np.ndarray, am: int, ah: int) -> np.ndarray:
    m = mask.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (m > 0).astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < am or h < ah:
            m[labels == i] = 0
    return m


def refine_tr_pipeline(tr_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    tr_mask = refine_tr_mask(tr_mask, gt_mask)
    tr_mask = remove_above_rpe(tr_mask, gt_mask)
    tr_mask = remove_below_choria(tr_mask, gt_mask)
    tr_mask = filter_by(tr_mask, 20, 5)
    return tr_mask


def normalize_oct_by_anatomy(
    image: np.ndarray,
    mask: np.ndarray,
    background_class: int = 255,
    rpe_class: int = 9,
    min_pixels: int = 100,
    fallback_low_pct: float = 2.0,
    fallback_high_pct: float = 98.0,
    target_max: float = 255.0,
) -> np.ndarray:
    """Normalize OCT intensity using anatomical references with robust fallbacks.

    Background is restricted to pixels strictly above the retina.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    retina_mask = mask != background_class
    valid_cols = np.any(retina_mask, axis=0)

    top_line = np.full(w, h, dtype=np.int32)
    if np.any(valid_cols):
        top_line[valid_cols] = np.argmax(retina_mask[:, valid_cols], axis=0)

    rows = np.arange(h)[:, None]
    above_retina = rows < top_line
    above_retina[:, ~valid_cols] = False
    bg_mask = above_retina & (mask == background_class)
    bg_pixels = img[bg_mask]

    rpe_pixels = img[mask == rpe_class]

    bg_valid = bg_pixels.size >= min_pixels
    rpe_valid = rpe_pixels.size >= min_pixels

    if not bg_valid:
        logging.info(f'bg_pixels.size {bg_pixels.size} is smaller than min_pixels {min_pixels}')
    if not rpe_valid:
        logging.info(f'rpe_pixels.size {rpe_pixels.size} is smaller than min_pixels {min_pixels}')

    bg_ref = np.mean(bg_pixels) if bg_valid else np.percentile(img, fallback_low_pct)
    rpe_ref = np.mean(rpe_pixels) if rpe_valid else np.percentile(img, fallback_high_pct)

    denominator = rpe_ref - bg_ref

    if denominator <= 1e-6:
        p_low = np.percentile(img, fallback_low_pct)
        p_high = np.percentile(img, fallback_high_pct)
        denominator = max(p_high - p_low, 1e-6)
        bg_ref = p_low

    normalized = (img - bg_ref) / denominator * target_max
    return np.clip(normalized, 0.0, target_max).astype(np.float32)


def component_statistics(
    component: np.ndarray,
    gt: np.ndarray,
    intensity_image: np.ndarray,
    stats: np.ndarray,
    lbl: int,
    centroids: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Compute the feature vector for a single connected component.

    The returned feature order must match the order used to train the
    CatBoost model.
    """
    H, W = component.shape
    o36 = np.isin(gt, [3, 6]).sum()
    o45 = np.isin(gt, [4, 5]).sum()

    area = stats[lbl, cv2.CC_STAT_AREA]
    width = stats[lbl, cv2.CC_STAT_WIDTH]
    height = stats[lbl, cv2.CC_STAT_HEIGHT]

    aspect_ratio = width / height if height > 0 else np.nan

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else np.nan
    else:
        circularity = np.nan
        solidity = np.nan

    skeleton = skeletonize(component.astype(bool))
    skeleton_length = skeleton.sum()
    skeleton_thickness = area / skeleton_length if skeleton_length > 0 else np.nan

    cx, cy = centroids[lbl]

    mean_intensity = cv2.mean(intensity_image, mask=component)[0]

    features = np.array([
        area,
        circularity,
        o36,
        o45,
        height,
        n_components - 1,
        aspect_ratio,
        solidity,
        skeleton_length,
        skeleton_thickness,
        cx / W,
        cy / H,
        mean_intensity,
    ], dtype=np.float32)

    return features.reshape(1, -1)


def generate_prediction_mask(
    classifier,
    image: np.ndarray,
    tr: np.ndarray,
    p_gt: np.ndarray,
) -> np.ndarray:
    H, W = image.shape
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    if tr.sum() == 0:
        return pred_mask

    mask = (tr > 0).astype(np.uint8)
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for lbl in range(1, n_components):
        component = (labels == lbl).astype(np.uint8)
        features = component_statistics(component, p_gt, image, stats, lbl, centroids, n_components)
        target = classifier.predict(features)
        pred_mask += component * _CATBOOST_LABEL_TO_CLASS[target.item()]

    return pred_mask


def postprocess_mask(classifier, image: np.ndarray, tr: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    tr = refine_tr_pipeline(tr, p_gt)
    tr = tr | (p_gt == 1)
    image = normalize_oct_by_anatomy(image, p_gt)
    return generate_prediction_mask(classifier, image, tr, p_gt)

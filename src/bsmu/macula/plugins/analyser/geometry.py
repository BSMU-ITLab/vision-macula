"""Геометрия масок: границы, сплайны, сглаживание.

Модуль отвечает только за низкоуровневые операции над масками и кривыми;
измерения патологий живут в :mod:`bsmu.macula.plugins.analyser.measurements`.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from bsmu.macula.plugins.analyser.schema import CLASS_CHOROID

#: Ядро морфологических операций при сглаживании маски хориоидеи.
SMOOTH_MORPH_KERNEL = np.ones((5, 5), np.uint8)
#: Размер ядра гауссова размытия.
SMOOTH_BLUR_KERNEL = (7, 7)
#: Порог бинаризации после размытия.
SMOOTH_BINARY_THRESHOLD = 127


def smooth_mask(mask: np.ndarray, class_id: int = CLASS_CHOROID) -> np.ndarray:
    """Бинарная маска класса после open/close и гауссова сглаживания."""
    mask_bin = ((mask == class_id).astype(np.uint8)) * 255
    opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, SMOOTH_MORPH_KERNEL)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, SMOOTH_MORPH_KERNEL)
    blurred = cv2.GaussianBlur(closed, SMOOTH_BLUR_KERNEL, 0)
    return (blurred > SMOOTH_BINARY_THRESHOLD).astype(np.uint8)


def extract_upper_boundary(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Верхняя граница хориоидеи по X-координатам.

    Возвращает ``(xs, upper)`` либо ``(None, None)``, если границу найти не удалось.
    """
    mask_smooth = smooth_mask(mask)
    contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    _, width = mask.shape
    xs = np.arange(width)
    upper = np.full(width, np.nan)

    for point in contour:
        x, y = point[0]
        if np.isnan(upper[x]) or y < upper[x]:
            upper[x] = y

    known = ~np.isnan(upper)
    upper_interp = np.interp(xs, xs[known], upper[known])
    return xs, upper_interp


def smooth_boundary(xs: np.ndarray, ys: np.ndarray,
                    smooth_factor: float = 50) -> tuple[np.ndarray, UnivariateSpline]:
    """Сглаживает границу сплайном и возвращает значения вместе со сплайном."""
    spline = UnivariateSpline(xs, ys, s=smooth_factor)
    return spline(xs), spline


def get_foveola_center(fovea_mask: np.ndarray) -> tuple[int, int] | None:
    """Центр фовеолы (зона 1 маски фовеа) как центр масс."""
    ys, xs = np.where(fovea_mask == 1)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def downward_normal(spline: UnivariateSpline, x: float) -> tuple[float, float]:
    """Единичная нормаль к границе в точке ``x``, направленная вниз."""
    slope = float(spline.derivative()(x))
    nx = -slope
    ny = 1.0
    length = np.sqrt(nx * nx + ny * ny)
    return nx / length, ny / length


def upward_normal(spline: UnivariateSpline, x: float) -> tuple[float, float]:
    """Единичная нормаль к границе в точке ``x``, направленная вверх."""
    slope = float(spline.derivative()(x))
    nx = slope
    ny = -1.0
    length = np.sqrt(nx * nx + ny * ny)
    return nx / length, ny / length



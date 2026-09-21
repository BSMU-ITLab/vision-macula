"""Измерение друз (каждый контур — отдельная друза)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from bsmu.macula.plugins.analyser.measurements.kernel import (
    DEFAULT_PERP_LENGTH,
    height_um,
    lateral_extent,
    max_perpendicular,
    width_along_boundary,
)
from bsmu.macula.plugins.analyser.schema import to_um2

#: Порядок элементов результата измерения друзы.
DrusenResult = tuple[
    float | None, float | None, float | None,
    int | None, int | None, tuple | None, str | None, list | None,
]

EMPTY_RESULT: DrusenResult = (None, None, None, None, None, None, None, None)


def measure_drusen(
    mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    contour: np.ndarray,
    fovea_mask: np.ndarray | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    perp_len: int = DEFAULT_PERP_LENGTH,
) -> DrusenResult:
    """Измеряет одну друзу относительно сглаженной линии хориоидеи.

    Локализация друзы определяется отдельно
    (:meth:`DataConverter.get_drusen_location`) и здесь не возвращается.
    """
    del fovea_mask

    area_px_value = float(cv2.contourArea(contour))
    if area_px_value == 0:
        return EMPTY_RESULT

    height, width = mask.shape
    target_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(target_mask, [contour], -1, 1, -1)

    extent = lateral_extent(target_mask)
    if extent is None:
        return EMPTY_RESULT
    left_x, right_x = extent

    width_um, choroid_segment = width_along_boundary(
        smooth_upper, target_mask, scale_x, scale_y
    )
    max_height_px, max_perp_point = max_perpendicular(
        target_mask, smooth_upper, spline, left_x, right_x, perp_len
    )

    return (
        width_um,
        height_um(max_height_px, max_perp_point, scale_x, scale_y),
        to_um2(area_px_value, scale_x, scale_y),
        left_x,
        right_x,
        max_perp_point,
        None,
        choroid_segment,
    )

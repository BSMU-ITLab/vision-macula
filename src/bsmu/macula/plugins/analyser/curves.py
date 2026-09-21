"""Измерения вдоль сглаженной линии хориоидеи.

Здесь живут «путевые» измерения: центральная толщина сетчатки (ЦТС) в
фовеоле и рядом с фовеолой/фовеей, перпендикуляр через фовеолу и толщина
хориоидеи в центре.

Алгоритм ЦТС соответствует документу «Классы структур на ОКТ»: находим длину
пути до хориоидеи и вычитаем из неё области отслоек, друз, субретинального и
интраретинального гиперрефлективного материала, СРЖ и РПЭ.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from bsmu.macula.plugins.analyser.geometry import downward_normal
from bsmu.macula.plugins.analyser.schema import (
    CLASS_CHOROID,
    CLASS_INTRARETINAL_HYPERREFLECTIVE,
    CLASS_IRF,
    CLASS_NEUROEPITHELIAL_DETACHMENT,
    CLASS_RPE,
    CLASS_SUBRETINAL_HYPERREFLECTIVE,
)

#: Классы, площадь которых вычитается из длины пути до хориоидеи:
#: друзы (1), отслойки (2, 10, 11, 16), СРГМ (4), ИРГМ (5), СРЖ (6), РПЭ (9).
SUBTRACTED_CLASSES = frozenset(
    {
        1,
        2,
        CLASS_IRF,
        CLASS_SUBRETINAL_HYPERREFLECTIVE,
        CLASS_INTRARETINAL_HYPERREFLECTIVE,
        CLASS_NEUROEPITHELIAL_DETACHMENT,
        CLASS_RPE,
        10,
        11,
        16,
    }
)

#: Длина перпендикуляра по умолчанию (пиксели).
DEFAULT_PERP_LENGTH = 200
#: Длина перпендикуляра при измерении толщины хориоидеи.
CHOROID_PERP_LENGTH = 400


def central_width_of_retina(
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    foveola_center: tuple[int, int],
    mask: np.ndarray,
    perp_len: int = DEFAULT_PERP_LENGTH,
) -> tuple[int, int, float, tuple[int, int], int] | None:
    """Толщина сетчатки от точки ``foveola_center`` вниз до хориоидеи.

    :return: ``(cx, cy, dist_corrected_px, choroid_point, count_subtract_px)``
        либо ``None``, если хориоидея не достигнута.
    """
    cx, cy = foveola_center
    height, width = mask.shape

    if cx < 0 or cx >= len(smooth_upper):
        return None

    nx, ny = downward_normal(spline, cx)
    choroid_point = None
    count_subtract = 0

    for t in range(0, perp_len):
        xx = int(cx + nx * t)
        yy = int(cy + ny * t)
        if xx < 0 or xx >= width or yy < 0 or yy >= height:
            break

        cls = mask[yy, xx]
        if cls == CLASS_CHOROID:
            choroid_point = (xx, yy)
            break
        if cls in SUBTRACTED_CLASSES:
            count_subtract += 1

    if choroid_point is None:
        return None

    dist_full_px = float(np.hypot(choroid_point[0] - cx, choroid_point[1] - cy))
    dist_corrected_px = max(0.0, dist_full_px - count_subtract)
    return cx, cy, dist_corrected_px, choroid_point, count_subtract


def central_width_near(
    img: np.ndarray,
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    fovea_mask: np.ndarray,
    mask: np.ndarray,
    ind: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Крайние точки зоны фовеа/фовеолы на границе с фоном.

    :param ind: 1 — фовеола, 2 — фовеа.
    """
    del img, smooth_upper, spline  # параметры сохранены ради совместимости вызовов

    zone_mask = (fovea_mask == ind).astype(np.uint8)
    background_mask = (mask == 0).astype(np.uint8)

    if zone_mask.sum() == 0 or background_mask.sum() == 0:
        return None

    kernel = np.ones((3, 3), np.uint8)
    boundary = (cv2.dilate(zone_mask, kernel) & cv2.dilate(background_mask, kernel)).astype(bool)
    ys, xs = np.where(boundary)

    if len(xs) == 0:
        return None

    left = int(np.argmin(xs))
    right = int(np.argmax(xs))
    return (int(xs[left]), int(ys[left])), (int(xs[right]), int(ys[right]))


def perpendicular_through_foveola(
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    foveola_center: tuple[int, int],
    mask: np.ndarray,
    perp_len: int = DEFAULT_PERP_LENGTH,
) -> tuple[int, int, int, tuple[int, int]] | None:
    """Перпендикуляр от верхней границы хориоидеи до её нижней границы."""
    cx, cy = foveola_center
    height, width = mask.shape

    del cy  # перпендикуляр строится от верхней границы хориоидеи

    if cx < 0 or cx >= len(smooth_upper):
        return None

    y0 = float(smooth_upper[cx])
    nx, ny = downward_normal(spline, cx)
    inside = False
    best_point = None

    for t in range(0, perp_len):
        xx = int(cx + nx * t)
        yy = int(y0 + ny * t)
        if xx < 0 or xx >= width or yy < 0 or yy >= height:
            break

        if mask[yy, xx] == CLASS_CHOROID:
            inside = True
            best_point = (xx, yy)
        elif inside:
            break

    if best_point is None:
        return None

    distance = int(np.hypot(best_point[0] - cx, best_point[1] - y0))
    return cx, int(y0), distance, best_point


def measure_choroid_thickness_at_center(
    mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    center_x: int,
    scale_x: float | None = None,
    scale_y: float | None = None,
    foveola_center: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int], float, float] | None:
    """Толщина хориоидеи в центре: часть перпендикуляра ЦТС внутри хориоидеи.

    :return: ``(верхняя_точка, нижняя_точка, dx_мкм, dy_мкм)`` или ``None``.
    """
    if center_x < 0 or center_x >= len(smooth_upper) or foveola_center is None:
        return None

    cx, cy = foveola_center
    height, width = mask.shape
    nx, ny = downward_normal(spline, center_x)

    t_start = None
    t_end = None
    for t in range(0, CHOROID_PERP_LENGTH):
        xx = int(cx + nx * t)
        yy = int(cy + ny * t)
        if xx < 0 or xx >= width or yy < 0 or yy >= height:
            break
        if mask[yy, xx] == CLASS_CHOROID and t_start is None:
            t_start = t
        if t_start is not None and mask[yy, xx] != CLASS_CHOROID:
            t_end = t - 1
            break

    if t_start is not None and t_end is None:
        t_end = CHOROID_PERP_LENGTH - 1
    if t_start is None:
        return None

    upper_point = (int(cx + nx * t_start), int(cy + ny * t_start))
    lower_point = (int(cx + nx * t_end), int(cy + ny * t_end))

    if not (scale_x and scale_y):
        return None

    dx_um = abs(lower_point[0] - upper_point[0]) * scale_x
    dy_um = abs(lower_point[1] - upper_point[1]) * scale_y
    return upper_point, lower_point, dx_um, dy_um

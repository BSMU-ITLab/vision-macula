"""Общее ядро измерений объектов относительно линии хориоидеи.

Три почти одинаковые функции исходного кода
(``detect_and_measure_detachments``, ``measure_drusen``,
``measure_neuroepithelial_detachment``) повторяли один алгоритм:

1. левая/правая границы объекта;
2. ширина — длина сглаженной линии под объектом (или верхней границы объекта);
3. высота — наибольший перпендикуляр к линии хориоидеи через объект.

Здесь алгоритм реализован один раз.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

from bsmu.macula.plugins.analyser.geometry import upward_normal
from bsmu.macula.plugins.analyser.schema import to_um, to_um2

#: Максимальная длина перпендикуляра при поиске высоты (пиксели).
DEFAULT_PERP_LENGTH = 400


def area_px(mask: np.ndarray) -> float:
    """Площадь бинарной маски в пикселях."""
    return float(np.sum(mask))


def area_um2(mask: np.ndarray, scale_x: float | None, scale_y: float | None) -> float | None:
    """Площадь бинарной маски в мкм²."""
    return to_um2(area_px(mask), scale_x, scale_y)


def lateral_extent(mask: np.ndarray) -> tuple[int, int] | None:
    """Левая и правая X-границы объекта."""
    _, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(np.min(xs)), int(np.max(xs))


def width_along_boundary(
    boundary: np.ndarray,
    mask: np.ndarray,
    scale_x: float | None,
    scale_y: float | None,
    bridge_gap: int | None = None,
) -> tuple[float | None, list[tuple[int, int] | None]]:
    """Ширина = длина участка границы под объектом.

    :param boundary: сглаженная граница (значение Y на каждый X);
    :param bridge_gap: если задан, очаги, разделённые горизонтальным
        промежутком больше ``bridge_gap``, считаются отдельными, а разрыв
        отмечается в сегменте как ``None``.
    """
    extent = lateral_extent(mask)
    if extent is None:
        return None, []
    left_x, right_x = extent

    has_pixel_in_column = mask.sum(axis=0) > 0
    segment: list[tuple[int, int] | None] = []
    width_um = 0.0 if (scale_x is not None and scale_y is not None) else None
    prev_point: tuple[int, int] | None = None

    for x in range(left_x, right_x + 1):
        if not has_pixel_in_column[x]:
            if segment and segment[-1] is not None:
                segment.append(None)
            prev_point = None
            continue

        point = (x, int(float(boundary[x])))
        if prev_point is not None:
            if bridge_gap is not None and (x - prev_point[0]) > bridge_gap:
                segment.append(None)
            elif width_um is not None:
                width_um += to_um(prev_point, point, scale_x, scale_y) or 0.0
        segment.append(point)
        prev_point = point

    while segment and segment[-1] is None:
        segment.pop()
    return width_um, segment


def perpendicular_points(
    mask: np.ndarray,
    start: tuple[float, float],
    direction: tuple[float, float],
    length: int,
) -> list[tuple[float, float]]:
    """Точки объекта вдоль перпендикуляра в обе стороны от линии хориоидеи."""
    height, width = mask.shape
    fx, fy = float(start[0]), float(start[1])
    nx, ny = direction
    points: list[tuple[float, float]] = []

    for t in range(0, length):
        px = int(round(fx + nx * t))
        py = int(round(fy + ny * t))
        if px < 0 or px >= width or py < 0 or py >= height:
            break
        if mask[py, px] == 1:
            points.append((fx + nx * t, fy + ny * t))

    for t in range(1, length):
        px = int(round(fx - nx * t))
        py = int(round(fy - ny * t))
        if px < 0 or px >= width or py < 0 or py >= height:
            break
        if mask[py, px] == 1:
            points.append((fx - nx * t, fy - ny * t))

    return points


def farthest_pair(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Пара самых удалённых точек и расстояние между ними.

    Точки лежат вдоль почти прямого перпендикуляра, поэтому искомая пара —
    крайние точки. Функция оставлена с полным перебором для точного
    совпадения с прежней реализацией.
    """
    best_distance = -1.0
    best_a, best_b = points[0], points[0]
    for i, point_a in enumerate(points):
        for point_b in points[i + 1:]:
            distance = float(np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))
            if distance > best_distance:
                best_distance = distance
                best_a, best_b = point_a, point_b
    return best_a, best_b, max(best_distance, 0.0)


def max_perpendicular(
    mask: np.ndarray,
    boundary: np.ndarray,
    spline: UnivariateSpline,
    left_x: int,
    right_x: int,
    perp_length: int = DEFAULT_PERP_LENGTH,
) -> tuple[float, tuple[float, float, float, float] | None]:
    """Наибольший перпендикуляр к линии хориоидеи, проходящий через объект.

    :return: ``(высота_в_пикселях, (x1, y1, x2, y2))``.
    """
    max_height_px = 0.0
    max_points: tuple[float, float, float, float] | None = None

    for x in range(left_x, right_x + 1):
        y_base = float(boundary[x])
        nx, ny = upward_normal(spline, x)
        points = perpendicular_points(mask, (x, y_base), (nx, ny), perp_length)

        if len(points) < 2:
            continue

        first, last, distance = farthest_pair(points)
        if distance > max_height_px:
            max_height_px = distance
            max_points = (first[0], first[1], last[0], last[1])

    return max_height_px, max_points


def height_um(
    max_height_px: float,
    max_points: tuple[float, float, float, float] | None,
    scale_x: float | None,
    scale_y: float | None,
) -> float | None:
    """Высота в мкм по крайним точкам перпендикуляра."""
    if max_height_px <= 0 or max_points is None:
        return None
    x1, y1, x2, y2 = max_points
    return to_um((x1, y1), (x2, y2), scale_x, scale_y)


def upper_boundary_of_mask(mask: np.ndarray, left_x: int, right_x: int) -> np.ndarray:
    """Верхняя граница бинарной маски: минимальный Y для каждого X."""
    _, width = mask.shape
    boundary = np.full(width, -1, dtype=np.float32)
    ys, xs = np.where(mask > 0)
    for x in range(left_x, right_x + 1):
        column_ys = ys[xs == x]
        if len(column_ys) > 0:
            boundary[x] = np.min(column_ys)
    return boundary


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Убирает из бинарной маски компоненты меньше ``min_area`` пикселей.

    Нужно, чтобы посторонние кляксы разметки (компас, шкала) не завышали
    высоту, ширину и площадь.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    cleaned = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return cleaned

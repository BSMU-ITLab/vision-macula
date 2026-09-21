"""Анализ ретинального пигментного эпителия (РПЭ, класс 9).

Модуль объединяет три прежние функции (``analyze_rpe``,
``detect_rpe_defects``, ``measure_rpe_thickness``) и выносит общий для них
код измерения толщины вдоль перпендикуляра к скелету РПЭ.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from bsmu.macula.plugins.analyser.schema import CLASS_RPE, Location

#: Текст состояния эпителия.
RPE_PRESERVED = "Эпителий сохранен"
RPE_UNEVEN = "Эпителий неравномерный"
RPE_SINGLE_GAPS = "Единичные разрывы"
RPE_MULTIPLE_GAPS = "Множественные разрывы"
RPE_UNDEFINED = "Эпителий не определяется"

#: Минимальное количество пикселей РПЭ, чтобы считать его определимым.
MIN_RPE_PIXELS = 100
#: Коэффициент вариации высоты, ниже которого эпителий считается равномерным.
UNIFORM_VARIATION_THRESHOLD = 0.2
#: Полуокно поиска соседних точек скелета при оценке касательной.
TANGENT_WINDOW = 10
#: Максимальная длина перпендикуляра при измерении толщины РПЭ.
MAX_PERP_LENGTH = 50


def rpe_mask(mask: np.ndarray) -> np.ndarray:
    """Бинарная маска РПЭ."""
    return (mask == CLASS_RPE).astype(np.uint8)


def local_tangent(skeleton: np.ndarray, y_skel: int, x_skel: int) -> tuple[float, float]:
    """Направление касательной к скелету в точке (через PCA окрестности)."""
    height, width = skeleton.shape
    y0 = max(0, y_skel - TANGENT_WINDOW)
    y1 = min(height, y_skel + TANGENT_WINDOW + 1)
    x0 = max(0, x_skel - TANGENT_WINDOW)
    x1 = min(width, x_skel + TANGENT_WINDOW + 1)

    patch = skeleton[y0:y1, x0:x1]
    local_yx = np.argwhere(patch > 0)
    center_dy = y_skel - y0
    center_dx = x_skel - x0
    not_center = ~((local_yx[:, 0] == center_dy) & (local_yx[:, 1] == center_dx))
    local_yx = local_yx[not_center]

    if len(local_yx) < 2:
        return 0.0, 0.0

    neighbors_x = np.append(local_yx[:, 1] + x0, x_skel)
    neighbors_y = np.append(local_yx[:, 0] + y0, y_skel)
    centered_x = neighbors_x - np.mean(neighbors_x)
    centered_y = neighbors_y - np.mean(neighbors_y)

    cov_xx = np.mean(centered_x * centered_x)
    cov_yy = np.mean(centered_y * centered_y)
    cov_xy = np.mean(centered_x * centered_y)

    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    lambda1 = trace / 2 + np.sqrt(max(0.0, (trace / 2) ** 2 - det))

    if abs(cov_xy) > 1e-6:
        tangent_x, tangent_y = cov_xy, lambda1 - cov_xx
    elif abs(cov_xx - lambda1) > 1e-6:
        tangent_x, tangent_y = 1.0, 0.0
    else:
        tangent_x, tangent_y = 0.0, 1.0

    norm = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    if norm <= 0:
        return 0.0, 1.0
    return tangent_x / norm, tangent_y / norm


def measure_rpe_thickness(
    mask: np.ndarray,
    fovea_mask: np.ndarray,
    scale_x: float | None = None,
    scale_y: float | None = None,
    sample_interval: int = 50,
) -> tuple[np.ndarray | None, list | None, float | None, float | None]:
    """Толщина РПЭ методом скелетизации и построения перпендикуляров.

    :return: ``(точки_скелета, перпендикуляры, средняя_толщина, СКО)``.
    """
    del fovea_mask  # измерение толщины не зависит от зон фовеа

    rpe = rpe_mask(mask)
    if rpe.sum() == 0:
        return None, None, None, None

    skeleton = skeletonize(rpe > 0)
    skeleton_points = np.argwhere(skeleton > 0)
    if len(skeleton_points) == 0:
        return None, None, None, None

    height, width = mask.shape
    thicknesses: list[float] = []
    perpendiculars: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for point in skeleton_points[::sample_interval]:
        y_skel, x_skel = point
        tangent_x, tangent_y = local_tangent(skeleton, y_skel, x_skel)
        if tangent_x == 0.0 and tangent_y == 0.0:
            continue

        perp_x, perp_y = -tangent_y, tangent_x
        # РПЭ — почти горизонтальный слой, толщина измеряется поперёк.
        # Почти горизонтальный перпендикуляр означает ошибку оценки
        # касательной: мерили бы вдоль РПЭ и получили огромную «толщину».
        if abs(perp_y) < abs(perp_x):
            continue

        boundary_1 = None
        boundary_2 = None
        for t in range(0, MAX_PERP_LENGTH):
            xx = int(x_skel + perp_x * t)
            yy = int(y_skel + perp_y * t)
            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                break
            if rpe[yy, xx] > 0:
                boundary_1 = (xx, yy)
            else:
                break

        for t in range(0, MAX_PERP_LENGTH):
            xx = int(x_skel - perp_x * t)
            yy = int(y_skel - perp_y * t)
            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                break
            if rpe[yy, xx] > 0:
                boundary_2 = (xx, yy)
            else:
                break

        if boundary_1 is None or boundary_2 is None or boundary_1 == boundary_2:
            continue

        if scale_x is not None and scale_y is not None:
            dx_um = abs(boundary_1[0] - boundary_2[0]) * scale_x
            dy_um = abs(boundary_1[1] - boundary_2[1]) * scale_y
            thickness = float(np.sqrt(dx_um ** 2 + dy_um ** 2))
        else:
            thickness = float(np.hypot(boundary_1[0] - boundary_2[0], boundary_1[1] - boundary_2[1]))

        thicknesses.append(thickness)
        perpendiculars.append((boundary_1, boundary_2))

    if len(thicknesses) == 0:
        return skeleton_points, [], None, None

    # Отбрасываем выбросы по IQR (метод Тьюки): при ошибке оценки касательной
    # перпендикуляр иногда меряет вдоль РПЭ, раздувая СКО.
    values = np.array(thicknesses, dtype=np.float64)
    if len(values) >= 4:
        q1, q3 = np.percentile(values, [25, 75])
        keep = values <= q3 + 1.5 * (q3 - q1)
        if keep.any():
            values = values[keep]
            perpendiculars = [item for item, flag in zip(perpendiculars, keep) if flag]

    return skeleton_points, perpendiculars, float(np.mean(values)), float(np.std(values))


def analyze_rpe(
    mask: np.ndarray,
    fovea_mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline,
) -> str:
    """Состояние РПЭ: сохранён / неравномерный / разрывы / не определяется."""
    del fovea_mask, smooth_upper, spline

    rpe = rpe_mask(mask)
    total = int(rpe.sum())
    if total == 0 or total < MIN_RPE_PIXELS:
        return RPE_UNDEFINED

    skeleton = skeletonize(rpe > 0)
    contours, _ = cv2.findContours(rpe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gaps = len(contours) - 1
    if gaps > 3:
        return RPE_MULTIPLE_GAPS
    if gaps > 0:
        return RPE_SINGLE_GAPS

    skeleton_points = np.argwhere(skeleton > 0)
    if len(skeleton_points) == 0:
        return RPE_UNDEFINED

    xs = skeleton_points[::10, 1]
    ys = skeleton_points[::10, 0]
    heights: list[int] = []
    for y_skel, x_skel in zip(ys, xs):
        count = 0
        for dy in range(-20, 20):
            yy = y_skel + dy
            if 0 <= yy < rpe.shape[0] and 0 <= x_skel < rpe.shape[1] and rpe[yy, x_skel] > 0:
                count += 1
        if count > 0:
            heights.append(count)

    if not heights:
        return RPE_UNDEFINED

    heights_array = np.array(heights)
    mean_height = float(np.mean(heights_array))
    variation = float(np.std(heights_array)) / mean_height if mean_height > 0 else 0.0
    return RPE_PRESERVED if variation < UNIFORM_VARIATION_THRESHOLD else RPE_UNEVEN


def detect_rpe_defects(
    mask: np.ndarray,
    fovea_mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline,
    min_gap_width: int = 10,
    min_contour_width: int = 5,
) -> tuple[str, list[np.ndarray], tuple[int, int] | None]:
    """Локализация дефектов РПЭ (истончения и разрывы).

    Дефектом считается разрыв между значимыми фрагментами РПЭ; отслойка от
    мембраны Бруха без повреждения целостности РПЭ дефектом не является.

    :return: ``(локализация, контуры разрывов, координата первого дефекта)``.
    """
    del smooth_upper, spline, min_gap_width

    rpe = rpe_mask(mask)
    if rpe.sum() == 0:
        return Location.NONE_ZONE, [], None

    contours, _ = cv2.findContours(rpe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 1:
        return Location.NONE_ZONE, [], None

    height, width = mask.shape
    fragments = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if box_w >= min_contour_width:
            fragments.append((cx, cy, contour, x, y, box_w, box_h))

    if len(fragments) <= 1:
        return Location.NONE_ZONE, [], None

    fragments.sort(key=lambda item: item[0])

    gap_contours: list[np.ndarray] = []
    defect_coordinate: tuple[int, int] | None = None
    gap_circles: list[tuple[int, int, int]] = []

    # Разрывов ровно на один меньше, чем значимых фрагментов: каждый промежуток
    # между двумя соседними фрагментами считается разрывом.
    for i in range(len(fragments) - 1):
        _, _, _, x1, y1, w1, h1 = fragments[i]
        _, _, _, x2, y2, _, h2 = fragments[i + 1]

        gap_width = x2 - (x1 + w1)
        gap_x_center = (x1 + w1 + x2) // 2
        gap_y_center = (y1 + h1 // 2 + y2 + h2 // 2) // 2

        if defect_coordinate is None:
            defect_coordinate = (gap_x_center, gap_y_center)

        gap_radius = max(abs(gap_width), max(h1, h2)) // 2
        gap_circles.append((gap_x_center, gap_y_center, gap_radius))
        gap_contours.append(circle_contour(gap_x_center, gap_y_center, gap_radius))

    if not gap_contours:
        return Location.NONE_ZONE, [], None

    defects_in_foveola = False
    defects_in_fovea = False
    defects_in_macula = False

    for gap_x_center, gap_y_center, _ in gap_circles:
        if 0 <= gap_y_center < height and 0 <= gap_x_center < width:
            zone = fovea_mask[gap_y_center, gap_x_center]
            if zone == 1:
                defects_in_foveola = True
            elif zone == 2:
                defects_in_fovea = True
            else:
                defects_in_macula = True

    if not (defects_in_foveola or defects_in_fovea or defects_in_macula):
        result = Location.NONE_ZONE
    elif defects_in_foveola:
        result = Location.FOVEOLA_FOVEA_MACULA
    elif defects_in_fovea:
        result = Location.FOVEA_MACULA
    else:
        result = Location.MACULA_ONLY

    return result, gap_contours, defect_coordinate


def circle_contour(cx: int, cy: int, r: int, n: int = 32) -> np.ndarray:
    """Окружность как OpenCV-контур формы ``(N, 1, 2)``."""
    r = max(1, int(r))
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = (cx + r * np.cos(thetas)).round().astype(np.int32)
    ys = (cy + r * np.sin(thetas)).round().astype(np.int32)
    return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)


def combined_rpe_state(
    rpe: np.ndarray,
    gap_contours: list[np.ndarray],
    std_thickness: float | None,
    fallback_state: str,
    uneven_std_um: float = 50.0,
) -> str:
    """Итоговое состояние РПЭ по приоритету разрывов и равномерности высоты."""
    if int(rpe.sum()) < MIN_RPE_PIXELS:
        return RPE_UNDEFINED

    gaps = len(gap_contours) if gap_contours else 0
    if gaps > 3:
        return RPE_MULTIPLE_GAPS
    if gaps > 0:
        return RPE_SINGLE_GAPS
    if std_thickness is None:
        return fallback_state or RPE_PRESERVED
    return RPE_PRESERVED if std_thickness < uneven_std_um else RPE_UNEVEN

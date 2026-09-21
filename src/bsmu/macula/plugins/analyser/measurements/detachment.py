"""Измерение отслоек: PED (классы 2, 10, 11, 16) и СРЖ (класс 6)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from bsmu.macula.plugins.analyser.location import location_by_x_columns, location_by_check_order
from bsmu.macula.plugins.analyser.measurements.kernel import (
    DEFAULT_PERP_LENGTH,
    area_um2,
    filter_small_components,
    height_um,
    lateral_extent,
    max_perpendicular,
    upper_boundary_of_mask,
    width_along_boundary,
)
from bsmu.macula.plugins.analyser.schema import CLASS_NEUROEPITHELIAL_DETACHMENT

#: Горизонтальный разрыв (px), после которого очаги класса 6 считаются
#: раздельными и ширина через пустоту не считается.
BRIDGE_GAP = 10
#: Минимальное количество точек для построения сплайна верхней границы.
MIN_SPLINE_POINTS = 4
#: Минимальная площадь компоненты отслойки нейроэпителия (px).
MIN_COMPONENT_AREA = 100

#: Порядок элементов результата измерения отслойки.
DetachmentResult = tuple[
    float | None, float | None, float | None,
    int | None, int | None, tuple | None, str | None, list | None,
]

EMPTY_RESULT: DetachmentResult = (None, None, None, None, None, None, None, None)


def _measure_detachment_core(
    mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    target_class: int,
    scale_x: float | None = None,
    scale_y: float | None = None,
    perp_len: int = DEFAULT_PERP_LENGTH,
) -> DetachmentResult:
    """Измеряет отслойку указанного класса относительно линии хориоидеи."""
    target_mask = (mask == target_class).astype(np.uint8)
    area_px_value = float(np.sum(target_mask))
    if area_px_value == 0:
        return EMPTY_RESULT

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
        area_um2(target_mask, scale_x, scale_y),
        left_x,
        right_x,
        max_perp_point,
        None,
        choroid_segment,
    )


def measure_detachment(
    mask: np.ndarray,
    smooth_upper: np.ndarray,
    spline: UnivariateSpline,
    spec,
    fovea_mask: np.ndarray | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    perp_len: int = DEFAULT_PERP_LENGTH,
) -> DetachmentResult:
    """Измеряет отслойку по спецификации :class:`DetachmentSpec`.

    :return: ``(width, height, area, left_x, right_x, max_perp_point,
        location, choroid_segment)``.
    """
    result = list(
        _measure_detachment_core(
            mask, smooth_upper, spline, spec.class_id, scale_x, scale_y, perp_len
        )
    )
    if result[0] is None and result[1] is None:
        return EMPTY_RESULT

    if fovea_mask is not None:
        target_mask = (mask == spec.class_id).astype(np.uint8)
        _, xs = np.where(target_mask == 1)
        result[6] = location_by_x_columns(fovea_mask, xs)

    return tuple(result)  # type: ignore[return-value]


def measure_neuroepithelial_detachment(
    mask: np.ndarray,
    smooth_upper_choroid: np.ndarray,
    spline_choroid: UnivariateSpline,
    target_class: int = CLASS_NEUROEPITHELIAL_DETACHMENT,
    fovea_mask: np.ndarray | None = None,
    scale_x: float | None = None,
    scale_y: float | None = None,
    perp_len: int = DEFAULT_PERP_LENGTH,
    min_component_area: int = MIN_COMPONENT_AREA,
) -> DetachmentResult:
    """Отслойка нейроэпителия (СРЖ).

    Ширина — длина верхней границы самой отслойки (не хориоидеи);
    высота — наибольший перпендикуляр к линии хориоидеи.
    """
    detachment_mask = (mask == target_class).astype(np.uint8)
    detachment_mask = filter_small_components(detachment_mask, min_component_area)

    area_px_value = float(np.sum(detachment_mask))
    if area_px_value == 0:
        return EMPTY_RESULT

    extent = lateral_extent(detachment_mask)
    if extent is None:
        return EMPTY_RESULT
    left_x, right_x = extent

    upper_boundary = upper_boundary_of_mask(detachment_mask, left_x, right_x)
    valid_xs = [x for x in range(left_x, right_x + 1) if upper_boundary[x] >= 0]
    if len(valid_xs) < MIN_SPLINE_POINTS:
        return EMPTY_RESULT

    valid_ys = [float(upper_boundary[x]) for x in valid_xs]
    try:
        spline_upper = UnivariateSpline(valid_xs, valid_ys, s=len(valid_xs) * 2, k=3)
    except Exception:  # noqa: BLE001 — сплайн может не сойтись на коротких границах
        return EMPTY_RESULT

    width_um, upper_segment = _detachment_upper_width(
        spline_upper, detachment_mask, valid_xs, scale_x, scale_y
    )
    max_height_px, max_perp_point = max_perpendicular(
        detachment_mask, smooth_upper_choroid, spline_choroid, left_x, right_x, perp_len
    )

    location = None
    if fovea_mask is not None:
        ys, xs = np.where(detachment_mask == 1)
        in_foveola, in_fovea, in_macula = _zones_under(fovea_mask, ys, xs)
        if in_foveola or in_fovea or in_macula:
            location = location_by_check_order(in_foveola, in_fovea, in_macula)

    return (
        width_um,
        height_um(max_height_px, max_perp_point, scale_x, scale_y),
        area_um2(detachment_mask, scale_x, scale_y),
        left_x,
        right_x,
        max_perp_point,
        location,
        upper_segment,
    )


def _detachment_upper_width(
    spline_upper: UnivariateSpline,
    detachment_mask: np.ndarray,
    valid_xs: list[int],
    scale_x: float | None,
    scale_y: float | None,
) -> tuple[float | None, list[tuple[int, int] | None]]:
    """Ширина по верхней границе отслойки с разрывами между очагами."""
    from bsmu.macula.plugins.analyser.schema import to_um

    width_um = 0.0 if (scale_x is not None and scale_y is not None) else None
    segment: list[tuple[int, int] | None] = []
    prev_x = prev_y = None

    for x in valid_xs:
        y = float(spline_upper(x))
        if prev_x is not None and (x - prev_x) > BRIDGE_GAP:
            segment.append(None)
        elif prev_x is not None and width_um is not None:
            width_um += to_um((prev_x, prev_y), (x, y), scale_x, scale_y) or 0.0
        segment.append((int(x), int(y)))
        prev_x, prev_y = x, y

    return width_um, segment


def _zones_under(fovea_mask: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple[bool, bool, bool]:
    """Какие зоны фовеа присутствуют под пикселями объекта."""
    from bsmu.macula.plugins.analyser.location import classify_zones

    if len(xs) == 0:
        return False, False, False
    return classify_zones(fovea_mask[ys, xs])

"""Подсветка объектов и измерений при наведении в таблице результатов.

Раньше это была одна функция-замыкание на 450 строк с вложенными ``if`` по
ключам строки результата. Здесь — класс с маленькими методами-отрисовщиками
и одной таблицей диспетчеризации.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.visibility import Visibility

from bsmu.macula.plugins.analyser.scale import calculate_scales_from_L
from bsmu.macula.plugins.analyser.schema import (
    CLASS_INTRARETINAL_HYPERREFLECTIVE,
    CLASS_SUBRETINAL_HYPERREFLECTIVE,
    DETACHMENT_BY_MEASUREMENT_ID,
    HYPERREFLECTIVE_INTRARETINAL,
    HYPERREFLECTIVE_SUBRETINAL,
    RowKey,
)

#: Имя слоя подсветки.
HIGHLIGHT_LAYER_NAME = "hover-highlight"
#: Цвет основной подсветки (BGR — белый).
HIGHLIGHT_COLOR = (255, 255, 255)
#: Цвет линии хориоидеи под объектом.
CHOROID_COLOR = (0, 255, 0)
#: Цвет максимального перпендикуляра.
PERPENDICULAR_COLOR = (0, 0, 255)
#: Цвет контура L-объекта и его рамки.
L_CONTOUR_COLOR = (0, 255, 255)
L_BOX_COLOR = (255, 255, 0)


class OverlayResizeLayeredImage:
    """Прокси-слой, приводящий оверлеи к размеру исходного снимка.

    Оверлеи строятся в размере анализа (1024x512), а ложиться должны на
    снимок в его родном размере.
    """

    def __init__(self, layered_image, analysis_size: tuple[int, int], original_size: tuple[int, int]):
        self._layered_image = layered_image
        self._analysis_size = analysis_size
        self._original_size = original_size

    def __getattr__(self, name):
        return getattr(self._layered_image, name)

    def add_layer_or_modify_pixels(self, name, pixels, *args, **kwargs):
        resized = pixels
        if self._original_size != self._analysis_size:
            resized = cv2.resize(pixels, self._original_size, interpolation=cv2.INTER_NEAREST)
        return self._layered_image.add_layer_or_modify_pixels(name, resized, *args, **kwargs)


class HighlightRenderer:
    """Рисует подсветку строки результата на изображении-оверлее."""

    def __init__(self, layered_image, mask_shape: tuple[int, int],
                 original_size: tuple[int, int], analysis_size: tuple[int, int]):
        self._layered_image = OverlayResizeLayeredImage(layered_image, analysis_size, original_size)
        self._mask_shape = mask_shape

        self._handlers: tuple[tuple[str, Callable[[dict, np.ndarray], None]], ...] = (
            (RowKey.CONTOUR, self._draw_contour),
            (RowKey.DETACHMENT_BOUNDS, self._draw_detachment),
            (RowKey.L_CONTOUR, self._draw_l_contour),
            (RowKey.RPE_CONTOURS, self._draw_rpe_contours),
            (RowKey.GAP_CONTOURS, self._draw_gap_contours),
            (RowKey.IRF_CONTOURS, self._draw_irf_contours),
            (RowKey.ELLIPSOID_CONTOURS, self._draw_zone_contours),
            (RowKey.MYOID_CONTOURS, self._draw_zone_contours),
        )

    # --- публичный вход ---

    def render(self, row_data: dict | None, analyser) -> None:
        """Показывает подсветку для строки ``row_data`` (``None`` — убрать)."""
        if row_data is None:
            self._remove_layer()
            return

        canvas = np.zeros((*self._mask_shape, 3), dtype=np.uint8)
        try:
            self._draw_row(row_data, canvas, analyser)
        except Exception:  # noqa: BLE001 — подсветка не должна ломать окно результатов
            import traceback

            traceback.print_exc()

        if canvas.any():
            self._add_layer(canvas, visibility=1.0)
        else:
            self._remove_layer()

    # --- диспетчеризация ---

    def _draw_row(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        for key, handler in self._handlers:
            if row_data.get(key) is not None:
                handler(row_data, canvas, analyser)
                return

        measurement_id = row_data.get(RowKey.MEASUREMENT_ID, "")
        if "hyperreflective_" in measurement_id:
            self._draw_hyperreflective(measurement_id, canvas, analyser)
        elif measurement_id:
            self._draw_measurement(row_data, canvas, analyser)
        elif analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    # --- отрисовщики ---

    def _draw_l_contour(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contour = row_data.get(RowKey.L_CONTOUR) or analyser.L_contour
        if contour is None:
            return

        cv2.drawContours(canvas, [contour], -1, L_CONTOUR_COLOR, 2)
        box = np.int32(cv2.boxPoints(cv2.minAreaRect(contour)))
        cv2.drawContours(canvas, [box], 0, L_BOX_COLOR, 1)

        scales = calculate_scales_from_L(contour)
        if scales is None:
            return
        _, _, width_px, height_px = scales
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        cv2.putText(canvas, f"W: {width_px:.1f}px = 200um", (cx - 80, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)
        cv2.putText(canvas, f"H: {height_px:.1f}px = 200um", (cx - 80, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, HIGHLIGHT_COLOR, 1)

    def _draw_bounds(self, row_data: dict, canvas: np.ndarray, analyser) -> bool:
        """Рёбра отслойки/друзы: линия хориоидеи и максимальный перпендикуляр."""
        bounds = row_data.get(RowKey.DETACHMENT_BOUNDS)
        if not bounds or len(bounds) != 4:
            return False

        _, _, max_perp_point, choroid_segment = bounds
        if choroid_segment:
            # None в сегменте = разрыв между отдельными очагами: такие пары
            # пропускаем, иначе OpenCV принимает None за (0, 0) и рисует линию
            # из левого верхнего угла кадра.
            for i in range(len(choroid_segment) - 1):
                if choroid_segment[i] is not None and choroid_segment[i + 1] is not None:
                    cv2.line(canvas, choroid_segment[i], choroid_segment[i + 1], CHOROID_COLOR, 2)

        if max_perp_point is not None and len(max_perp_point) == 4:
            x_base, y_base, x_top, y_top = max_perp_point
            base = (int(round(x_base)), int(round(y_base)))
            top = (int(round(x_top)), int(round(y_top)))
            cv2.line(canvas, base, top, PERPENDICULAR_COLOR, 2)
            cv2.circle(canvas, base, 3, (255, 0, 0), -1)
            cv2.circle(canvas, top, 3, (255, 255, 0), -1)

        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)
        return True

    def _draw_contour(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contour = row_data.get(RowKey.CONTOUR)
        if contour is None:
            return
        cv2.drawContours(canvas, [contour], -1, HIGHLIGHT_COLOR, 2)
        self._draw_bounds(row_data, canvas, analyser)

    def _draw_detachment(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        bounds = row_data.get(RowKey.DETACHMENT_BOUNDS)
        if not bounds or len(bounds) != 4:
            self._draw_bounds(row_data, canvas, analyser)
            return

        self._draw_bounds(row_data, canvas, analyser)

        measurement_id = row_data.get(RowKey.MEASUREMENT_ID, "")
        spec = next(
            (item for prefix, item in DETACHMENT_BY_MEASUREMENT_ID.items() if prefix in measurement_id),
            None,
        )
        if spec is None:
            return

        detachment_mask = (analyser.mask == spec.class_id).astype(np.uint8)
        contours, _ = cv2.findContours(detachment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(canvas, contours, -1, HIGHLIGHT_COLOR, 2)

    def _draw_rpe_contours(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contours = row_data.get(RowKey.RPE_CONTOURS)
        if not contours:
            return
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    def _draw_gap_contours(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contours = row_data.get(RowKey.GAP_CONTOURS)
        if not contours:
            return
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)
        for contour in contours:
            cv2.fillPoly(canvas, [contour], (0, 0, 255))
        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    def _draw_irf_contours(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contours = row_data.get(RowKey.IRF_CONTOURS)
        if not contours:
            return
        cv2.drawContours(canvas, contours, -1, (255, 128, 0), 2)
        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    def _draw_zone_contours(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        contours = row_data.get(RowKey.ELLIPSOID_CONTOURS) or row_data.get(RowKey.MYOID_CONTOURS)
        if not contours:
            return
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    def _draw_hyperreflective(self, measurement_id: str, canvas: np.ndarray, analyser) -> None:
        if HYPERREFLECTIVE_SUBRETINAL in measurement_id:
            class_mask = (analyser.mask == CLASS_SUBRETINAL_HYPERREFLECTIVE).astype(np.uint8)
        elif HYPERREFLECTIVE_INTRARETINAL in measurement_id:
            class_mask = (analyser.mask == CLASS_INTRARETINAL_HYPERREFLECTIVE).astype(np.uint8)
        else:
            class_mask = (
                (analyser.mask == CLASS_SUBRETINAL_HYPERREFLECTIVE)
                | (analyser.mask == CLASS_INTRARETINAL_HYPERREFLECTIVE)
            ).astype(np.uint8)

        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, L_CONTOUR_COLOR, 2)

    def _draw_measurement(self, row_data: dict, canvas: np.ndarray, analyser) -> None:
        points = row_data.get(RowKey.POINTS)
        if points and len(points) >= 2:
            cv2.line(canvas, points[0], points[1], HIGHLIGHT_COLOR, 2)

        perpendicular = row_data.get(RowKey.PERPENDICULAR)
        if perpendicular and len(perpendicular) == 2:
            cv2.line(canvas, perpendicular[0], perpendicular[1], HIGHLIGHT_COLOR, 2)

        for upper_point, lower_point in row_data.get(RowKey.PERPENDICULARS, []) or []:
            cv2.line(canvas, upper_point, lower_point, HIGHLIGHT_COLOR, 2)

        for upper_point, lower_point in row_data.get(RowKey.RPE_PERPENDICULARS, []) or []:
            cv2.line(canvas, upper_point, lower_point, (255, 0, 255), 1)

        center = row_data.get(RowKey.CENTER)
        if center is not None:
            cv2.circle(canvas, center, 6, HIGHLIGHT_COLOR, -1)

        if analyser.L_contour is not None:
            self._draw_l_contour({RowKey.L_CONTOUR: analyser.L_contour}, canvas, analyser)

    # --- слой ---

    def _add_layer(self, canvas: np.ndarray, visibility: float) -> None:
        rgba = np.zeros((*canvas.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = canvas
        rgba[:, :, 3] = np.where(canvas.any(axis=2), 255, 0)
        self._layered_image.add_layer_or_modify_pixels(
            HIGHLIGHT_LAYER_NAME,
            rgba,
            FlatImage,
            visibility=Visibility(True, visibility),
        )

    def _remove_layer(self) -> None:
        existing = self._layered_image.layer_by_name(HIGHLIGHT_LAYER_NAME)
        if existing is not None:
            self._layered_image.remove_layer(existing)

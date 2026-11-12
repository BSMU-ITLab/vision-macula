"""Калибровка: поиск L-объекта и расчёт масштаба (мкм/пиксель)."""

from __future__ import annotations

import cv2
import numpy as np

#: Реальный размер L-объекта по умолчанию (мкм).
DEFAULT_L_SIZE_UM = 200.0
#: Порог бинаризации тёмного фона для поиска контура L.
BACKGROUND_THRESHOLD = 200
#: Порог бинаризации внутри угла L.
CORNER_THRESHOLD = 100
#: Доли ширины/высоты ограничивающего прямоугольника под «уголок» L.
CORNER_WIDTH_RATIO = 0.07
CORNER_HEIGHT_RATIO = 0.2

#: Порог яркости линий уголка 200 мкм в левом нижнем углу снимка.
MARKER_THRESHOLD = 200
#: Уголок - тонкие линии 2-6 px; толще - это ткань в углу снимка, а не уголок.
MAX_MARKER_STROKE_PX = 10
#: Эталонный уголок для размера снимка (ширина, высота) - самый частый на снимках набора:
#: ``(x левого края, y низа, ширина, высота)`` в px снимка. Уголок, совпадающий с
#: эталоном до ``MARKER_TOLERANCE_PX``, получает его размеры: дрожание порога на JPEG и
#: ткань рядом с уголком не должны менять масштаб между одинаковыми снимками.
REFERENCE_MARKERS = {
    (1172, 438): (11, 426, 26, 50),
    (1174, 438): (12, 426, 26, 50),
    (1267, 907): (10, 897, 40, 168),
    (1346, 907): (11, 896, 31, 83),
    (1941, 1185): (6, 1180, 45, 84),
}
MARKER_TOLERANCE_PX = 2
#: Снимки без уголка (12, 6g, 4g, 2v): масштаб по размеру снимка - у снимков одного
#: размера он один. Внутри типа размер гуляет на несколько px (обрезка скриншотов),
#: поэтому заданы диапазоны. Масштабы в мкм/px рабочего размера анализа 1024x512:
#: 12/6g - ширина скана 6 мм, высота откалибрована по линиям врача; 4g/2v - косвенная
#: калибровка.
SCALE_BY_IMAGE_SIZE = (
    # (ширина от, до), (высота от, до), (scale_x, scale_y)
    ((490, 500), (255, 264), (6000.0 / 1024, 2.94)),  # 12
    ((668, 679), (382, 391), (6000.0 / 1024, 2.94)),  # 6g
    ((439, 449), (342, 352), (5.310, 4.285)),  # 4g
    ((723, 732), (551, 560), (8.496, 4.094)),  # 2v
)


def find_L_in_image(image: np.ndarray) -> np.ndarray | None:
    """Находит белый L-образный уголок на тёмном фоне.

    Сначала ищет самый большой контур, затем левый нижний угол внутри него.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    corner_w = max(1, int(CORNER_WIDTH_RATIO * width))
    corner_h = max(1, int(CORNER_HEIGHT_RATIO * height))
    corner = gray[y + height - corner_h : y + height, x : x + corner_w]

    _, corner_bin = cv2.threshold(corner, CORNER_THRESHOLD, 255, cv2.THRESH_BINARY)
    corner_contours, _ = cv2.findContours(corner_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not corner_contours:
        return None

    contour = max(
        corner_contours,
        key=lambda item: cv2.boundingRect(item)[2] * cv2.boundingRect(item)[3],
    )
    return contour + np.array([[[x, y]]], dtype=np.int32)


def calculate_scale_from_L(contour: np.ndarray, L_size_micrometers: float = DEFAULT_L_SIZE_UM) -> float | None:
    """Единый масштаб (мкм/пиксель) по большей стороне L-контура."""
    if contour is None or len(contour) < 3:
        return None

    rect = cv2.minAreaRect(contour)
    if rect is None:
        return None

    width, height = rect[1]
    if width <= 0 or height <= 0:
        return None
    return L_size_micrometers / max(width, height)


def calculate_scales_from_L(contour: np.ndarray, L_size_micrometers: float = DEFAULT_L_SIZE_UM) -> tuple | None:
    """Масштабы по осям (мкм/пиксель) и размеры L в пикселях."""
    if contour is None or len(contour) < 3:
        return None

    rect = cv2.minAreaRect(contour)
    if rect is None:
        return None

    width, height = rect[1]
    if width <= 0 or height <= 0:
        return None

    return (L_size_micrometers / width, L_size_micrometers / height, width, height)


def measure_L_marker(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Уголок 200 мкм в левом нижнем углу снимка.

    :return: ``(x левого края, y низа, ширина, высота)`` в px снимка либо ``None``.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    top = int(height * 0.70)
    region = gray[top:height, 0 : int(width * 0.33)]
    _, binary = cv2.threshold(region, MARKER_THRESHOLD, 255, cv2.THRESH_BINARY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    best = None
    for i in range(1, count):
        x, y, _, box_h, area = stats[i]
        if box_h < region.shape[0] * 0.20 or area < 15:
            continue
        component = labels == i
        base_width = int(component[max(0, y + box_h - 4) : y + box_h, :].any(axis=0).sum())
        if base_width < 8:
            continue
        stroke = np.median(component[y + box_h // 4 : y + box_h // 2, :].sum(axis=1))
        if stroke > MAX_MARKER_STROKE_PX:
            continue
        if best is None or x < best[0]:
            best = (int(x), top + int(y + box_h), base_width, int(box_h))
    return best


def _marker_scale(marker, width: int, height: int, analysis_size: tuple[int, int]) -> tuple[float, float]:
    marker_w, marker_h = marker[2:]
    return (
        DEFAULT_L_SIZE_UM / marker_w * width / analysis_size[0],
        DEFAULT_L_SIZE_UM / marker_h * height / analysis_size[1],
    )


def image_scale(image: np.ndarray, analysis_size: tuple[int, int]) -> tuple[float, float, str] | None:
    """Масштаб снимка (мкм/пиксель) в рабочем размере анализа.

    По уголку на самом снимке (``"L"``; совпадающий с эталоном для этого размера -
    по эталону), без уголка - по размеру снимка (``"size"``).

    :return: ``(scale_x, scale_y, источник)`` либо ``None``, если уголка нет и
        размер снимка неизвестен.
    """
    height, width = image.shape[:2]
    reference = REFERENCE_MARKERS.get((width, height))
    marker = measure_L_marker(image)
    if marker is not None:
        if reference is not None and all(abs(a - b) <= MARKER_TOLERANCE_PX for a, b in zip(marker, reference)):
            marker = reference
        return (*_marker_scale(marker, width, height, analysis_size), 'L')
    if reference is not None:
        return (*_marker_scale(reference, width, height, analysis_size), 'size')
    for (w_min, w_max), (h_min, h_max), scale in SCALE_BY_IMAGE_SIZE:
        if w_min <= width <= w_max and h_min <= height <= h_max:
            return (*scale, 'size')
    return None

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


def find_L_in_image(image: np.ndarray) -> np.ndarray | None:
    """Находит белый L-образный уголок на тёмном фоне.

    Сначала ищет самый большой контур, затем левый нижний угол внутри него.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    _, binary = cv2.threshold(gray, BACKGROUND_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    corner_w = max(1, int(CORNER_WIDTH_RATIO * width))
    corner_h = max(1, int(CORNER_HEIGHT_RATIO * height))
    corner = gray[y + height - corner_h:y + height, x:x + corner_w]

    _, corner_bin = cv2.threshold(corner, CORNER_THRESHOLD, 255, cv2.THRESH_BINARY)
    corner_contours, _ = cv2.findContours(corner_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not corner_contours:
        return None

    contour = max(
        corner_contours,
        key=lambda item: cv2.boundingRect(item)[2] * cv2.boundingRect(item)[3],
    )
    return contour + np.array([[[x, y]]], dtype=np.int32)


def calculate_scale_from_L(contour: np.ndarray,
                           L_size_micrometers: float = DEFAULT_L_SIZE_UM) -> float | None:
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


def calculate_scales_from_L(contour: np.ndarray,
                            L_size_micrometers: float = DEFAULT_L_SIZE_UM) -> tuple | None:
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

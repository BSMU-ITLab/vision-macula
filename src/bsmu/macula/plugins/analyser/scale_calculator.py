"""Модуль для работы с масштабом изображения (калибровка по L-объекту)."""
from __future__ import annotations
from typing import Tuple

import cv2
import numpy as np

ANALYSIS_SIZE = (1024, 512)  # (width, height)
L_SIZE_UM = 200.0
# Уголок - тонкие линии 2-6 px; толще - это ткань в углу снимка, а не уголок.
MAX_L_STROKE_PX = 10

# Снимки без уголка (12, 6g, 4g, 2v): масштаб по размеру снимка - у снимков одного размера он один.
# Внутри типа размер гуляет на несколько px (обрезка скриншотов), поэтому заданы диапазоны.
# Масштабы в мкм/px пространства анализа. 12/6g: ширина скана 6 мм, высота откалибрована
# по линиям врача; 4g/2v - косвенная калибровка.
SCALE_BY_IMAGE_SIZE = (
    # (ширина от, до), (высота от, до), (scale_x, scale_y)
    ((490, 500), (255, 264), (6000.0 / ANALYSIS_SIZE[0], 2.94)),  # 12
    ((668, 679), (382, 391), (6000.0 / ANALYSIS_SIZE[0], 2.94)),  # 6g
    ((439, 449), (342, 352), (5.310, 4.285)),                     # 4g
    ((723, 732), (551, 560), (8.496, 4.094)),                     # 2v
)
# Уголка нет и размер неизвестен: эталонный уголок скана 021-01-R - 24 px по X, 60 px по Y.
DEFAULT_SCALE = (L_SIZE_UM / 24.0, L_SIZE_UM / 60.0)

# Эталонный уголок для размера снимка (ширина, высота) - самый частый на снимках набора.
# Уголок, совпадающий с эталоном до MARKER_TOLERANCE_PX, получает его размеры: дрожание порога
# на JPEG и ткань рядом с уголком не должны менять масштаб между одинаковыми снимками.
REFERENCE_MARKERS = {
    (1172, 438): (11, 426, 26, 50),
    (1174, 438): (12, 426, 26, 50),
    (1267, 907): (10, 897, 40, 168),
    (1346, 907): (11, 896, 31, 83),
    (1941, 1185): (6, 1180, 45, 84),
}
MARKER_TOLERANCE_PX = 2


def measure_L_marker(image: np.ndarray, threshold: int = 200) -> tuple[int, int, int, int] | None:
    """Уголок 200 мкм в левом нижнем углу: (x левого края, y низа, ширина, высота) в px снимка или None."""
    gray = image if image.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top = int(h * 0.70)
    region = gray[top:h, 0:int(w * 0.33)]
    _, binary = cv2.threshold(region, threshold, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    best = None
    for i in range(1, n):
        x, y, _, height, area = stats[i]
        if height < region.shape[0] * 0.20 or area < 15:
            continue
        component = labels == i
        base_width = int(component[max(0, y + height - 4):y + height, :].any(axis=0).sum())
        if base_width < 8:
            continue
        stroke = np.median(component[y + height // 4:y + height // 2, :].sum(axis=1))
        if stroke > MAX_L_STROKE_PX:
            continue
        if best is None or x < best[0]:
            best = (int(x), top + int(y + height), base_width, int(height))
    return best


def _marker_scale(marker, w, h, analysis_size):
    width_px, height_px = marker[2:]
    return (L_SIZE_UM / width_px * w / analysis_size[0],
            L_SIZE_UM / height_px * h / analysis_size[1])


def image_scale(image: np.ndarray, analysis_size: tuple[int, int] = ANALYSIS_SIZE) -> tuple[float, float, str]:
    """(scale_x, scale_y, источник) в мкм/px пространства анализа.

    Уголок на самом снимке ("L"; совпадающий с эталоном для этого размера - по эталону);
    без уголка - по размеру снимка ("size"); размер неизвестен - DEFAULT_SCALE ("default")."""
    h, w = image.shape[:2]
    reference = REFERENCE_MARKERS.get((w, h))
    marker = measure_L_marker(image)
    if marker is not None:
        if reference is not None and all(abs(a - b) <= MARKER_TOLERANCE_PX for a, b in zip(marker, reference)):
            marker = reference
        return (*_marker_scale(marker, w, h, analysis_size), "L")
    if reference is not None:
        return (*_marker_scale(reference, w, h, analysis_size), "size")
    for (w_min, w_max), (h_min, h_max), scale in SCALE_BY_IMAGE_SIZE:
        if w_min <= w <= w_max and h_min <= h <= h_max:
            return (*scale, "size")
    return (*DEFAULT_SCALE, "default")


class ScaleCalculator:
    """Класс для расчета и хранения масштабов изображения."""
    
    def __init__(self):
        self.scale_x: float | None = None  # мкм/пиксель по X
        self.scale_y: float | None = None  # мкм/пиксель по Y
        self.L_contour: np.ndarray | None = None  # Контур L-объекта для визуализации
        self.L_width_px: float | None = None  # Ширина L-объекта в пикселях
        self.L_height_px: float | None = None  # Высота L-объекта в пикселях
    
    def px_to_um_x(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры по X."""
        if self.scale_x is None:
            return None
        return value_px * self.scale_x
    
    def px_to_um_y(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры по Y."""
        if self.scale_y is None:
            return None
        return value_px * self.scale_y
    
    def px_to_um(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры (использует среднее из X и Y)."""
        if self.scale_x is None or self.scale_y is None:
            return None
        avg_scale = (self.scale_x + self.scale_y) / 2
        return value_px * avg_scale
    
    def calculate_distance_um(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float | None:
        """
        Вычисляет расстояние между двумя точками в микрометрах.
        
        Args:
            point1: кортеж (x, y) первой точки
            point2: кортеж (x, y) второй точки
            
        Returns:
            Расстояние в микрометрах или None если масштаб не задан
        """
        if self.scale_x is None or self.scale_y is None:
            return None
        
        dx_px = abs(point2[0] - point1[0])
        dy_px = abs(point2[1] - point1[1])
        
        dx_um = dx_px * self.scale_x
        dy_um = dy_px * self.scale_y
        
        total_dist_um = (dx_um**2 + dy_um**2)**0.5
        return total_dist_um
    
    @property
    def is_calibrated(self) -> bool:
        """Проверяет, был ли установлен масштаб."""
        return self.scale_x is not None and self.scale_y is not None

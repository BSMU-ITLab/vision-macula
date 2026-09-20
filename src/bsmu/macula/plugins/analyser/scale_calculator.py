"""Модуль для работы с масштабом изображения (калибровка по L-объекту)."""
from __future__ import annotations
from typing import Tuple
import numpy as np


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

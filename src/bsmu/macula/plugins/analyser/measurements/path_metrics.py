"""Пересчёт длины пути до хориоидеи («чистая» сетчатка).

Общая арифметика ЦТС для фовеолы и для точек рядом с фовеолой/фовеей:
полная длина пути в мкм умножается на долю пути, не занятую патологиями.
"""

from __future__ import annotations

import numpy as np


def corrected_distance_um(
    full_um: float | None,
    full_px: float,
    corrected_px: float,
) -> float | None:
    """Переводит «очищенную» длину пути из пикселей в мкм."""
    if full_um is None or full_px <= 0:
        return None
    return full_um * corrected_px / full_px


def distance_px(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Длина отрезка между двумя точками в пикселях."""
    return float(np.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1]))

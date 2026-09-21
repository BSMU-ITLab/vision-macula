"""Синтетические маски для тестов анализатора.

Позволяют прогнать весь конвейер анализа без реальных снимков: строится
изображение с «хориоидеей», «РПЭ», «фовеолой» и заданными патологиями.
"""

from __future__ import annotations

import numpy as np

from bsmu.macula.plugins.analyser.schema import (
    CLASS_CHOROID,
    CLASS_ELLIPSOID_ZONE,
    CLASS_INTRARETINAL_HYPERREFLECTIVE,
    CLASS_IRF,
    CLASS_MYOID_ZONE,
    CLASS_NEUROEPITHELIAL_DETACHMENT,
    CLASS_RPE,
    CLASS_SUB_BRUCH_FLUID,
    CLASS_SUBRETINAL_HYPERREFLECTIVE,
    FOVEA_ZONE,
    FOVEOLA_ZONE,
    MACULA_ZONE,
)

#: Размеры тестовой сцены.
HEIGHT = 300
WIDTH = 600
#: Класс «друзы» в тестовой конфигурации.
CLASS_DRUSEN = 1
#: Класс серозной отслойки ПЭ.
CLASS_SEROUS_PED = 2


def _ellipse_region(y0: int, y1: int, x0: int, x1: int) -> tuple[slice, slice]:
    return slice(y0, y1), slice(x0, x1)


def build_mask() -> np.ndarray:
    """Маска сегментации с хориоидеей и одной находкой каждого типа."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    # Хориоидея: полоса с наклоном и «ямкой» под фовеолой.
    for x in range(WIDTH):
        base = 200 + int(15 * np.sin(x / 90.0))
        mask[base:base + 30, x] = CLASS_CHOROID

    # РПЭ: непрерывная полоса выше хориоидеи с разрывом (дефектом).
    for x in range(60, 540):
        if 280 <= x <= 300:
            continue
        mask[170:176, x] = CLASS_RPE

    mask[_ellipse_region(150, 170, 120, 180)] = CLASS_SEROUS_PED
    mask[_ellipse_region(140, 170, 240, 300)] = CLASS_DRUSEN
    mask[_ellipse_region(150, 175, 360, 420)] = CLASS_NEUROEPITHELIAL_DETACHMENT
    mask[_ellipse_region(120, 140, 430, 470)] = CLASS_SUBRETINAL_HYPERREFLECTIVE
    mask[_ellipse_region(100, 120, 470, 500)] = CLASS_INTRARETINAL_HYPERREFLECTIVE
    mask[_ellipse_region(130, 150, 60, 100)] = CLASS_IRF
    mask[_ellipse_region(180, 190, 500, 540)] = CLASS_SUB_BRUCH_FLUID
    mask[_ellipse_region(176, 180, 60, 400)] = CLASS_ELLIPSOID_ZONE
    mask[_ellipse_region(180, 184, 200, 400)] = CLASS_MYOID_ZONE
    return mask


def build_fovea_mask() -> np.ndarray:
    """Маска зон: фовеола в центре, вокруг фовеа, дальше макула."""
    fovea = np.full((HEIGHT, WIDTH), MACULA_ZONE, dtype=np.uint8)
    fovea[:, :20] = 0
    fovea[:, 580:] = 0
    fovea[100:200, 260:340] = FOVEA_ZONE
    fovea[130:170, 285:315] = FOVEOLA_ZONE
    return fovea


def build_image() -> np.ndarray:
    """Серое изображение с ярким L-объектом в левом нижнем углу."""
    image = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
    image[200:250, 30:45] = 255
    image[235:250, 30:70] = 255
    return image


def build_class_configs() -> list[dict]:
    return [
        {"id": CLASS_DRUSEN, "name": "Drusen"},
        {"id": CLASS_SEROUS_PED, "name": "Serous PED"},
    ]

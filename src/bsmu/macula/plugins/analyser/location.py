"""Определение локализации находок по зонам макулы (маска ``mask-fovea``).

В исходном коде логика «в какой зоне находится объект» была скопирована
четыре раза (отслойки, друзы, нейроэпителиальная отслойка, СРЖ) и успела
разойтись. Здесь она собрана в одном месте.

Зоны маски фовеа: 0 — фон, 1 — фовеола, 2 — фовеа, 3 — макула.
"""

from __future__ import annotations

import numpy as np

from bsmu.macula.plugins.analyser.schema import (
    BACKGROUND_ZONE,
    FOVEA_ZONE,
    FOVEOLA_ZONE,
    MACULA_ZONE,
    Location,
)


def classify_zones(values: np.ndarray) -> tuple[bool, bool, bool]:
    """Какие зоны присутствуют в переданных значениях маски фовеа."""
    unique = np.unique(values)
    in_foveola = FOVEOLA_ZONE in unique
    in_fovea = FOVEA_ZONE in unique
    in_macula = (BACKGROUND_ZONE in unique) or (MACULA_ZONE in unique)
    return in_foveola, in_fovea, in_macula


def location_by_zone(in_foveola: bool, in_fovea: bool, in_macula: bool) -> str:
    """Локализация с приоритетом фовеола → фовеа → макула."""
    if in_foveola:
        return Location.FOVEOLA_FOVEA_MACULA
    if in_fovea:
        return Location.FOVEA_MACULA
    if in_macula:
        return Location.MACULA_ONLY
    return Location.ABSENT


def location_by_check_order(in_foveola: bool, in_fovea: bool, in_macula: bool) -> str:
    """Локализация с нумерацией зон по порядку (СРЖ, гиперрефлективный материал)."""
    if in_foveola:
        return Location.FOVEOLA
    if in_fovea:
        return Location.FOVEA
    if in_macula:
        return Location.MACULA
    return Location.OUTSIDE_MACULA


def location_from_fovea_mask(fovea_mask: np.ndarray, object_mask: np.ndarray,
                             mode: str = "zone") -> str | None:
    """Локализация объекта по пикселям маски фовеа под этим объектом.

    :param mode: ``"zone"`` — приоритет фовеола → фовеа → макула,
        ``"order"`` — нумерация зон по порядку.
    :return: текст локализации либо ``None``, если пикселей/зон нет.
    """
    ys, xs = np.where(object_mask > 0)
    if len(xs) == 0:
        return None

    in_foveola, in_fovea, in_macula = classify_zones(fovea_mask[ys, xs])
    if not (in_foveola or in_fovea or in_macula):
        return None

    if mode == "order":
        return location_by_check_order(in_foveola, in_fovea, in_macula)
    return location_by_zone(in_foveola, in_fovea, in_macula)


def location_by_x_columns(fovea_mask: np.ndarray, xs: np.ndarray) -> str | None:
    """Локализация по набору X-координат объекта.

    Объект считается лежащим в зоне, если хотя бы в одной из его X-колонок
    маска фовеа содержит пиксель этой зоны.
    """
    _, width = fovea_mask.shape[:2]
    zone_values: list[np.ndarray] = []
    for x in np.unique(xs):
        if 0 <= x < width:
            zone_values.append(fovea_mask[:, x])

    if not zone_values:
        return None

    in_foveola, in_fovea, in_macula = classify_zones(np.concatenate(zone_values))
    if not (in_foveola or in_fovea or in_macula):
        return None
    return location_by_zone(in_foveola, in_fovea, in_macula)


def cme_location(fovea_mask: np.ndarray, object_mask: np.ndarray) -> str:
    """Локализация кистозного макулярного отека (отдельный текст «отсутствует»)."""
    if object_mask.sum() == 0:
        return Location.CME_ABSENT
    return location_from_fovea_mask(fovea_mask, object_mask) or Location.CME_ABSENT

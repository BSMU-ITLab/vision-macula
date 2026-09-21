"""Единый источник правды о структуре результатов анализа.

До рефакторинга «контракт» между анализатором, таблицей результатов и
конвертером в ``PatientExamData`` существовал только в виде магических строк,
разбросанных по трём модулям: названия классов отслоек, суффиксы
``measurement_id``, ключи строк (``points``, ``contour``, ``detachment_bounds``,
``gap_contours`` ...) и тексты локализаций дублировались и расходились.

Этот модуль собирает весь контракт в одном месте:

* :class:`RowKey` - ключи словаря-строки измерения;
* :data:`DETACHMENTS` / :class:`DetachmentSpec` - спецификации отслоек;
* :class:`MeasurementId` - идентификаторы измерений;
* :data:`LOCATION_*` - канонические тексты локализаций;
* :func:`detachment_rows` - построение строк отслойки (без дублирования
  четырёх однотипных ``rows.append`` на каждую отслойку).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


class RowKey:
    """Ключи словаря-строки, из которых состоит результат анализа.

    Строка результата намеренно остаётся обычным ``dict`` (её читают и
    мутируют Qt-модели), но имена ключей больше не пишутся строками по коду.
    """

    PARAMETER = "parameter"
    VALUE = "value"
    UNIT = "unit"
    MEASUREMENT_ID = "measurement_id"
    IS_GROUP = "is_group"
    EXPANDED = "expanded"
    CHILDREN = "children"

    # Данные для подсветки объектов в окне визуализации
    POINTS = "points"
    PERPENDICULAR = "perpendicular"
    PERPENDICULARS = "perpendiculars"
    RPE_PERPENDICULARS = "rpe_perpendiculars"
    SKELETON_POINTS = "skeleton_points"
    CENTER = "center"
    CONTOUR = "contour"
    DETACHMENT_BOUNDS = "detachment_bounds"
    DEFECT_COORD = "defect_coord"
    RPE_CONTOURS = "rpe_contours"
    GAP_CONTOURS = "gap_contours"
    IRF_CONTOURS = "irf_contours"
    ELLIPSOID_CONTOURS = "ellipsoid_contours"
    MYOID_CONTOURS = "myoid_contours"
    L_CONTOUR = "L_contour"


class MeasurementId:
    """Идентификаторы измерений (``measurement_id``)."""

    CTS_FOVEOLA = "cts_foveola"
    CTS_NEAR_FOVEOLA = "cts_near_foveola"
    CTS_NEAR_FOVEA = "cts_near_fovea"
    CHOROID_THICKNESS_CENTER = "choroid_thickness_center"

    SBF_AREA = "sbf_area"
    SBF_LOCATION = "sbf_location"

    HYPERREFLECTIVE_LOCATION = "hyperreflective_material_location"
    HYPERREFLECTIVE_SUBRETINAL_AREA = "hyperreflective_subretinal_area"
    HYPERREFLECTIVE_INTRARETINAL_AREA = "hyperreflective_intraretinal_area"
    HYPERREFLECTIVE_TOTAL_AREA = "hyperreflective_total_area"


class Parameter:
    """Названия строк «Параметр» (они же ключи ручных полей и полей БД)."""

    VISIT_DATE = "Дата визита"
    DISEASE_DURATION = "Длительность заболевания"
    AREDS = "Стадия по AREDS"
    NEOVASCULARIZATION = "Тип неоваскуляризации"
    REFRACTION = "Рефракция"
    BCVA = "МКОЗ"
    RETINA_VOLUME = "Объем сетчатки"
    DRUG = "Препарат"
    INJECTIONS_PLANNED = "Количество назначенных инъекций"
    INJECTIONS_DONE = "Количество выполненных инъекций"

    RPE_CONDITION = "Состояние РПЭ"
    RPE_DEFECTS_LOCATION = "Локализация дефектов РПЭ"
    ELLIPSOID_CONDITION = "Состояние эллипсоидной зоны"
    ELLIPSOID_DEFECTS_LOCATION = "Локализация дефектов эллипсоидной зоны"
    MYOID_CONDITION = "Состояние миоидной зоны"
    MYOID_DEFECTS_LOCATION = "Локализация дефектов миоидной зоны"
    CME_LOCATION = "Локализация кистозного макулярного отека"
    HYPERREFLECTIVE_LOCATION = "Гиперрефлективный материал (локализация)"

    #: Группа ручного заполнения (значения вводит пользователь)
    MANUAL_GROUP = "       "


#: Поля, которые пользователь может редактировать в основной таблице.
EDITABLE_PARAMETERS = frozenset(
    {
        Parameter.VISIT_DATE,
        Parameter.DISEASE_DURATION,
        Parameter.AREDS,
        Parameter.NEOVASCULARIZATION,
        Parameter.REFRACTION,
        Parameter.BCVA,
        Parameter.RETINA_VOLUME,
        Parameter.DRUG,
        Parameter.INJECTIONS_PLANNED,
        Parameter.INJECTIONS_DONE,
        Parameter.CME_LOCATION,
    }
)

#: Ручные поля в порядке отображения: (название, единица измерения).
MANUAL_FIELDS: tuple[tuple[str, str], ...] = (
    (Parameter.VISIT_DATE, ""),
    (Parameter.DISEASE_DURATION, ""),
    (Parameter.AREDS, ""),
    (Parameter.NEOVASCULARIZATION, ""),
    (Parameter.REFRACTION, "D"),
    (Parameter.BCVA, ""),
    (Parameter.RETINA_VOLUME, "мм³"),
    (Parameter.DRUG, ""),
    (Parameter.INJECTIONS_PLANNED, ""),
    (Parameter.INJECTIONS_DONE, ""),
)


class Location:
    """Канонические тексты локализаций.

    Используются два взаимно однозначных набора строк:

    * ``FOVEOLA_FOVEA_MACULA`` / ``FOVEA_MACULA`` / ``MACULA_ONLY`` - приоритет
      фовеола → фовеа → макула (дефекты РПЭ, друзы, отслойки);
    * ``FOVEOLA`` / ``FOVEA`` / ``MACULA`` / ``OUTSIDE_MACULA`` - нумерация зон
      по порядку (СРЖ, гиперрефлективный материал).
    """

    NONE_ZONE = "0 - Дефекты отсутствуют"
    FOVEOLA_FOVEA_MACULA = "2 - Фовеола + фовеа + макула"
    FOVEA_MACULA = "1 - Фовеа + макула (без фовеолы)"
    MACULA_ONLY = "3 - Макула (без фовеа и фовеолы)"
    ABSENT = "0 - отсутствует"

    FOVEOLA = "0 - Фовеола"
    FOVEA = "1 - Фовеа (без фовеолы)"
    MACULA = "2 - Макула (без фовеолы и фовеа)"
    OUTSIDE_MACULA = "3 - Вне макулы"

    CME_ABSENT = "0 - Отек отсутствует"


#: Идентификаторы зон маски фовеа.
FOVEOLA_ZONE = 1
FOVEA_ZONE = 2
MACULA_ZONE = 3
BACKGROUND_ZONE = 0

#: Классы сегментации, используемые анализатором.
CLASS_IRF = 3
CLASS_SUBRETINAL_HYPERREFLECTIVE = 4
CLASS_INTRARETINAL_HYPERREFLECTIVE = 5
CLASS_NEUROEPITHELIAL_DETACHMENT = 6
CLASS_SUB_BRUCH_FLUID = 7
CLASS_RPE = 9
CLASS_ELLIPSOID_ZONE = 12
CLASS_MYOID_ZONE = 13
CLASS_CHOROID = 14

#: Идентификаторы, различающие состояние гиперрефлективного материала.
HYPERREFLECTIVE_SUBRETINAL = "subretinal"
HYPERREFLECTIVE_INTRARETINAL = "intraretinal"

#: Суффиксы измерений, порождаемых для каждого объекта/отслойки.
SUFFIX_WIDTH = "width"
SUFFIX_HEIGHT = "height"
SUFFIX_AREA = "area"
SUFFIX_LOCATION = "location"


@dataclass(frozen=True)
class DetachmentSpec:
    """Описание одного типа отслойки.

    :param class_id: класс сегментации на маске;
    :param section: заголовок секции в таблице результатов;
    :param measurement_id: префикс ``measurement_id`` (``serous_ped_width`` и т.д.);
    :param table_item: подпись в таблице деталей отслоек.
    """

    class_id: int
    section: str
    measurement_id: str
    table_item: str

    def row_id(self, suffix: str) -> str:
        return f"{self.measurement_id}_{suffix}"


SEROUS_PED = DetachmentSpec(2, "Серозная отслойка ПЭ (СОПЭ)", "serous_ped", "Серозная ОПЭ")
HEMORRHAGIC_PED = DetachmentSpec(
    16, "Геморрагическая отслойка ПЭ (ГОПЭ)", "hemorrhagic_ped", "Геморрагическая ОПЭ"
)
FIBROVASCULAR_PED = DetachmentSpec(
    10, "Фиброваскулярная отслойка ПЭ (ФВОПЭ)", "fibrovascular_ped", "Фиброваскулярная ОПЭ"
)
DRUSENOID_PED = DetachmentSpec(
    11, "Друзеноидная отслойка ПЭ", "drusenoid_ped", "Друзеноидная ОПЭ"
)
NEUROEPITHELIAL_DETACHMENT = DetachmentSpec(
    6, "Отслойка нейроэпителия", "neuroepithelial_detachment", "Отслойка нейроэпителия"
)

#: PED-отслойки: отдельные измерения ширины/высоты/площади/локализации.
PED_DETACHMENTS: tuple[DetachmentSpec, ...] = (
    SEROUS_PED,
    HEMORRHAGIC_PED,
    FIBROVASCULAR_PED,
    DRUSENOID_PED,
)

#: Все отслойки, включая отслойку нейроэпителия (СРЖ).
ALL_DETACHMENTS: tuple[DetachmentSpec, ...] = PED_DETACHMENTS + (NEUROEPITHELIAL_DETACHMENT,)

#: Соответствие «класс сегментации -> спецификация отслойки».
DETACHMENT_BY_CLASS: dict[int, DetachmentSpec] = {spec.class_id: spec for spec in ALL_DETACHMENTS}

#: Соответствие «префикс measurement_id -> спецификация отслойки».
DETACHMENT_BY_MEASUREMENT_ID: dict[str, DetachmentSpec] = {
    spec.measurement_id: spec for spec in ALL_DETACHMENTS
}

#: Подсказки для распознавания класса «друзы» по названию из конфигурации.
DRUSEN_CLASS_HINT = "drusen"
DRUSEN_CLASS_HINT_RU = "друз"
DRUSEN_EXCLUDE_HINT = "друзеноид"


def is_drusen_class(class_name: str) -> bool:
    """Похоже ли название класса сегментации на «друзы».

    Друзеноидная отслойка обрабатывается отдельно и сюда не попадает.
    """
    lowered = class_name.lower()
    is_drusen = DRUSEN_CLASS_HINT in lowered or DRUSEN_CLASS_HINT_RU in lowered
    return is_drusen and DRUSEN_EXCLUDE_HINT not in lowered


def is_drusen_parameter(parameter: str) -> bool:
    """Строка таблицы описывает измерение конкретной друзы (``...#1 (ширина)``)."""
    return "#" in parameter and (
        DRUSEN_CLASS_HINT in parameter.lower() or DRUSEN_CLASS_HINT_RU in parameter.lower()
    )


def drusen_number(parameter: str) -> int | None:
    """Номер друзы из строки вида ``Drusen #1 (ширина)``."""
    hash_index = parameter.find("#")
    if hash_index < 0:
        return None
    digits = ""
    for char in parameter[hash_index + 1 :]:
        if not char.isdigit():
            break
        digits += char
    return int(digits) if digits else None


def to_um2(area_px: float, scale_x: float | None, scale_y: float | None) -> float | None:
    """Переводит площадь из пикселей в мкм² (``None``, если масштаб неизвестен)."""
    if scale_x is None or scale_y is None:
        return None
    return float(area_px) * scale_x * scale_y


def to_um(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
    scale_x: float | None,
    scale_y: float | None,
) -> float | None:
    """Евклидово расстояние между точками в мкм с раздельными масштабами осей."""
    if scale_x is None or scale_y is None:
        return None
    dx_um = (point_b[0] - point_a[0]) * scale_x
    dy_um = (point_b[1] - point_a[1]) * scale_y
    return float(np.sqrt(dx_um * dx_um + dy_um * dy_um))


def make_row(
    parameter: str,
    value: Any = "",
    unit: str = "",
    measurement_id: str | None = None,
    **extra: Any,
) -> dict:
    """Создаёт строку результата с единообразным набором ключей."""
    row: dict[str, Any] = {
        RowKey.PARAMETER: parameter,
        RowKey.VALUE: value,
        RowKey.UNIT: unit,
    }
    if measurement_id is not None:
        row[RowKey.MEASUREMENT_ID] = measurement_id
    row.update(extra)
    return row


def manual_rows() -> list[dict]:
    """Строки группы «ручное заполнение» для основной таблицы."""
    return [make_row(name, "", unit) for name, unit in MANUAL_FIELDS]


def _format(value: float | str) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else value


def detachment_rows(
    spec: DetachmentSpec,
    *,
    width: float | None = None,
    height: float | None = None,
    area: float | None = None,
    location: str | None = None,
    bounds: tuple | None = None,
    extra: dict | None = None,
) -> list[dict]:
    """Строки измерений одной отслойки.

    Значения, которые не удалось измерить (``None``), пропускаются - это
    заменяет четыре одинаковых блока ``if ... is not None: rows.append(...)``
    для каждой отслойки.
    """
    row_extra = dict(extra or {})
    if bounds is not None:
        row_extra[RowKey.DETACHMENT_BOUNDS] = bounds

    candidates = (
        (SUFFIX_WIDTH, f"{spec.section} (ширина)", "мкм", width),
        (SUFFIX_HEIGHT, f"{spec.section} (высота)", "мкм", height),
        (SUFFIX_AREA, f"{spec.section} (площадь)", "мкм²", area),
        (SUFFIX_LOCATION, f"{spec.section} (локализация)", "", location),
    )
    return [
        make_row(
            parameter,
            _format(value),
            unit,
            measurement_id=spec.row_id(suffix),
            **row_extra,
        )
        for suffix, parameter, unit, value in candidates
        if value is not None
    ]


def drusen_rows(
    class_name: str,
    index: int,
    *,
    width: float | None = None,
    height: float | None = None,
    area: float | None = None,
    location: str | None = None,
    contour: np.ndarray | None = None,
    bounds: tuple | None = None,
) -> list[dict]:
    """Строки измерений одной друзы (нумерация с единицы)."""
    row_extra: dict[str, Any] = {}
    if contour is not None:
        row_extra[RowKey.CONTOUR] = contour
    if bounds is not None:
        row_extra[RowKey.DETACHMENT_BOUNDS] = bounds

    number = index + 1
    candidates = (
        (SUFFIX_WIDTH, "ширина", "мкм", width),
        (SUFFIX_HEIGHT, "высота", "мкм", height),
        (SUFFIX_AREA, "площадь", "мкм²", area),
        (SUFFIX_LOCATION, "локализация", "", location),
    )
    return [
        make_row(
            f"{class_name} #{number} ({label})",
            _format(value),
            unit,
            measurement_id=f"drusen_{number}_{suffix}",
            **row_extra,
        )
        for suffix, label, unit, value in candidates
        if value is not None
    ]


def group_rows(title: str, children: Iterable[dict], expanded: bool = True) -> dict:
    """Сворачиваемая группа строк для основной таблицы."""
    return {
        RowKey.PARAMETER: title,
        RowKey.VALUE: "",
        RowKey.UNIT: "",
        RowKey.IS_GROUP: True,
        RowKey.EXPANDED: expanded,
        RowKey.CHILDREN: list(children),
    }

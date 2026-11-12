"""Структуры данных осмотра: измерения, состояние зон, данные пациента.

Индексы колонок SQL-запроса собраны в :class:`ExamColumn`, поэтому
соответствие «поле -> столбец» больше не разбросано по вызову
``Measurement.from_model(model, row, 19, 20, 21, 22)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from PySide6.QtSql import QSqlQueryModel


def _value(model: QSqlQueryModel, row: int, column: int):
    return model.data(model.index(row, column))


def _as_str(model: QSqlQueryModel, row: int, column: int) -> str | None:
    value = _value(model, row, column)
    return str(value) if value else None


def _as_float(model: QSqlQueryModel, row: int, column: int) -> float | None:
    value = _value(model, row, column)
    try:
        return float(value) if value else None
    except (ValueError, TypeError):
        return None


def _as_date(model: QSqlQueryModel, row: int, column: int):
    value = _value(model, row, column)
    return value.toPyDate() if hasattr(value, 'toPyDate') else value


@dataclass
class Measurement:
    """Измерение с локализацией и размерами."""

    location: str | None = None
    width: float | None = None
    height: float | None = None
    area: float | None = None

    @classmethod
    def from_model(
        cls, model: QSqlQueryModel, row: int, loc_idx: int, w_idx: int, h_idx: int, a_idx: int
    ) -> 'Measurement':
        return cls(
            location=_as_str(model, row, loc_idx),
            width=_as_float(model, row, w_idx),
            height=_as_float(model, row, h_idx),
            area=_as_float(model, row, a_idx),
        )


@dataclass
class ZoneStatus:
    """Состояние анатомической зоны: описание и локализация дефекта."""

    condition: str | None = None
    defect_location: str | None = None

    @classmethod
    def from_model(cls, model: QSqlQueryModel, row: int, cond_idx: int, loc_idx: int) -> 'ZoneStatus':
        return cls(
            condition=_as_str(model, row, cond_idx),
            defect_location=_as_str(model, row, loc_idx),
        )


class ExamColumn:
    """Индексы колонок выборки осмотра (см. SQL-запрос в плагине БД)."""

    VISIT_DATE = 3
    DISEASE_DURATION = 4
    AREDS = 7
    REFRACTION = 8
    NEOVASCULARIZATION = 9
    CHOROIDAL_THICKNESS = 10
    FOVEAL_THICKNESS = 11
    TOTAL_VOLUME = 14
    AVERAGE_VOLUME = 15
    RPE_CONDITION = 16
    RPE_DEFECT_LOCATION = 17
    CME_LOCATION = 18
    SEROUS_PED = (19, 20, 21, 22)
    HEMORRHAGIC_PED = (23, 24, 25, 26)
    FIBROVASCULAR_PED = (27, 28, 29, 30)
    DRUSENOID_PED = (31, 32, 33, 34)
    DRUSEN = (35, 36, 37, 38)
    SUB_RPE_FLUID_AREA = 39
    SUB_RPE_FLUID_LOCATION = 40
    ELLIPSOID_ZONE = (41, 42)
    MYOID_ZONE = (43, 44)
    NSR_DETACHMENT = (45, 46, 47, 48)
    HYPERREFLECTIVE_LOCATION = 49
    HYPERREFLECTIVE_AREA = 50


@dataclass
class PatientExamData:
    """Данные осмотра: метаданные, толщины, зоны и патологии."""

    # Основные метаданные
    visit_date: date | None = None
    disease_duration: str | None = None
    tomograph_type: str | None = 'Топкон'
    areds_criteria: str | None = None
    refraction: str | None = None
    neovascularization_type: str | None = None
    bcva: str | None = None

    # Общие измерения сетчатки
    choroidal_center_thickness: float | None = None
    foveal_retinal_thickness: float | None = None
    cts_near_foveolla: float | None = None
    cts_near_fovea: float | None = None
    total_retinal_volume: float | None = None
    average_retinal_volume: float | None = None

    # Состояние РПЭ и отек
    rpe_status: ZoneStatus = field(default_factory=ZoneStatus)
    cmo_location: str | None = None

    # Отслойки пигментного эпителия
    serous_ped: Measurement = field(default_factory=Measurement)
    hemorrhagic_ped: Measurement = field(default_factory=Measurement)
    fibrovascular_ped: Measurement = field(default_factory=Measurement)
    drusenoid_ped: Measurement = field(default_factory=Measurement)

    # Друзы (усреднённое измерение)
    drusen: Measurement = field(default_factory=Measurement)

    # Жидкость под РПЭ
    sub_rpe_fluid: Measurement = field(default_factory=Measurement)

    # Состояние зон
    ellipsoid_zone: ZoneStatus = field(default_factory=ZoneStatus)
    myoid_zone: ZoneStatus = field(default_factory=ZoneStatus)

    # Прочие патологии
    nsr_detachment: Measurement = field(default_factory=Measurement)
    hyperreflective_material: Measurement = field(default_factory=Measurement)

    @classmethod
    def from_query_model(cls, model: QSqlQueryModel, row: int) -> 'PatientExamData':
        """Собирает данные осмотра из строки модели SQL-запроса."""
        return cls(
            visit_date=_as_date(model, row, ExamColumn.VISIT_DATE),
            disease_duration=_as_str(model, row, ExamColumn.DISEASE_DURATION),
            areds_criteria=_as_str(model, row, ExamColumn.AREDS),
            refraction=_as_str(model, row, ExamColumn.REFRACTION),
            neovascularization_type=_as_str(model, row, ExamColumn.NEOVASCULARIZATION),
            choroidal_center_thickness=_as_float(model, row, ExamColumn.CHOROIDAL_THICKNESS),
            foveal_retinal_thickness=_as_float(model, row, ExamColumn.FOVEAL_THICKNESS),
            total_retinal_volume=_as_float(model, row, ExamColumn.TOTAL_VOLUME),
            average_retinal_volume=_as_float(model, row, ExamColumn.AVERAGE_VOLUME),
            rpe_status=ZoneStatus.from_model(model, row, ExamColumn.RPE_CONDITION, ExamColumn.RPE_DEFECT_LOCATION),
            cmo_location=_as_str(model, row, ExamColumn.CME_LOCATION),
            serous_ped=Measurement.from_model(model, row, *ExamColumn.SEROUS_PED),
            hemorrhagic_ped=Measurement.from_model(model, row, *ExamColumn.HEMORRHAGIC_PED),
            fibrovascular_ped=Measurement.from_model(model, row, *ExamColumn.FIBROVASCULAR_PED),
            drusenoid_ped=Measurement.from_model(model, row, *ExamColumn.DRUSENOID_PED),
            drusen=Measurement.from_model(model, row, *ExamColumn.DRUSEN),
            sub_rpe_fluid=Measurement(
                area=_as_float(model, row, ExamColumn.SUB_RPE_FLUID_AREA),
                location=_as_str(model, row, ExamColumn.SUB_RPE_FLUID_LOCATION),
            ),
            ellipsoid_zone=ZoneStatus.from_model(model, row, *ExamColumn.ELLIPSOID_ZONE),
            myoid_zone=ZoneStatus.from_model(model, row, *ExamColumn.MYOID_ZONE),
            nsr_detachment=Measurement.from_model(model, row, *ExamColumn.NSR_DETACHMENT),
            hyperreflective_material=Measurement(
                location=_as_str(model, row, ExamColumn.HYPERREFLECTIVE_LOCATION),
                area=_as_float(model, row, ExamColumn.HYPERREFLECTIVE_AREA),
            ),
        )

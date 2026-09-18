from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List

from PySide6.QtSql import QSqlQueryModel


@dataclass
class Measurement:
    """Базовый класс для измерений с локализацией и размерами"""
    location: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    area: Optional[float] = None

    @classmethod
    def from_model(cls, model: QSqlQueryModel, row: int,
                 loc_idx: int, w_idx: int, h_idx: int, a_idx: int):
        return cls(
            location=model.data(model.index(row, loc_idx)),
            width=float(model.data(model.index(row, w_idx))) if model.data(model.index(row, w_idx)) else None,
            height=float(model.data(model.index(row, h_idx))) if model.data(model.index(row, h_idx)) else None,
            area=float(model.data(model.index(row, a_idx))) if model.data(model.index(row, a_idx)) else None
        )

@dataclass
class ZoneStatus:
    """Состояние анатомических зон"""
    condition: Optional[str] = None
    defect_location: Optional[str] = None

    @classmethod
    def from_model(cls, model: QSqlQueryModel, row: int,
                  cond_idx: int, loc_idx: int):
        return cls(
            condition=model.data(model.index(row, cond_idx)),
            defect_location=model.data(model.index(row, loc_idx))
        )

@dataclass
class PatientExamData:
    # Основные метаданные
    visit_date: Optional[date] = None
    disease_duration: Optional[str] = None
    tomograph_type: Optional[str] = "Топкон"  # Добавлено поле типа томографа
    disease_duration: Optional[str] = None
    refraction: Optional[str] = None
    neovascularization_type: Optional[str] = None
    bcva: Optional[str] = None  # Максимальная корригированная острота зрения

    # Общие измерения сетчатки
    choroidal_center_thickness: Optional[float] = None
    foveal_retinal_thickness: Optional[float] = None
    cts_near_foveolla: Optional[float] = None  # ЦТС возле фовеолы (внутренняя)
    cts_near_fovea: Optional[float] = None  # ЦТС возле фовеа (наружная)
    total_retinal_volume: Optional[float] = None
    average_retinal_volume: Optional[float] = None

    # Состояние РПЭ
    rpe_status: ZoneStatus = field(default_factory=ZoneStatus)
    cmo_location: Optional[str] = None

    # Типы ОПЭ
    serous_ped: Measurement = field(default_factory=Measurement)
    hemorrhagic_ped: Measurement = field(default_factory=Measurement)
    fibrovascular_ped: Measurement = field(default_factory=Measurement)
    drusenoid_ped: Measurement = field(default_factory=Measurement)

    # Друзы
    drusen: Measurement = field(default_factory=list)

    # Жидкость под РПЭ
    sub_rpe_fluid: Measurement = field(default_factory=Measurement)

    # Состояние зон
    ellipsoid_zone: ZoneStatus = field(default_factory=ZoneStatus)
    myoid_zone: ZoneStatus = field(default_factory=ZoneStatus)

    # Патологии
    nsr_detachment: Measurement = field(default_factory=Measurement)
    hyperreflective_material: Measurement = field(default_factory=Measurement)

    @classmethod
    def from_query_model(cls, model: QSqlQueryModel, row: int):
        def get_date(idx):
            val = model.data(model.index(row, idx))
            return val.toPyDate() if hasattr(val, 'toPyDate') else val

        def get_str(idx):
            val = model.data(model.index(row, idx))
            return str(val) if val else None

        def get_float(idx):
            val = model.data(model.index(row, idx))
            try:
                return float(val) if val else None
            except (ValueError, TypeError):
                return None

        return cls(
            visit_date=get_date(3),
            disease_duration=get_str(4),
            tomograph_type="Топкон",  # Установлено значение по умолчанию
            areds_criteria=get_str(7),
            refraction=get_str(8),
            neovascularization_type=get_str(9),
            choroidal_center_thickness=get_float(10),
            foveal_retinal_thickness=get_float(11),
            total_retinal_volume=get_float(14),
            average_retinal_volume=get_float(15),
            rpe_status=ZoneStatus.from_model(model, row, 16, 17),
            cmo_location=get_str(18),
            serous_ped=Measurement.from_model(model, row, 19, 20, 21, 22),
            hemorrhagic_ped=Measurement.from_model(model, row, 23, 24, 25, 26),
            fibrovascular_ped=Measurement.from_model(model, row, 27, 28, 29, 30),
            drusenoid_ped=Measurement.from_model(model, row, 31, 32, 33, 34),
            drusen=Measurement.from_model(model, row, 35, 36, 37, 38),
            sub_rpe_fluid=Measurement(
                area=get_float(39),
                location=get_str(40)
            ),
            ellipsoid_zone=ZoneStatus.from_model(model, row, 41, 42),
            myoid_zone=ZoneStatus.from_model(model, row, 43, 44),
            nsr_detachment=Measurement.from_model(model, row, 45, 46, 47, 48),
            hyperreflective_material=Measurement(
                location=get_str(49),
                area=get_float(50)
            )
        )
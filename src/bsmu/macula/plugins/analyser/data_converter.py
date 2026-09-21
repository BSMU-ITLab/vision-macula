"""Преобразование строк анализа в объект :class:`PatientExamData`.

Раньше преобразование было реализовано дважды и по-разному:

* ``TableWindow._fill_patient_exam_data`` — из строк таблицы;
* ``DataConverter.to_patient_exam_data`` — из тех же строк, но с другим
  набором полей (не заполнялись ручные поля, СРЖ и гиперрефлективный
  материал).

Теперь есть один канонический проход :meth:`PatientExamDataBuilder.build`,
которым пользуются оба потребителя. Сохранены прежние правила:

* размеры отслоек берутся из строк измерений, локализация — из строки
  «локализация»;
* для друз берётся среднее по всем найденным друзам, локализация — самая
  частая;
* локализация дефектов РПЭ хранится как координаты ``"x,y"``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np

from bsmu.macula.records.eye_info_data import Measurement, PatientExamData
from bsmu.macula.plugins.analyser.schema import (
    NEUROEPITHELIAL_DETACHMENT,
    PED_DETACHMENTS,
    MeasurementId,
    Parameter,
    RowKey,
    SUFFIX_AREA,
    SUFFIX_LOCATION,
    is_drusen_class,
    is_drusen_parameter,
)

#: Порог X для «слева» и «справа» в прежней эвристике локализации.
LEGACY_LEFT_BOUND = 300
LEGACY_RIGHT_BOUND = 700
#: Форматы даты, которые принимает ручное поле «Дата визита».
DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y")
#: Ширина разрыва по умолчанию при определении локализации дефектов зоны.
DEFAULT_ZONE_GAP = 18
#: Приблизительная площадь отслойки как площадь эллипса (как было раньше).
ELLIPSE_AREA_FACTOR = 3.14159

#: Соответствие «measurement_id толщины -> поле PatientExamData».
THICKNESS_FIELDS = {
    MeasurementId.CHOROID_THICKNESS_CENTER: "choroidal_center_thickness",
    MeasurementId.CTS_FOVEOLA: "foveal_retinal_thickness",
    MeasurementId.CTS_NEAR_FOVEOLA: "cts_near_foveolla",
    MeasurementId.CTS_NEAR_FOVEA: "cts_near_fovea",
}

#: Соответствие «префикс измерения отслойки -> поле PatientExamData».
DETACHMENT_FIELDS = {
    spec.measurement_id: spec.measurement_id for spec in PED_DETACHMENTS
}
DETACHMENT_FIELDS[NEUROEPITHELIAL_DETACHMENT.measurement_id] = "nsr_detachment"

#: Поля локализации зон: «параметр строки -> (объект, атрибут)».
ZONE_FIELDS = {
    Parameter.ELLIPSOID_CONDITION: ("ellipsoid_zone", "condition"),
    Parameter.ELLIPSOID_DEFECTS_LOCATION: ("ellipsoid_zone", "defect_location"),
    Parameter.MYOID_CONDITION: ("myoid_zone", "condition"),
    Parameter.MYOID_DEFECTS_LOCATION: ("myoid_zone", "defect_location"),
}


def parse_float(value) -> float | None:
    """Мягкий парсинг числа (``None`` вместо исключения)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class PatientExamDataBuilder:
    """Собирает ``PatientExamData`` из строк результатов анализа."""

    def __init__(self, fovea_mask: np.ndarray | None = None):
        self.fovea_mask = fovea_mask

    # --- публичный вход ---

    def build(self, rows: Iterable[dict]) -> PatientExamData:
        rows = list(rows)
        exam_data = PatientExamData()
        self._apply_manual_fields(exam_data, rows)
        self._apply_measurements(exam_data, rows)
        self._apply_drusen(exam_data, rows)
        return exam_data

    # --- ручные поля ---

    def _apply_manual_fields(self, exam_data: PatientExamData, rows: list[dict]) -> None:
        manual = {
            row.get(RowKey.PARAMETER): row.get(RowKey.VALUE, "")
            for row in rows
            if row.get(RowKey.PARAMETER)
        }

        visit_date = manual.get(Parameter.VISIT_DATE)
        if visit_date:
            exam_data.visit_date = self._parse_date(str(visit_date))
        exam_data.disease_duration = manual.get(Parameter.DISEASE_DURATION) or None
        exam_data.refraction = manual.get(Parameter.REFRACTION) or None
        exam_data.neovascularization_type = manual.get(Parameter.NEOVASCULARIZATION) or None
        exam_data.bcva = manual.get(Parameter.BCVA) or None
        exam_data.areds_criteria = manual.get(Parameter.AREDS) or None
        exam_data.cmo_location = manual.get(Parameter.CME_LOCATION) or None
        exam_data.total_retinal_volume = parse_float(manual.get(Parameter.RETINA_VOLUME))

    @staticmethod
    def _parse_date(value: str):
        for date_format in DATE_FORMATS:
            try:
                return datetime.strptime(value, date_format).date()
            except ValueError:
                continue
        return None

    # --- измерения ---

    def _apply_measurements(self, exam_data: PatientExamData, rows: list[dict]) -> None:
        for row in rows:
            measurement_id = row.get(RowKey.MEASUREMENT_ID, "")
            value_str = row.get(RowKey.VALUE, "")
            value = parse_float(value_str)

            if measurement_id in THICKNESS_FIELDS:
                if value is not None:
                    setattr(exam_data, THICKNESS_FIELDS[measurement_id], value)
                continue

            if measurement_id == MeasurementId.SBF_AREA:
                if value is not None:
                    exam_data.sub_rpe_fluid.area = value
                continue
            if measurement_id == MeasurementId.SBF_LOCATION:
                exam_data.sub_rpe_fluid.location = value_str
                continue

            if "hyperreflective_" in measurement_id:
                if "area" in measurement_id:
                    if value is not None:
                        exam_data.hyperreflective_material.area = value
                elif "location" in measurement_id:
                    exam_data.hyperreflective_material.location = value_str
                continue

            prefix = self._detachment_prefix(measurement_id)
            if prefix is not None:
                self._apply_detachment_value(exam_data, prefix, measurement_id, value, value_str)
                continue

            parameter = row.get(RowKey.PARAMETER, "")
            if parameter == Parameter.RPE_CONDITION:
                exam_data.rpe_status.condition = value_str
                self._apply_defect_coordinate(exam_data, row)
            elif parameter == Parameter.RPE_DEFECTS_LOCATION:
                self._apply_defect_coordinate(exam_data, row)
            elif parameter in ZONE_FIELDS:
                attribute, field_name = ZONE_FIELDS[parameter]
                setattr(getattr(exam_data, attribute), field_name, value_str)

    @staticmethod
    def _detachment_prefix(measurement_id: str) -> str | None:
        for spec in DETACHMENT_FIELDS:
            if measurement_id.startswith(spec + "_"):
                return spec
        return None

    @staticmethod
    def _apply_detachment_value(exam_data: PatientExamData, prefix: str, measurement_id: str,
                                value: float | None, value_str: str) -> None:
        measurement = getattr(exam_data, DETACHMENT_FIELDS[prefix])
        suffix = measurement_id.rsplit("_", 1)[-1]
        if suffix == SUFFIX_LOCATION:
            measurement.location = value_str
        elif suffix == SUFFIX_AREA:
            if value is not None:
                measurement.area = value
        elif suffix == "height":
            if value is not None:
                measurement.height = value
        elif suffix == "width":
            if value is not None:
                measurement.width = value

    @staticmethod
    def _apply_defect_coordinate(exam_data: PatientExamData, row: dict) -> None:
        coordinate = row.get(RowKey.DEFECT_COORD)
        if coordinate is not None:
            exam_data.rpe_status.defect_location = f"{coordinate[0]},{coordinate[1]}"

    # --- друзы ---

    def _apply_drusen(self, exam_data: PatientExamData, rows: list[dict]) -> None:
        widths: list[float] = []
        heights: list[float] = []
        areas: list[float] = []
        locations: list[str] = []

        for row in rows:
            parameter = row.get(RowKey.PARAMETER, "")
            if not (is_drusen_class(parameter) and is_drusen_parameter(parameter)):
                continue

            value = parse_float(row.get(RowKey.VALUE, ""))
            lowered = parameter.lower()
            if "ширина" in lowered or "width" in lowered:
                if value is not None:
                    widths.append(value)
            elif "высота" in lowered or "height" in lowered:
                if value is not None:
                    heights.append(value)
            elif "площадь" in lowered or "area" in lowered:
                if value is not None:
                    areas.append(value)
            elif "локализация" in lowered or "location" in lowered:
                value_str = row.get(RowKey.VALUE, "")
                if value_str:
                    locations.append(value_str)

        exam_data.drusen = Measurement(
            width=self._mean(widths),
            height=self._mean(heights),
            area=self._mean(areas),
            location=Counter(locations).most_common(1)[0][0] if locations else None,
        )

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    # --- совместимые вспомогательные методы ---

    def _parse_float(self, value) -> float | None:
        return parse_float(value)

    def _extract_location(self, row: dict) -> str | None:
        """Прежняя эвристика «слева/в центре/справа» по X-границам отслойки."""
        bounds = row.get(RowKey.DETACHMENT_BOUNDS)
        if not bounds:
            return None
        left_x, right_x = bounds[0], bounds[1]
        center_x = (left_x + right_x) / 2
        if center_x < LEGACY_LEFT_BOUND:
            return "слева"
        if center_x > LEGACY_RIGHT_BOUND:
            return "справа"
        return "в центре"

    def _calculate_area(self, width: float | None, height: float | None) -> float | None:
        """Приблизительная площадь как площадь эллипса."""
        if width is None or height is None:
            return None
        return (width / 2) * (height / 2) * ELLIPSE_AREA_FACTOR

    def _parse_measurement_from_value_string(self, value_str: str) -> Measurement:
        """Парсит строку вида ``w=123.45; h=67.89; area=890.12``."""
        measurement = Measurement()
        if not value_str:
            return measurement
        for part in value_str.split(";"):
            part = part.strip()
            if part.startswith("w="):
                measurement.width = parse_float(part[2:])
            elif part.startswith("h="):
                measurement.height = parse_float(part[2:])
            elif part.startswith("area="):
                measurement.area = parse_float(part[5:])
        return measurement

    def _extract_drusen_location(self, row: dict) -> str | None:
        """Прежняя эвристика локализации друзы по центру контура."""
        contour = row.get(RowKey.CONTOUR)
        if contour is None:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        if cx < LEGACY_LEFT_BOUND:
            return "слева"
        if cx > LEGACY_RIGHT_BOUND:
            return "справа"
        return "в центре"

    def get_zone_defects_location(self, contours: list[np.ndarray],
                                  min_gap_width: int = DEFAULT_ZONE_GAP) -> str | None:
        """Локализация дефектов (разрывов) эллипсоидной/миоидной зоны.

        Дефекты — горизонтальные разрывы между сегментами зоны; алгоритм тот
        же, что для дефектов РПЭ: приоритет фовеола → фовеа → макула.
        """
        from bsmu.macula.plugins.analyser.location import classify_zones, location_by_zone

        if self.fovea_mask is None or not contours or len(contours) < 2:
            return None

        _, width = self.fovea_mask.shape[:2]
        boxes = sorted((cv2.boundingRect(contour) for contour in contours), key=lambda b: b[0])

        gap_xs: list[int] = []
        for i in range(len(boxes) - 1):
            x1, _, w1, _ = boxes[i]
            x2, _, _, _ = boxes[i + 1]
            gap_left, gap_right = x1 + w1, x2
            if gap_right - gap_left >= min_gap_width:
                gap_xs.extend(range(max(0, gap_left), min(width, gap_right)))

        if not gap_xs:
            return None

        in_foveola, in_fovea, in_macula = classify_zones(
            np.concatenate([self.fovea_mask[:, x] for x in gap_xs])
        )
        if not (in_foveola or in_fovea or in_macula):
            return None
        return location_by_zone(in_foveola, in_fovea, in_macula)

    def get_drusen_location(self, contour: np.ndarray | None,
                            mask_shape: tuple) -> str | None:
        """Локализация друзы по совпадению её X-колонок с зонами фовеа."""
        from bsmu.macula.plugins.analyser.location import location_by_x_columns

        if contour is None or self.fovea_mask is None:
            return None

        drusen_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(drusen_mask, [contour], -1, 1, -1)
        _, xs = np.where(drusen_mask == 1)
        return location_by_x_columns(self.fovea_mask, xs)


class DataConverter(PatientExamDataBuilder):
    """Совместимое имя прежнего конвертера."""

    def to_patient_exam_data(self, rows: list[dict]) -> PatientExamData:
        return self.build(rows)

"""Анализ маски ОКТ: превращение сегментации в набор измерений.

Класс :class:`MaskAnalyser` — фасад анализа. Он вызывает измерительные
функции и складывает результат в строки таблицы. Раньше это был один метод
``analyze()`` на 700 строк с копипастой на каждую патологию; теперь это
набор небольших ``_append_*`` методов, по одному на группу измерений.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from bsmu.macula.plugins.analyser.data_converter import DataConverter
from bsmu.macula.plugins.analyser.geometry import (
    extract_upper_boundary,
    get_foveola_center,
    smooth_boundary,
)
from bsmu.macula.plugins.analyser.location import cme_location, location_by_check_order
from bsmu.macula.plugins.analyser.measurements.detachment import (
    measure_detachment,
    measure_neuroepithelial_detachment,
)
from bsmu.macula.plugins.analyser.measurements.drusen import measure_drusen
from bsmu.macula.plugins.analyser.measurements.path_metrics import (
    corrected_distance_um,
    distance_px,
)
from bsmu.macula.plugins.analyser.measurements.rpe import (
    analyze_rpe,
    combined_rpe_state,
    detect_rpe_defects,
    measure_rpe_thickness,
    rpe_mask,
)
from bsmu.macula.plugins.analyser.curves import (
    central_width_near,
    central_width_of_retina,
    measure_choroid_thickness_at_center,
)
from bsmu.macula.plugins.analyser.scale_calculator import ScaleCalculator
from bsmu.macula.plugins.analyser.schema import (
    CLASS_ELLIPSOID_ZONE,
    CLASS_INTRARETINAL_HYPERREFLECTIVE,
    CLASS_IRF,
    CLASS_MYOID_ZONE,
    CLASS_SUB_BRUCH_FLUID,
    CLASS_SUBRETINAL_HYPERREFLECTIVE,
    MeasurementId,
    PED_DETACHMENTS,
    NEUROEPITHELIAL_DETACHMENT,
    Parameter,
    RowKey,
    group_rows,
    is_drusen_class,
    make_row,
    manual_rows,
    detachment_rows,
    drusen_rows,
    to_um2,
)

#: Рабочий размер снимка, к которому приводятся все маски перед анализом.
ANALYSIS_SIZE = (1024, 512)  # (ширина, высота)
#: Масштаб задан эталонным L-объектом рабочего размера: 200 мкм / 24 px и
#: 200 мкм / 60 px. Автодетект L нестабилен, поэтому используется константа.
REFERENCE_SCALE_X = 200.0 / 24.0
REFERENCE_SCALE_Y = 200.0 / 60.0
#: Коэффициент сглаживания верхней границы хориоидеи.
BOUNDARY_SMOOTH_FACTOR = 30000


class MaskAnalyser:
    """Основной класс для анализа OCT-масок."""

    def __init__(self, image: np.ndarray, mask: np.ndarray,
                 mask_fovea: np.ndarray, class_configs: List[Dict]):
        self.mask = mask
        self.fovea_mask = mask_fovea
        self.class_configs = class_configs
        self.image = image

        self.scale_calc = ScaleCalculator()
        self.data_converter = DataConverter(fovea_mask=mask_fovea)

    # --- доступ к калибровке (сохранён для обратной совместимости) ---

    def _px_to_um_x(self, value_px: float) -> float | None:
        return self.scale_calc.px_to_um_x(value_px)

    def _px_to_um_y(self, value_px: float) -> float | None:
        return self.scale_calc.px_to_um_y(value_px)

    def _px_to_um(self, value_px: float) -> float | None:
        return self.scale_calc.px_to_um(value_px)

    def _calculate_distance_um(self, point1: tuple, point2: tuple) -> float | None:
        return self.scale_calc.calculate_distance_um(point1, point2)

    @property
    def scale_x(self) -> float | None:
        return self.scale_calc.scale_x

    @scale_x.setter
    def scale_x(self, value: float | None):
        self.scale_calc.scale_x = value

    @property
    def scale_y(self) -> float | None:
        return self.scale_calc.scale_y

    @scale_y.setter
    def scale_y(self, value: float | None):
        self.scale_calc.scale_y = value

    @property
    def L_contour(self) -> np.ndarray | None:
        return self.scale_calc.L_contour

    @L_contour.setter
    def L_contour(self, value: np.ndarray | None):
        self.scale_calc.L_contour = value

    @property
    def L_width_px(self) -> float | None:
        return self.scale_calc.L_width_px

    @L_width_px.setter
    def L_width_px(self, value: float | None):
        self.scale_calc.L_width_px = value

    @property
    def L_height_px(self) -> float | None:
        return self.scale_calc.L_height_px

    @L_height_px.setter
    def L_height_px(self, value: float | None):
        self.scale_calc.L_height_px = value

    # --- основной сценарий ---

    def analyze(self) -> list[dict]:
        """Прогоняет все измерения и возвращает строки для таблицы результатов."""
        rows: list[dict] = []
        rows.append(group_rows(Parameter.MANUAL_GROUP, manual_rows(), expanded=True))

        self._apply_reference_scale()

        xs, upper = extract_upper_boundary(self.mask)
        if xs is None or upper is None:
            return rows
        smooth_upper, spline = smooth_boundary(xs, upper, BOUNDARY_SMOOTH_FACTOR)

        foveola_center = get_foveola_center(self.fovea_mask)
        if foveola_center is not None:
            rows.extend(self._foveal_thickness_rows(smooth_upper, spline, foveola_center))
            rows.extend(self._choroid_thickness_rows(smooth_upper, spline, foveola_center))
        rows.extend(self._near_fovea_thickness_rows(smooth_upper, spline))
        rows.extend(self._rpe_rows(smooth_upper, spline))
        rows.extend(self._zone_rows(CLASS_ELLIPSOID_ZONE,
                                    Parameter.ELLIPSOID_CONDITION,
                                    Parameter.ELLIPSOID_DEFECTS_LOCATION,
                                    RowKey.ELLIPSOID_CONTOURS))
        rows.extend(self._zone_rows(CLASS_MYOID_ZONE,
                                    Parameter.MYOID_CONDITION,
                                    Parameter.MYOID_DEFECTS_LOCATION,
                                    RowKey.MYOID_CONTOURS))
        rows.extend(self._irf_rows())
        rows.extend(self._ped_rows(smooth_upper, spline))
        rows.extend(self._neuroepithelial_rows(smooth_upper, spline))
        rows.extend(self._sub_bruch_fluid_rows())
        rows.extend(self._hyperreflective_rows())
        rows.extend(self._drusen_rows(smooth_upper, spline))
        return rows

    def to_patient_exam_data(self, rows: list[dict]):
        """Преобразует результаты анализа в объект ``PatientExamData``."""
        return self.data_converter.to_patient_exam_data(rows)

    # --- группы измерений ---

    def _apply_reference_scale(self) -> None:
        """Фиксирует масштаб по эталонному L-объекту рабочего размера."""
        self.scale_x = REFERENCE_SCALE_X
        self.scale_y = REFERENCE_SCALE_Y

    def _foveal_thickness_rows(self, smooth_upper, spline, foveola_center) -> list[dict]:
        """Центральная толщина сетчатки в фовеоле."""
        cts = central_width_of_retina(smooth_upper, spline, foveola_center, self.mask)
        if cts is None:
            return []

        x0, y0, corrected_px, point, _ = cts
        row = make_row(
            "Центральная толщина сетчатки (фовеола)",
            measurement_id=MeasurementId.CTS_FOVEOLA,
            **{RowKey.POINTS: [(x0, y0), point], RowKey.CENTER: foveola_center},
        )
        full_um = self._calculate_distance_um((x0, y0), point)
        full_px = distance_px((x0, y0), point)
        corrected_um = corrected_distance_um(full_um, full_px, corrected_px)
        if corrected_um is not None:
            row[RowKey.VALUE] = f"{corrected_um:.2f}"
            row[RowKey.UNIT] = "мкм"
        else:
            row[RowKey.VALUE] = f"{corrected_px:.2f}"
            row[RowKey.UNIT] = "пиксели"
        return [row]

    def _choroid_thickness_rows(self, smooth_upper, spline, foveola_center) -> list[dict]:
        """Толщина хориоидеи в центре (перпендикуляр к верхней границе)."""
        result = measure_choroid_thickness_at_center(
            self.mask, smooth_upper, spline, foveola_center[0],
            self.scale_x, self.scale_y, foveola_center,
        )
        if result is None:
            return []

        upper_point, lower_point, dx_um, dy_um = result
        total_um = float((dx_um ** 2 + dy_um ** 2) ** 0.5)
        return [
            make_row(
                "Толщина хориоидеи в центре",
                f"{total_um:.2f}",
                "мкм",
                measurement_id=MeasurementId.CHOROID_THICKNESS_CENTER,
                **{RowKey.POINTS: [upper_point, lower_point]},
            )
        ]

    def _near_fovea_thickness_rows(self, smooth_upper, spline) -> list[dict]:
        """ЦТС рядом с фовеолой и рядом с фовеей (тот же алгоритм, что и в фовеоле)."""
        rows: list[dict] = []
        groups = (
            (1, "возле фовеолы", MeasurementId.CTS_NEAR_FOVEOLA),
            (2, "возле фовеа", MeasurementId.CTS_NEAR_FOVEA),
        )
        for zone, label, measure_id in groups:
            points = central_width_near(self.image, smooth_upper, spline, self.fovea_mask, self.mask, zone)
            if points is None:
                continue
            left_point, right_point = points

            cts_left = central_width_of_retina(smooth_upper, spline, left_point, self.mask)
            cts_right = central_width_of_retina(smooth_upper, spline, right_point, self.mask)
            if cts_left is None or cts_right is None:
                continue

            row = make_row(
                f"Центральная толщина сетчатки ({label})",
                measurement_id=measure_id,
                **{RowKey.PERPENDICULARS: [(left_point, cts_left[3]), (right_point, cts_right[3])]},
            )

            left_um = corrected_distance_um(
                self._calculate_distance_um((cts_left[0], cts_left[1]), cts_left[3]),
                distance_px((cts_left[0], cts_left[1]), cts_left[3]),
                cts_left[2],
            )
            right_um = corrected_distance_um(
                self._calculate_distance_um((cts_right[0], cts_right[1]), cts_right[3]),
                distance_px((cts_right[0], cts_right[1]), cts_right[3]),
                cts_right[2],
            )
            if left_um is not None and right_um is not None:
                row[RowKey.VALUE] = f"{(left_um + right_um) / 2:.2f}"
                row[RowKey.UNIT] = "мкм"
            else:
                row[RowKey.VALUE] = f"{(cts_left[2] + cts_right[2]) / 2:.2f}"
                row[RowKey.UNIT] = "пиксели"
            rows.append(row)
        return rows

    def _rpe_rows(self, smooth_upper, spline) -> list[dict]:
        """Состояние РПЭ и локализация его дефектов."""
        rpe = rpe_mask(self.mask)
        contours, _ = cv2.findContours(rpe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        skeleton_points, perpendiculars, _, std_thickness = measure_rpe_thickness(
            self.mask, self.fovea_mask, self.scale_x, self.scale_y
        )
        defects_location, gap_contours, defect_coord = detect_rpe_defects(
            self.mask, self.fovea_mask, smooth_upper, spline, min_gap_width=32
        )
        fallback_state = analyze_rpe(self.mask, self.fovea_mask, smooth_upper, spline)
        state = combined_rpe_state(rpe, gap_contours, std_thickness, fallback_state)

        extras = {
            RowKey.RPE_CONTOURS: contours,
            RowKey.GAP_CONTOURS: gap_contours,
            RowKey.RPE_PERPENDICULARS: perpendiculars,
            RowKey.SKELETON_POINTS: skeleton_points,
            RowKey.DEFECT_COORD: defect_coord,
        }
        return [
            make_row(Parameter.RPE_CONDITION, state, "", **extras),
            make_row(
                Parameter.RPE_DEFECTS_LOCATION,
                defects_location,
                "",
                **{RowKey.GAP_CONTOURS: gap_contours, RowKey.DEFECT_COORD: defect_coord},
            ),
        ]

    def _zone_rows(self, class_id: int, condition_parameter: str,
                   defects_parameter: str, contours_key: str) -> list[dict]:
        """Состояние эллипсоидной/миоидной зоны и локализация её дефектов."""
        zone = (self.mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant, gaps = self._zone_fragments(class_id)

        if zone.sum() < 50 or not significant:
            state, defects_location = "Не определяется", None
        elif gaps == 0:
            state, defects_location = "Сохранена", None
        else:
            state = "Неравномерная (фрагментация)"
            defects_location = self._get_zone_defects_location(significant)

        rows = [make_row(condition_parameter, state, "", **{contours_key: contours})]
        if defects_location is not None:
            rows.append(make_row(defects_parameter, defects_location, ""))
        return rows

    def _irf_rows(self) -> list[dict]:
        """Локализация кистозного макулярного отека (ИРЖ, класс 3)."""
        irf = (self.mask == CLASS_IRF).astype(np.uint8)
        contours, _ = cv2.findContours(irf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            make_row(
                Parameter.CME_LOCATION,
                cme_location(self.fovea_mask, irf),
                "",
                **{RowKey.IRF_CONTOURS: contours},
            )
        ]

    def _ped_rows(self, smooth_upper, spline) -> list[dict]:
        """Отслойки пигментного эпителия: ширина, высота, площадь, локализация."""
        rows: list[dict] = []
        for spec in PED_DETACHMENTS:
            width, height, area, left_x, right_x, max_point, location, segment = measure_detachment(
                self.mask, smooth_upper, spline, spec,
                self.fovea_mask, self.scale_x, self.scale_y,
            )
            if width is None or height is None:
                continue
            rows.extend(
                detachment_rows(
                    spec,
                    width=width,
                    height=height,
                    area=area,
                    location=location,
                    bounds=(left_x, right_x, max_point, segment),
                )
            )
        return rows

    def _neuroepithelial_rows(self, smooth_upper, spline) -> list[dict]:
        """Отслойка нейроэпителия (СРЖ, класс 6)."""
        width, height, area, left_x, right_x, max_point, location, segment = (
            measure_neuroepithelial_detachment(
                self.mask, smooth_upper, spline, NEUROEPITHELIAL_DETACHMENT.class_id,
                self.fovea_mask, self.scale_x, self.scale_y,
            )
        )
        if width is None or height is None:
            return []
        return detachment_rows(
            NEUROEPITHELIAL_DETACHMENT,
            width=width,
            height=height,
            area=area,
            location=location,
            bounds=(left_x, right_x, max_point, segment),
        )

    def _sub_bruch_fluid_rows(self) -> list[dict]:
        """Sub-Bruch's Fluid (класс 7): площадь и локализация."""
        sbf = (self.mask == CLASS_SUB_BRUCH_FLUID).astype(np.uint8)
        if sbf.sum() == 0:
            return []

        area_um2 = to_um2(float(sbf.sum()), self.scale_x, self.scale_y)
        rows = []
        if area_um2 is not None:
            rows.append(
                make_row("Sub-Bruch's Fluid (площадь)", f"{area_um2:.2f}", "мкм²",
                         measurement_id=MeasurementId.SBF_AREA)
            )

        ys, xs = np.where(sbf == 1)
        zone_values = self.fovea_mask[ys, xs] if len(xs) else np.array([])
        in_foveola = 1 in zone_values
        in_fovea = 2 in zone_values
        in_macula = bool(len(zone_values)) and not in_foveola and not in_fovea
        location = location_by_check_order(in_foveola, in_fovea, in_macula)
        rows.append(make_row("Sub-Bruch's Fluid (локализация)", location, "",
                             measurement_id=MeasurementId.SBF_LOCATION))
        return rows

    def _hyperreflective_rows(self) -> list[dict]:
        """Гиперрефлективный материал (классы 4 и 5): локализация и площади."""
        subretinal = (self.mask == CLASS_SUBRETINAL_HYPERREFLECTIVE).astype(np.uint8)
        intraretinal = (self.mask == CLASS_INTRARETINAL_HYPERREFLECTIVE).astype(np.uint8)
        has_subretinal = bool(subretinal.sum())
        has_intraretinal = bool(intraretinal.sum())

        if not has_subretinal and not has_intraretinal:
            location = "Отсутствует"
        elif has_subretinal and has_intraretinal:
            location = "Субретинальный + интраретинальный"
        elif has_subretinal:
            location = "Субретинальный"
        else:
            location = "Интраретинальный"

        rows = [
            make_row(Parameter.HYPERREFLECTIVE_LOCATION, location, "",
                     measurement_id=MeasurementId.HYPERREFLECTIVE_LOCATION)
        ]

        subretinal_area = to_um2(float(subretinal.sum()), self.scale_x, self.scale_y)
        intraretinal_area = to_um2(float(intraretinal.sum()), self.scale_x, self.scale_y)

        if has_subretinal and subretinal_area is not None:
            rows.append(make_row(
                "Гиперрефлективный материал субретинальный (площадь)",
                f"{subretinal_area:.2f}", "мкм²",
                measurement_id=MeasurementId.HYPERREFLECTIVE_SUBRETINAL_AREA,
            ))
        if has_intraretinal and intraretinal_area is not None:
            rows.append(make_row(
                "Гиперрефлективный материал интраретинальный (площадь)",
                f"{intraretinal_area:.2f}", "мкм²",
                measurement_id=MeasurementId.HYPERREFLECTIVE_INTRARETINAL_AREA,
            ))
        if has_subretinal and has_intraretinal and subretinal_area is not None and intraretinal_area is not None:
            rows.append(make_row(
                "Гиперрефлективный материал общая площадь",
                f"{subretinal_area + intraretinal_area:.2f}", "мкм²",
                measurement_id=MeasurementId.HYPERREFLECTIVE_TOTAL_AREA,
            ))
        return rows

    def _drusen_rows(self, smooth_upper, spline) -> list[dict]:
        """Объекты по конфигурации классов: друзы измеряются относительно хориоидеи."""
        rows: list[dict] = []
        for class_cfg in self.class_configs:
            class_id = class_cfg["id"]
            class_name = class_cfg["name"]
            if not is_drusen_class(class_name):
                continue

            for index, contour in enumerate(self._find_objects_by_class(class_id)):
                width, height, area, left_x, right_x, max_point, _, segment = measure_drusen(
                    self.mask, smooth_upper, spline, contour,
                    self.fovea_mask, self.scale_x, self.scale_y,
                )
                if width is None or height is None:
                    continue
                rows.extend(
                    drusen_rows(
                        class_name, index,
                        width=width,
                        height=height,
                        area=area,
                        location=self._get_drusen_location(contour),
                        contour=contour,
                        bounds=(left_x, right_x, max_point, segment),
                    )
                )
        return rows

    # --- вспомогательные операции над маской ---

    def _find_objects_by_class(self, class_id: int) -> list:
        """Контуры объектов заданного класса."""
        class_mask = (self.mask == class_id).astype(np.uint8)
        objects, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return objects

    def _zone_fragments(self, class_id: int, min_size: int = 18, min_gap: int = 18):
        """Значимые фрагменты зоны (EZ/MZ) и число разрывов между ними.

        Учитываются только фрагменты шириной >= ``min_size``, разрывом считается
        горизонтальный промежуток >= ``min_gap``: так мелкие куски и шум
        сегментации не дают ложную «фрагментацию».
        """
        class_mask = (self.mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, _, width, _ = cv2.boundingRect(contour)
            if width >= min_size:
                boxes.append((x, width, contour))
        boxes.sort(key=lambda box: box[0])

        gaps = 0
        for i in range(len(boxes) - 1):
            x1, w1, _ = boxes[i]
            x2, _, _ = boxes[i + 1]
            if x2 - (x1 + w1) >= min_gap:
                gaps += 1

        return [box[2] for box in boxes], gaps

    # --- делегирование в DataConverter (обратная совместимость) ---

    def _parse_float(self, value) -> float | None:
        return self.data_converter._parse_float(value)

    def _extract_location(self, row: dict) -> str | None:
        return self.data_converter._extract_location(row)

    def _calculate_area(self, width: float | None, height: float | None) -> float | None:
        return self.data_converter._calculate_area(width, height)

    def _parse_measurement_from_value_string(self, value_str: str):
        return self.data_converter._parse_measurement_from_value_string(value_str)

    def _extract_drusen_location(self, row: dict) -> str | None:
        return self.data_converter._extract_drusen_location(row)

    def _get_zone_defects_location(self, contours):
        return self.data_converter.get_zone_defects_location(contours)

    def _get_drusen_location(self, contour) -> str | None:
        return self.data_converter.get_drusen_location(contour, self.mask.shape)

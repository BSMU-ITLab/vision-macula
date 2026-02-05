"""Модуль для конвертации результатов анализа в PatientExamData."""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import cv2

from bsmu.macula.records.eye_info_data import PatientExamData, Measurement


class DataConverter:
    """Класс для преобразования результатов анализа в структуру PatientExamData."""
    
    def __init__(self, fovea_mask: np.ndarray | None = None):
        """
        Args:
            fovea_mask: Маска зон фовеа (1=фовеола, 2=фовеа, 3=макула)
        """
        self.fovea_mask = fovea_mask
    
    def to_patient_exam_data(self, rows: List[Dict]) -> PatientExamData:
        """Преобразует результаты анализа в объект PatientExamData."""
        exam_data = PatientExamData()
        
        # Словарь для временного хранения данных
        values = {}
        
        # Извлекаем значения из rows
        for row in rows:
            param = row.get("parameter", "")
            value = row.get("value", "")
            measurement_id = row.get("measurement_id")
            
            # Сохраняем по measurement_id если он есть
            if measurement_id:
                values[measurement_id] = value
            
            # Сохраняем редактируемые поля
            if param == "Стадия по AREDS":
                exam_data.areds_criteria = value if value else None
            elif param == "Тип неоваскуляризации":
                exam_data.neovascularization_type = value if value else None
            elif param == "Рефракция":
                exam_data.refraction = value if value else None
            elif param == "МКОЗ":
                exam_data.bcva = value if value else None
            elif param == "Состояние РПЭ":
                exam_data.rpe_status.condition = value if value else None
            elif param == "Дефекты РПЭ":
                exam_data.rpe_status.defect_location = value if value else None
            elif param == "Состояние эллипсоидной зоны":
                exam_data.ellipsoid_zone.condition = value if value else None
            elif param == "Локализация дефектов эллипсоидной зоны":
                exam_data.ellipsoid_zone.defect_location = value if value else None
            elif param == "Состояние миоидной зоны":
                exam_data.myoid_zone.condition = value if value else None
            elif param == "Локализация дефектов миоидной зоны":
                exam_data.myoid_zone.defect_location = value if value else None
        
        # Заполняем толщины
        self._fill_thicknesses(exam_data, values)
        
        # Заполняем отслойки РПЭ
        self._fill_detachments(exam_data, values, rows)
        
        # Заполняем друзы
        self._fill_drusen(exam_data, rows)
        
        return exam_data
    
    def _fill_thicknesses(self, exam_data: PatientExamData, values: Dict[str, str]) -> None:
        """Заполняет данные о толщинах из values."""
        if "choroid_thickness_center" in values:
            exam_data.choroidal_center_thickness = self._parse_float(values["choroid_thickness_center"])
        
        if "cts_foveola" in values:
            exam_data.foveal_retinal_thickness = self._parse_float(values["cts_foveola"])
        
        if "cts_near_foveola" in values:
            exam_data.cts_near_foveolla = self._parse_float(values["cts_near_foveola"])
        
        if "cts_near_fovea" in values:
            exam_data.cts_near_fovea = self._parse_float(values["cts_near_fovea"])
    
    def _fill_detachments(self, exam_data: PatientExamData, values: Dict[str, str], rows: List[Dict]) -> None:
        """Заполняет данные об отслойках РПЭ."""
        detachment_types = [
            ("serous_ped", "serous_ped_width", "serous_ped_height"),
            ("hemorrhagic_ped", "hemorrhagic_ped_width", "hemorrhagic_ped_height"),
            ("fibrovascular_ped", "fibrovascular_ped_width", "fibrovascular_ped_height"),
            ("drusenoid_ped", "drusenoid_ped_width", "drusenoid_ped_height"),
        ]
        
        for attr_name, width_key, height_key in detachment_types:
            if width_key in values or height_key in values:
                measurement = Measurement(
                    width=self._parse_float(values.get(width_key)),
                    height=self._parse_float(values.get(height_key)),
                )
                
                # Извлекаем локализацию и площадь
                for row in rows:
                    if row.get("measurement_id") == width_key:
                        measurement.location = self._extract_location(row)
                        measurement.area = self._calculate_area(measurement.width, measurement.height)
                        break
                
                setattr(exam_data, attr_name, measurement)
    
    def _fill_drusen(self, exam_data: PatientExamData, rows: List[Dict]) -> None:
        """Заполняет данные о друзах (берем первую найденную друзу)."""
        for row in rows:
            param = row.get("parameter", "")
            if "Drusen" in param or ("#" in param and "друз" in param.lower()):
                value_str = row.get("value", "")
                exam_data.drusen = self._parse_measurement_from_value_string(value_str)
                exam_data.drusen.location = self._extract_drusen_location(row)
                break
    
    def _parse_float(self, value) -> float | None:
        """Парсит значение в float."""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _extract_location(self, row: Dict) -> str | None:
        """Извлекает локализацию из данных отслойки."""
        detachment_bounds = row.get("detachment_bounds")
        if detachment_bounds:
            # Новый формат: (left_x, right_x, max_perp_point, choroid_segment)
            if len(detachment_bounds) == 4:
                left_x, right_x, max_perp_point, choroid_segment = detachment_bounds
            else:
                # Старый формат
                left_x, right_x, max_point = detachment_bounds
            
            # Простая логика определения локализации
            center_x = (left_x + right_x) / 2
            if center_x < 300:  # примерные границы
                return "слева"
            elif center_x > 700:
                return "справа"
            else:
                return "в центре"
        return None
    
    def _calculate_area(self, width: float | None, height: float | None) -> float | None:
        """Вычисляет площадь из ширины и высоты (приблизительно как эллипс)."""
        if width is None or height is None:
            return None
        return (width / 2) * (height / 2) * 3.14159  # Площадь эллипса
    
    def _parse_measurement_from_value_string(self, value_str: str) -> Measurement:
        """Парсит строку вида 'w=123.45; h=67.89; area=890.12' в Measurement."""
        measurement = Measurement()
        if not value_str:
            return measurement
        
        try:
            parts = value_str.split(";")
            for part in parts:
                part = part.strip()
                if part.startswith("w="):
                    measurement.width = float(part[2:])
                elif part.startswith("h="):
                    measurement.height = float(part[2:])
                elif part.startswith("area="):
                    measurement.area = float(part[5:])
        except (ValueError, IndexError):
            pass
        
        return measurement
    
    def _extract_drusen_location(self, row: Dict) -> str | None:
        """Извлекает локализацию друзы из контура."""
        contour = row.get("contour")
        if contour is None:
            return None
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if cx < 300:
                return "слева"
            elif cx > 700:
                return "справа"
            else:
                return "в центре"
        return None
    
    def get_zone_defects_location(self, contours: List[np.ndarray]) -> str | None:
        """Определяет локализацию дефектов зоны (между контурами) по маске фовеа."""
        if self.fovea_mask is None or not contours or len(contours) < 2:
            return None
        
        # Находим промежутки между контурами
        all_points = []
        for contour in contours:
            all_points.extend(contour.reshape(-1, 2))
        
        if not all_points:
            return None
        
        all_points = np.array(all_points)
        x_min = all_points[:, 0].min()
        x_max = all_points[:, 0].max()
        y_min = all_points[:, 1].min()
        y_max = all_points[:, 1].max()
        
        # Проверяем локализацию области разрывов
        in_foveola = False
        in_fovea = False
        in_macula = False
        
        for y in range(max(0, int(y_min)), min(self.fovea_mask.shape[0], int(y_max) + 1)):
            for x in range(max(0, int(x_min)), min(self.fovea_mask.shape[1], int(x_max) + 1)):
                zone = self.fovea_mask[y, x]
                if zone == 1:
                    in_foveola = True
                elif zone == 2:
                    in_fovea = True
                else:
                    in_macula = True
        
        # Определяем локализацию
        if in_foveola:
            return "0 – Фовеола"
        elif in_fovea:
            return "1 – Фовеа (без фовеолы)"
        elif in_macula:
            return "2 – Макула (без фовеолы и фовеа)"
        else:
            return "3 – Вне макулы"
    
    def get_drusen_location(self, contour: np.ndarray, mask_shape: tuple) -> str | None:
        """Определяет локализацию друзы по совпадению X-координат с зонами фовеа.
        
        Классы:
        - 2 – Фовеола + фовеа + макула (хотя бы одна общая X-координата с фовеолой)
        - 1 – Фовеа + макула (без фовеолы, но хотя бы одна общая X с фовеа)
        - 3 – Макула (без фовеа и фовеолы)
        - 0 – отсутствует
        """
        if contour is None or self.fovea_mask is None:
            return None
        
        # Создаем маску для этой друзы
        drusen_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(drusen_mask, [contour], -1, 1, -1)
        
        # Проверяем совпадение X-координат с зонами фовеа
        in_foveola = False
        in_fovea = False
        in_macula = False
        
        # Получаем уникальные X координаты друзы
        ys, xs = np.where(drusen_mask == 1)
        if len(xs) > 0:
            unique_xs_drusen = set(xs)
            
            # Получаем X координаты для каждой зоны фовеа
            h, w = self.fovea_mask.shape
            for x in unique_xs_drusen:
                if x >= w:
                    continue
                # Проверяем все Y в этом X в маске фовеа
                column = self.fovea_mask[:, x]
                if 1 in column:  # Фовеола
                    in_foveola = True
                if 2 in column:  # Фовеа
                    in_fovea = True
                if 0 in column or 3 in column:  # Макула
                    in_macula = True
                
                # Если уже нашли фовеолу, можем выйти
                if in_foveola:
                    break
        
        # Применяем алгоритм определения с приоритетом
        if in_foveola:
            return "2 – Фовеола + фовеа + макула"
        elif in_fovea:
            return "1 – Фовеа + макула (без фовеолы)"
        elif in_macula:
            return "3 – Макула (без фовеа и фовеолы)"
        else:
            return "0 – отсутствует"

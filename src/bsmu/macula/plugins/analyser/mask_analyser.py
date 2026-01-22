from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
import numpy as np
import cv2

from PySide6.QtCore import Qt
from bsmu.vision.core.plugins import Plugin
from bsmu.macula.records.eye_info_data import PatientExamData, Measurement, ZoneStatus
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindowPlugin, MainWindow
from bsmu.macula.plugins.db.SQLiteTableViewer import TableWidgetExample
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder
from bsmu.macula.plugins.analyser.analyzed_table import TableWindow
from PySide6.QtWidgets import QTableView
from bsmu.macula.plugins.analyser.analyzed_table import ObjectsTableModel
from bsmu.vision.core.visibility import Visibility

from .scale_calculator import ScaleCalculator
from .data_converter import DataConverter
from .analyzer_utils import (
    extract_upper_boundary,
    smooth_boundary,
    get_foveola_center,
    central_width_of_retina,
    central_width_near,
    perpendicular_through_foveola,
    analyze_rpe,
    detect_rpe_defects,
    measure_rpe_thickness,
    detect_and_measure_detachments,
    measure_drusen,
    measure_neuroepithelial_detachment,
    find_L_in_image,
    calculate_scale_from_L,
    calculate_scales_from_L,
    measure_choroid_thickness_at_center,
)


if TYPE_CHECKING:
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin, Mdi

class MaskAnalyserPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin'
    }

    def __init__(self, main_window_plugin: MainWindowPlugin, mdi_plugin: MdiPlugin):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin

        self._main_window: MainWindow | None = None
        self._mdi_plugin = mdi_plugin
        self._mdi: Mdi | None = None

    @property
    def main_window(self) -> MainWindow | None:
        return self._main_window

    def _enable_gui(self):
        print("Enabling GUI for MaskAnalyzerPlugin...") 
        self._main_window = self._main_window_plugin.main_window
        self._mdi = self._mdi_plugin._mdi
        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Process Mask'),
            self._process_mask
        )

    def _process_mask(self):
        layered_image_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        if layered_image_viewer_sub_window is None:
            return

        layered_image_viewer = layered_image_viewer_sub_window.layered_image_viewer
        layered_image = layered_image_viewer.data
        mask_layer = layered_image_viewer.layer_by_name('masks')
        mask_pixels = mask_layer.image_pixels
        
        # Получаем маску fovea из layered image
        mask_fovea_layer = layered_image_viewer.layer_by_name('mask-fovea')
        if mask_fovea_layer is None:
            print("Ошибка: слой 'mask-fovea' не найден")
            return
        mask_fovea_pixels = mask_fovea_layer.image_pixels
        
        image_pixels = layered_image_viewer.layer_by_name('images').image_pixels

        classes = self.config_value("classes", [])

        mask_analyser = MaskAnalyser(image_pixels, mask_pixels, mask_fovea_pixels, classes)
        # Выполняем анализ
        objects = mask_analyser.analyze()

        if not objects:
            return
        
        # Преобразуем результаты в PatientExamData
        patient_exam_data = mask_analyser.to_patient_exam_data(objects)

        # Белый цвет для всех измерений (BGR для OpenCV)
        HIGHLIGHT_COLOR = (255, 255, 255)  # белый
        
        # Hover callback для подсвечивания объектов и измерений
        def _hover_callback(row_data):
            highlight_name = 'hover-highlight'
            try:
                # remove existing highlight
                existing = layered_image.layer_by_name(highlight_name)
                if row_data is None:
                    if existing is not None:
                        layered_image.remove_layer(existing)
                    return

                # Проверяем тип измерения по ID
                measurement_id = row_data.get('measurement_id')
                
                # Случай 1: Объект с контуром (друзы)
                if 'contour' in row_data:
                    contour = row_data.get('contour')
                    if contour is None:
                        if existing is not None:
                            layered_image.remove_layer(existing)
                        return

                    # Создаем RGBA маску с прозрачным фоном
                    highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                    
                    # Рисуем контур белым на временной RGB маске
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    cv2.drawContours(temp_mask, [contour], -1, color=HIGHLIGHT_COLOR, thickness=2)
                    
                    # Если есть данные об измерениях относительно хориоидеи, визуализируем их
                    detachment_data = row_data.get('detachment_bounds')
                    if detachment_data and len(detachment_data) == 4:
                        left_x, right_x, max_perp_point, choroid_segment = detachment_data
                        
                        # Визуализируем линию хориоидеи под друзой (зеленым цветом)
                        if choroid_segment:
                            for i in range(len(choroid_segment) - 1):
                                pt1 = choroid_segment[i]
                                pt2 = choroid_segment[i + 1]
                                cv2.line(temp_mask, pt1, pt2, color=(0, 255, 0), thickness=2)
                        
                        # Визуализируем максимальный перпендикуляр (красным цветом)
                        if max_perp_point is not None and len(max_perp_point) == 4:
                            x_base, y_base, x_top, y_top = max_perp_point
                            # Округляем float координаты для визуализации
                            pt_base = (int(round(x_base)), int(round(y_base)))
                            pt_top = (int(round(x_top)), int(round(y_top)))
                            cv2.line(temp_mask, pt_base, pt_top, color=(0, 0, 255), thickness=2)
                            # Отмечаем конечные точки
                            cv2.circle(temp_mask, pt_base, 3, (255, 0, 0), -1)  # База - синий
                            cv2.circle(temp_mask, pt_top, 3, (255, 255, 0), -1)  # Верх - желтый
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    # Копируем цвета и создаем альфа-канал (непрозрачно там, где есть цвет)
                    highlight_mask_rgba[:, :, :3] = temp_mask
                    highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                    
                    from bsmu.vision.core.image import FlatImage
                    layered_image.add_layer_or_modify_pixels(
                        highlight_name,
                        highlight_mask_rgba,
                        FlatImage,
                        visibility=Visibility(True, 1.0),
                    )
                
                # Случай 2: Отслойка
                elif 'detachment_bounds' in row_data:
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    detachment_data = row_data['detachment_bounds']
                    
                    # Новый формат: (left_x, right_x, max_perp_point, choroid_segment)
                    if len(detachment_data) == 4:
                        left_x, right_x, max_perp_point, choroid_segment = detachment_data
                        
                        # Визуализируем линию хориоидеи под отслойкой (зеленым цветом)
                        if choroid_segment:
                            for i in range(len(choroid_segment) - 1):
                                pt1 = choroid_segment[i]
                                pt2 = choroid_segment[i + 1]
                                cv2.line(temp_mask, pt1, pt2, color=(0, 255, 0), thickness=2)
                        
                        # Визуализируем максимальный перпендикуляр (красным цветом)
                        if max_perp_point is not None and len(max_perp_point) == 4:
                            x_base, y_base, x_top, y_top = max_perp_point
                            # Округляем float координаты для визуализации
                            pt_base = (int(round(x_base)), int(round(y_base)))
                            pt_top = (int(round(x_top)), int(round(y_top)))
                            cv2.line(temp_mask, pt_base, pt_top, color=(0, 0, 255), thickness=2)
                            # Отмечаем конечные точки
                            cv2.circle(temp_mask, pt_base, 3, (255, 0, 0), -1)  # База - синий
                            cv2.circle(temp_mask, pt_top, 3, (255, 255, 0), -1)  # Верх - желтый
                        
                        # Обводим контур отслойки белым цветом
                        # Получаем класс отслойки из measurement_id
                        measurement_id = row_data.get('measurement_id', '')
                        detachment_class = None
                        if 'serous_ped' in measurement_id:
                            detachment_class = 2
                        elif 'hemorrhagic_ped' in measurement_id:
                            detachment_class = 16
                        elif 'fibrovascular_ped' in measurement_id:
                            detachment_class = 10
                        elif 'drusenoid_ped' in measurement_id:
                            detachment_class = 11
                        elif 'neuroepithelial_detachment' in measurement_id:
                            detachment_class = 6
                        
                        # Если определили класс, находим и обводим контур
                        if detachment_class is not None:
                            detachment_mask = (mask_analyser.mask == detachment_class).astype(np.uint8)
                            contours, _ = cv2.findContours(detachment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                cv2.drawContours(temp_mask, contours, -1, color=(255, 255, 255), thickness=2)
                    
                    else:
                        # Старый формат для совместимости
                        left_x, right_x, max_point = detachment_data
                        if max_point is not None:
                            # Рисуем прямоугольник отслойки белым
                            height_px = 30
                            cv2.rectangle(temp_mask, (left_x, max_point[1] - height_px), 
                                        (right_x, max_point[1]), color=HIGHLIGHT_COLOR, thickness=2)
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    if temp_mask.any():
                        # Создаем RGBA маску с прозрачным фоном
                        highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                        highlight_mask_rgba[:, :, :3] = temp_mask
                        highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                        
                        from bsmu.vision.core.image import FlatImage
                        layered_image.add_layer_or_modify_pixels(
                            highlight_name,
                            highlight_mask_rgba,
                            FlatImage,
                            visibility=Visibility(True, 1.0),
                        )
                
                # Случай 3: L-объект (проверяем до measurement_id)
                elif 'L_contour' in row_data:
                    L_contour = row_data.get('L_contour')
                    if L_contour is None:
                        if existing is not None:
                            layered_image.remove_layer(existing)
                        return
                    
                    print(f"Visualizing L-object, contour shape: {L_contour.shape}")
                    
                    # Рисуем L-объект желтым цветом
                    highlight_mask_rgb = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    cv2.drawContours(highlight_mask_rgb, [L_contour], -1, color=(0, 255, 255), thickness=2)
                    
                    # Получаем bounding box для отображения размеров
                    rect = cv2.minAreaRect(L_contour)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # Рисуем ограничивающий прямоугольник
                    cv2.drawContours(highlight_mask_rgb, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    # Вычисляем масштабы и отображаем размеры
                    result = calculate_scales_from_L(L_contour)
                    if result is not None:
                        scale_x, scale_y, w_px, h_px = result
                        
                        # Центр L-объекта для текста
                        M = cv2.moments(L_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Отображаем размеры
                            cv2.putText(highlight_mask_rgb, f"W: {w_px:.1f}px = 200um", 
                                        (cx - 80, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            cv2.putText(highlight_mask_rgb, f"H: {h_px:.1f}px = 200um", 
                                        (cx - 80, cy + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    if highlight_mask_rgb.any():
                        from bsmu.vision.core.image import FlatImage
                        layered_image.add_layer_or_modify_pixels(
                            highlight_name,
                            highlight_mask_rgb,
                            FlatImage,
                            visibility=Visibility(True, 1.0),
                        )
                        print("L-object layer added")
                
                # Случай 3б: РПЭ (Ретинальный пигментный эпителий)
                elif 'rpe_contours' in row_data:
                    rpe_contours = row_data.get('rpe_contours')
                    if rpe_contours is None or len(rpe_contours) == 0:
                        if existing is not None:
                            layered_image.remove_layer(existing)
                        return
                    
                    # Создаем RGBA маску с прозрачным фоном
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    
                    # Рисуем контуры РПЭ зеленым цветом
                    cv2.drawContours(temp_mask, rpe_contours, -1, color=(0, 255, 0), thickness=2)
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    # Создаем RGBA маску с прозрачным фоном
                    highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                    highlight_mask_rgba[:, :, :3] = temp_mask
                    highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                    
                    from bsmu.vision.core.image import FlatImage
                    layered_image.add_layer_or_modify_pixels(
                        highlight_name,
                        highlight_mask_rgba,
                        FlatImage,
                        visibility=Visibility(True, 1.0),
                    )
                
                # Случай 3в: Разрывы РПЭ
                elif 'gap_contours' in row_data:
                    gap_contours = row_data.get('gap_contours')
                    if gap_contours is None or len(gap_contours) == 0:
                        if existing is not None:
                            layered_image.remove_layer(existing)
                        return
                    
                    # Создаем RGBA маску с прозрачным фоном
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    
                    # Рисуем разрывы красным цветом
                    cv2.drawContours(temp_mask, gap_contours, -1, color=(0, 0, 255), thickness=2)
                    # Заполняем разрывы полупрозрачным красным
                    for gap_contour in gap_contours:
                        cv2.fillPoly(temp_mask, [gap_contour], color=(0, 0, 255))
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    # Создаем RGBA маску с прозрачным фоном
                    highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                    highlight_mask_rgba[:, :, :3] = temp_mask
                    highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 128, 0)  # Полупрозрачность для разрывов
                    
                    from bsmu.vision.core.image import FlatImage
                    layered_image.add_layer_or_modify_pixels(
                        highlight_name,
                        highlight_mask_rgba,
                        FlatImage,
                        visibility=Visibility(True, 0.7),  # Немного прозрачный
                    )
                
                # Случай 3г: Контуры ИРЖ (кистозный макулярный отек)
                elif 'irf_contours' in row_data:
                    irf_contours = row_data.get('irf_contours')
                    if irf_contours is None or len(irf_contours) == 0:
                        if existing is not None:
                            layered_image.remove_layer(existing)
                        return
                    
                    # Создаем RGBA маску с прозрачным фоном
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    
                    # Рисуем контуры ИРЖ синим цветом
                    cv2.drawContours(temp_mask, irf_contours, -1, color=(255, 128, 0), thickness=2)  # Голубой
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    # Создаем RGBA маску с прозрачным фоном
                    highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                    highlight_mask_rgba[:, :, :3] = temp_mask
                    highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                    
                    from bsmu.vision.core.image import FlatImage
                    layered_image.add_layer_or_modify_pixels(
                        highlight_name,
                        highlight_mask_rgba,
                        FlatImage,
                        visibility=Visibility(True, 1.0),
                    )
                
                # Случай 3в: Гиперрефлективный материал
                elif measurement_id and 'hyperreflective_' in measurement_id:
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    
                    # Определяем какой класс показывать
                    if 'subretinal' in measurement_id:
                        class_mask = (mask_analyser.mask == 4).astype(np.uint8)
                    elif 'intraretinal' in measurement_id:
                        class_mask = (mask_analyser.mask == 5).astype(np.uint8)
                    else:
                        class_mask = ((mask_analyser.mask == 4) | (mask_analyser.mask == 5)).astype(np.uint8)
                    
                    # Находим контуры
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Рисуем контуры желтым цветом
                    cv2.drawContours(temp_mask, contours, -1, color=(0, 255, 255), thickness=2)
                    
                    if temp_mask.any():
                        highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                        highlight_mask_rgba[:, :, :3] = temp_mask
                        highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                        
                        from bsmu.vision.core.image import FlatImage
                        layered_image.add_layer_or_modify_pixels(
                            highlight_name,
                            highlight_mask_rgba,
                            FlatImage,
                            visibility=Visibility(True, 1.0),
                        )
                    elif existing is not None:
                        layered_image.remove_layer(existing)
                
                # Случай 4: Измерение (линия, точка и т.д.)
                elif measurement_id:
                    # Создаём RGB маску для подсвечивания измерения
                    temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                    
                    # Получаем параметры из row_data для рисования
                    if 'points' in row_data:
                        # Для линий и линий перпендикуляра
                        points = row_data['points']
                        if len(points) >= 2:
                            pt1, pt2 = points[0], points[1]
                            cv2.line(temp_mask, pt1, pt2, color=HIGHLIGHT_COLOR, thickness=2)
                    
                    if 'perpendicular' in row_data:
                        # Для одного перпендикуляра (возле фовеолы/фовеа)
                        perpendicular = row_data['perpendicular']
                        if len(perpendicular) == 2:
                            upper_pt, lower_pt = perpendicular
                            cv2.line(temp_mask, upper_pt, lower_pt, color=HIGHLIGHT_COLOR, thickness=2)
                    
                    if 'perpendiculars' in row_data:
                        # Для двух перпендикуляров (старый формат для совместимости)
                        perpendiculars = row_data['perpendiculars']
                        for upper_pt, lower_pt in perpendiculars:
                            cv2.line(temp_mask, upper_pt, lower_pt, color=HIGHLIGHT_COLOR, thickness=2)
                    
                    if 'rpe_perpendiculars' in row_data:
                        # Для перпендикуляров РПЭ
                        rpe_perpendiculars = row_data['rpe_perpendiculars']
                        if rpe_perpendiculars:
                            for upper_pt, lower_pt in rpe_perpendiculars:
                                cv2.line(temp_mask, upper_pt, lower_pt, color=(255, 0, 255), thickness=1)  # Пурпурный
                        
                        # Также рисуем скелет РПЭ если есть
                        # skeleton_points = row_data.get('skeleton_points')
                        # if skeleton_points is not None and len(skeleton_points) > 0:
                        #     for point in skeleton_points:
                        #         y, x = point
                        #         if 0 <= y < temp_mask.shape[0] and 0 <= x < temp_mask.shape[1]:
                        #             cv2.circle(temp_mask, (x, y), 1, color=(0, 255, 255), thickness=-1)  # Желтый
                    
                    if 'center' in row_data:
                        # Для центра фовеолы
                        center = row_data['center']
                        cv2.circle(temp_mask, center, 6, color=HIGHLIGHT_COLOR, thickness=-1)
                    
                    # Добавляем визуализацию L-объекта
                    if mask_analyser.L_contour is not None:
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                    
                    if temp_mask.any():
                        # Создаем RGBA маску с прозрачным фоном
                        highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                        highlight_mask_rgba[:, :, :3] = temp_mask
                        highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                        
                        from bsmu.vision.core.image import FlatImage
                        layered_image.add_layer_or_modify_pixels(
                            highlight_name,
                            highlight_mask_rgba,
                            FlatImage,
                            visibility=Visibility(True, 1.0),
                        )
                    elif existing is not None:
                        layered_image.remove_layer(existing)
                
                # Случай 5: Нет специальной визуализации, но показываем L-объект если он есть
                else:
                    # Для всех остальных строк (в том числе редактируемые поля и локализации)
                    # показываем только L-объект если он есть
                    if mask_analyser.L_contour is not None:
                        temp_mask = np.zeros((*mask_pixels.shape, 3), dtype=np.uint8)
                        cv2.drawContours(temp_mask, [mask_analyser.L_contour], -1, color=(0, 255, 255), thickness=2)
                        rect = cv2.minAreaRect(mask_analyser.L_contour)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(temp_mask, [box], 0, color=(255, 255, 0), thickness=1)
                        
                        if temp_mask.any():
                            highlight_mask_rgba = np.zeros((*mask_pixels.shape, 4), dtype=np.uint8)
                            highlight_mask_rgba[:, :, :3] = temp_mask
                            highlight_mask_rgba[:, :, 3] = np.where(temp_mask.any(axis=2), 255, 0)
                            
                            from bsmu.vision.core.image import FlatImage
                            layered_image.add_layer_or_modify_pixels(
                                highlight_name,
                                highlight_mask_rgba,
                                FlatImage,
                                visibility=Visibility(True, 1.0),
                            )
                    elif existing is not None:
                        layered_image.remove_layer(existing)
                        
            except Exception as e:
                print(f"Hover highlight error: {e}")
                import traceback
                traceback.print_exc()

        self._table_window = TableWindow(objects, patient_exam_data, highlight_callback=_hover_callback)
        self._table_window.show()
        self._table_window.raise_()
        self._table_window.activateWindow()


        


    def _find_contours_by_class(self, mask: np.ndarray, class_id: int) -> list:
        
        class_mask = (mask == class_id).astype(np.uint8)

        objects, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return objects

    def _disable(self):
        self._main_window = None



class MaskAnalyser:
    """Основной класс для анализа OCT-масок."""
    
    def __init__(self, image: np.ndarray, mask: np.ndarray, mask_fovea: np.ndarray, class_configs: List[Dict]):
        self.mask = mask
        self.fovea_mask = mask_fovea
        self.class_configs = class_configs
        self.image = image
        
        # Используем ScaleCalculator для работы с масштабом
        self.scale_calc = ScaleCalculator()
        
        # Используем DataConverter для преобразования в PatientExamData
        self.data_converter = DataConverter(fovea_mask=mask_fovea)
    
    # Методы делегирования для обратной совместимости
    def _px_to_um_x(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры по X."""
        return self.scale_calc.px_to_um_x(value_px)
    
    def _px_to_um_y(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры по Y."""
        return self.scale_calc.px_to_um_y(value_px)
    
    def _px_to_um(self, value_px: float) -> float | None:
        """Конвертирует пиксели в микрометры (использует среднее из X и Y)."""
        return self.scale_calc.px_to_um(value_px)
    
    def _calculate_distance_um(self, point1: tuple, point2: tuple) -> float | None:
        """Вычисляет расстояние между двумя точками в микрометрах."""
        return self.scale_calc.calculate_distance_um(point1, point2)
    
    @property
    def scale_x(self) -> float | None:
        """Масштаб по X (мкм/пиксель)."""
        return self.scale_calc.scale_x
    
    @scale_x.setter
    def scale_x(self, value: float | None):
        self.scale_calc.scale_x = value
    
    @property
    def scale_y(self) -> float | None:
        """Масштаб по Y (мкм/пиксель)."""
        return self.scale_calc.scale_y
    
    @scale_y.setter
    def scale_y(self, value: float | None):
        self.scale_calc.scale_y = value
    
    @property
    def L_contour(self) -> np.ndarray | None:
        """Контур L-объекта для визуализации."""
        return self.scale_calc.L_contour
    
    @L_contour.setter
    def L_contour(self, value: np.ndarray | None):
        self.scale_calc.L_contour = value
    
    @property
    def L_width_px(self) -> float | None:
        """Ширина L-объекта в пикселях."""
        return self.scale_calc.L_width_px
    
    @L_width_px.setter
    def L_width_px(self, value: float | None):
        self.scale_calc.L_width_px = value
    
    @property
    def L_height_px(self) -> float | None:
        """Высота L-объекта в пикселях."""
        return self.scale_calc.L_height_px
    
    @L_height_px.setter
    def L_height_px(self, value: float | None):
        self.scale_calc.L_height_px = value

    def analyze(self) -> list[dict]:
        rows: list[dict] = []

        # --- Блок для ручного заполнения ---
        manual_fields = []
        manual_fields.append({
            "parameter": "Дата визита",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Длительность заболевания",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Стадия по AREDS",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Тип неоваскуляризации",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Рефракция",
            "value": "",
            "unit": "D",
        })
        manual_fields.append({
            "parameter": "МКОЗ",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Объем сетчатки",
            "value": "",
            "unit": "мм³",
        })
        manual_fields.append({
            "parameter": "Препарат",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Количество назначенных инъекций",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Количество выполненных инъекций",
            "value": "",
            "unit": "",
        })
        manual_fields.append({
            "parameter": "Локализация кистозного макулярного отека",
            "value": "",
            "unit": "",
        })
        
        # Создаём группу для ручного заполнения
        rows.append({
            "parameter": "Заполнение вручную",
            "value": "",
            "unit": "",
            "is_group": True,
            "expanded": True,  # По умолчанию раскрыта
            "children": manual_fields,
        })

        # --- 0. Расчёт масштаба по L-объекту (отдельно по X и Y) ---
        L_contour = find_L_in_image(self.image)
        if L_contour is not None:
            result = calculate_scales_from_L(L_contour, L_size_micrometers=200.0)
            if result is not None:
                self.scale_x, self.scale_y, L_width_px, L_height_px = result
                avg_scale = (self.scale_x + self.scale_y) / 2
                
                # Сохраняем L-контур для визуализации при наведении на измерения
                self.L_contour = L_contour
                self.L_width_px = L_width_px
                self.L_height_px = L_height_px
                
                # НЕ добавляем записи о L-объекте в таблицу
                # L-объект используется только для расчёта масштаба
                # и визуализируется при наведении на любое измерение

        # --- 1. Верхняя граница хориоидеи ---
        xs, upper = extract_upper_boundary(self.mask)
        if xs is None or upper is None:
            return rows

        smooth_upper, spline = smooth_boundary(xs, upper, smooth_factor=30000)

        # --- 2. Фовеола ---
        foveola_center = get_foveola_center(self.fovea_mask)

        # --- 3. Центральная толщина сетчатки ---
        if foveola_center is not None:
            cts = central_width_of_retina(
                smooth_upper, spline, foveola_center, self.mask
            )
            if cts is not None:
                x0, y0, dist, pt = cts
                
                row_entry = {
                    "parameter": "Центральная толщина сетчатки (фовеола)",
                    "measurement_id": "cts_foveola",
                    "points": [(x0, y0), pt],
                    "center": foveola_center,
                }
                
                # Вычисляем расстояние с учетом раздельных масштабов
                dist_um = self._calculate_distance_um((x0, y0), pt)
                if dist_um is not None:
                    row_entry["value"] = f"{dist_um:.2f}"
                    row_entry["unit"] = "мкм"
                else:
                    row_entry["value"] = f"{dist:.2f}"
                    row_entry["unit"] = "пиксели"
                    
                rows.append(row_entry)

        # --- 3б. Толщина хориоидеи в центре (перпендикуляр к верхней границе) ---
        if foveola_center is not None:
            choroid_result = measure_choroid_thickness_at_center(
                self.mask, smooth_upper, spline, foveola_center[0],
                self.scale_x, self.scale_y, foveola_center
            )
            if choroid_result is not None:
                upper_pt, lower_pt, dx_um, dy_um = choroid_result
                
                # Вычисляем общее расстояние по теореме Пифагора
                total_dist_um = (dx_um**2 + dy_um**2)**0.5
                
                # Толщина хориоидеи (общее расстояние)
                row_entry_total = {
                    "parameter": "Толщина хориоидеи в центре",
                    "measurement_id": "choroid_thickness_center",
                    "points": [upper_pt, lower_pt],
                    "value": f"{total_dist_um:.2f}",
                    "unit": "мкм",
                }
                rows.append(row_entry_total)

        # --- 4. Центральная толщина рядом с фовеолой / фовеей ---
        for ind, label, measure_id in [(1, "возле фовеолы", "cts_near_foveola"), (2, "возле фовеа", "cts_near_fovea")]:
            print(f"Checking central width near: ind={ind}, label={label}")
            pts = central_width_near(
                self.image, smooth_upper, spline,
                self.fovea_mask, self.mask, ind
            )
            print(f"Result: pts={pts}")
            if pts is None:
                print(f"Skipping {label} - no points found")
                continue

            left_pt, right_pt = pts
            
            # Строим перпендикуляры прямо из найденных точек (как для ЦТС)
            cx_l, cy_l = left_pt
            cx_r, cy_r = right_pt
            h, w = self.mask.shape
            
            RETINA_LAYERS = {4, 5, 6, 9}  # Слои сетчатки
            CHOROID_ID = 14
            
            # Для левой точки
            best_point_l = None
            if 0 <= cx_l < len(smooth_upper):
                slope_l = float(spline.derivative()(cx_l))
                nx_l = -slope_l
                ny_l = 1.0
                L_l = np.sqrt(nx_l*nx_l + ny_l*ny_l)
                nx_l /= L_l
                ny_l /= L_l
                
                # Ищем первое попадание в хориоидею или слои сетчатки
                for t in range(0, 200):
                    xx = int(cx_l + nx_l * t)
                    yy = int(cy_l + ny_l * t)
                    if xx < 0 or xx >= w or yy < 0 or yy >= h:
                        break
                    if self.mask[yy, xx] == CHOROID_ID or self.mask[yy, xx] in RETINA_LAYERS:
                        best_point_l = (xx, yy)
                        break
            
            # Для правой точки
            best_point_r = None
            if 0 <= cx_r < len(smooth_upper):
                slope_r = float(spline.derivative()(cx_r))
                nx_r = -slope_r
                ny_r = 1.0
                L_r = np.sqrt(nx_r*nx_r + ny_r*ny_r)
                nx_r /= L_r
                ny_r /= L_r
                
                # Ищем первое попадание в хориоидею или слои сетчатки
                for t in range(0, 200):
                    xx = int(cx_r + nx_r * t)
                    yy = int(cy_r + ny_r * t)
                    if xx < 0 or xx >= w or yy < 0 or yy >= h:
                        break
                    if self.mask[yy, xx] == CHOROID_ID or self.mask[yy, xx] in RETINA_LAYERS:
                        best_point_r = (xx, yy)
                        break
            
            # Если нашли точки для обеих линий
            if best_point_l is not None and best_point_r is not None:
                # Вычисляем расстояния с учетом раздельных масштабов
                dist_l_um = self._calculate_distance_um(left_pt, best_point_l)
                dist_r_um = self._calculate_distance_um(right_pt, best_point_r)
                
                row_entry = {
                    "parameter": f"Центральная толщина сетчатки ({label})",
                    "measurement_id": measure_id,
                    "perpendiculars": [(left_pt, best_point_l), (right_pt, best_point_r)],
                }
                
                if dist_l_um is not None and dist_r_um is not None:
                    # Среднее расстояние в микрометрах
                    avg_um = (dist_l_um + dist_r_um) / 2
                    row_entry["value"] = f"{avg_um:.2f}"
                    row_entry["unit"] = "мкм"
                else:
                    # Если нет масштаба, используем пиксели
                    dx_l = abs(best_point_l[0] - left_pt[0])
                    dy_l = abs(best_point_l[1] - left_pt[1])
                    dist_l_px = np.sqrt(dx_l**2 + dy_l**2)
                    
                    dx_r = abs(best_point_r[0] - right_pt[0])
                    dy_r = abs(best_point_r[1] - right_pt[1])
                    dist_r_px = np.sqrt(dx_r**2 + dy_r**2)
                    
                    avg_px = (dist_l_px + dist_r_px) / 2
                    row_entry["value"] = f"{avg_px:.2f}"
                    row_entry["unit"] = "пиксели"
                    
                rows.append(row_entry)

        # --- 5. Состояние РПЭ ---
        # Анализируем состояние РПЭ и собираем все характеристики
        rpe_state = analyze_rpe(
            self.mask, self.fovea_mask, smooth_upper, spline
        )
        
        # Определяем количество разрывов для более точной диагностики
        rpe_mask = (self.mask == 9).astype(np.uint8)
        rpe_contours, _ = cv2.findContours(rpe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Измеряем высоту (толщину) РПЭ для определения равномерности
        skeleton_points, rpe_perpendiculars, mean_thickness, std_thickness = measure_rpe_thickness(
            self.mask, self.fovea_mask, self.scale_x, self.scale_y
        )
        
        # Определяем разрывы
        _, gap_contours, defect_coord = detect_rpe_defects(
            self.mask, self.fovea_mask, smooth_upper, spline
        )
        
        # Формируем итоговое состояние РПЭ по приоритету:
        # 1. Эпителий сохранен
        # 2. Эпителий неравномерный
        # 3. Единичные разрывы
        # 4. Множественные разрывы
        # 5. Эпителий не определяется
        
        if rpe_mask.sum() == 0 or rpe_mask.sum() < 100:
            rpe_state_combined = "Эпителий не определяется"
        else:
            # Проверяем количество разрывов
            num_gaps = len(gap_contours) if gap_contours else 0
            
            if num_gaps > 3:
                rpe_state_combined = "Множественные разрывы"
            elif num_gaps > 0:
                rpe_state_combined = "Единичные разрывы"
            else:
                # Нет разрывов - проверяем равномерность по std
                if std_thickness is not None:
                    # std до 50 (мкм) - это равномерный эпителий
                    if std_thickness < 50:
                        rpe_state_combined = "Эпителий сохранен"
                    else:
                        rpe_state_combined = "Эпителий неравномерный"
                else:
                    # Если не удалось измерить высоту, используем результат analyze_rpe
                    rpe_state_combined = rpe_state if rpe_state else "Эпителий сохранен"
        
        # Определяем локализацию дефектов РПЭ (текстовую)
        rpe_defects_location, gap_contours_updated, defect_coord_updated = detect_rpe_defects(
            self.mask, self.fovea_mask, smooth_upper, spline
        )
        # Обновляем gap_contours и defect_coord если функция вернула новые
        if gap_contours_updated:
            gap_contours = gap_contours_updated
        if defect_coord_updated:
            defect_coord = defect_coord_updated
        
        # Добавляем одну строку для состояния РПЭ
        rows.append({
            "parameter": "Состояние РПЭ",
            "value": rpe_state_combined,
            "unit": "",
            "rpe_contours": rpe_contours,
            "gap_contours": gap_contours,
            "rpe_perpendiculars": rpe_perpendiculars,
            "skeleton_points": skeleton_points,
            "defect_coord": defect_coord,
        })
        
        # Добавляем строку для локализации дефектов РПЭ
        rows.append({
            "parameter": "Локализация дефектов РПЭ",
            "value": rpe_defects_location,
            "unit": "",
            "gap_contours": gap_contours,
            "defect_coord": defect_coord,
        })
        
        # --- 5в. Состояние эллипсоидной зоны (Ellipsoid Zone - класс 12) ---
        ellipsoid_mask = (self.mask == 12).astype(np.uint8)
        
        # Находим контуры эллипсоидной зоны
        ellipsoid_contours, _ = cv2.findContours(ellipsoid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Определяем состояние
        if ellipsoid_mask.sum() == 0 or ellipsoid_mask.sum() < 50:
            ellipsoid_state = "Не определяется"
            ellipsoid_defects_location = None
        else:
            num_contours = len(ellipsoid_contours)
            
            if num_contours == 1:
                # Одна непрерывная зона
                ellipsoid_state = "Сохранена"
                ellipsoid_defects_location = None
            elif num_contours == 2:
                # Зона разорвана в одном месте
                ellipsoid_state = "Не определяется локально"
                # Определяем локализацию разрыва
                ellipsoid_defects_location = self._get_zone_defects_location(ellipsoid_contours)
            elif num_contours > 2:
                # Зона разорвана во многих местах
                ellipsoid_state = "Неравномерная (фрагментация)"
                ellipsoid_defects_location = self._get_zone_defects_location(ellipsoid_contours)
            else:
                ellipsoid_state = "Не определено"
                ellipsoid_defects_location = None
        
        # Добавляем состояние эллипсоидной зоны
        rows.append({
            "parameter": "Состояние эллипсоидной зоны",
            "value": ellipsoid_state,
            "unit": "",
            "ellipsoid_contours": ellipsoid_contours,
        })
        
        # Добавляем локализацию дефектов если есть
        if ellipsoid_defects_location is not None:
            rows.append({
                "parameter": "Локализация дефектов эллипсоидной зоны",
                "value": ellipsoid_defects_location,
                "unit": "",
            })
        
        # --- 5г. Состояние миоидной зоны (Myoid Zone - класс 13) ---
        myoid_mask = (self.mask == 13).astype(np.uint8)
        
        # Находим контуры миоидной зоны
        myoid_contours, _ = cv2.findContours(myoid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Определяем состояние
        if myoid_mask.sum() == 0 or myoid_mask.sum() < 50:
            myoid_state = "Не определяется"
            myoid_defects_location = None
        else:
            num_contours = len(myoid_contours)
            
            if num_contours == 1:
                # Одна непрерывная зона
                myoid_state = "Сохранена"
                myoid_defects_location = None
            elif num_contours > 1:
                # Зона разорвана хотя бы в одном месте
                myoid_state = "Неравномерная (фрагментация)"
                myoid_defects_location = self._get_zone_defects_location(myoid_contours)
            else:
                myoid_state = "Не определено"
                myoid_defects_location = None
        
        # Добавляем состояние миоидной зоны
        rows.append({
            "parameter": "Состояние миоидной зоны",
            "value": myoid_state,
            "unit": "",
            "myoid_contours": myoid_contours,
        })
        
        # Добавляем локализацию дефектов если есть
        if myoid_defects_location is not None:
            rows.append({
                "parameter": "Локализация дефектов миоидной зоны",
                "value": myoid_defects_location,
                "unit": "",
            })
        
        # --- 5б. Локализация кистозного макулярного отека (ИРЖ - класс 3) ---
        irf_mask = (self.mask == 3).astype(np.uint8)
        
        if irf_mask.sum() == 0:
            cme_location = "0 – Отек отсутствует"
        else:
            # Проверяем локализацию ИРЖ
            h, w = self.mask.shape
            in_foveola = False
            in_fovea = False
            in_macula = False
            
            # Проверяем каждый пиксель ИРЖ
            for y in range(h):
                for x in range(w):
                    if irf_mask[y, x] > 0:
                        zone = self.fovea_mask[y, x]
                        if zone == 1:  # Фовеола
                            in_foveola = True
                        elif zone == 2:  # Фовеа
                            in_fovea = True
                        else:  # Макула
                            in_macula = True
            
            # Применяем алгоритм определения
            if in_foveola:
                cme_location = "2 – Фовеола + фовеа + макула"
            elif in_fovea:
                cme_location = "1 – Фовеа + макула (без фовеолы)"
            elif in_macula:
                cme_location = "3 – Макула (без фовеа и фовеолы)"
            else:
                cme_location = "0 – Отек отсутствует"
        
        # Находим контуры ИРЖ для визуализации
        irf_contours, _ = cv2.findContours(irf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rows.append({
            "parameter": "Локализация кистозного макулярного отека",
            "value": cme_location,
            "unit": "",
            "irf_contours": irf_contours,
        })

        # --- 6. Отслойки (PED, серозная, геморрагическая и т.д.) ---
        DETACHMENTS = {
            2: ("Серозная отслойка ПЭ (СОПЭ)", "serous_ped"),
            16: ("Геморрагическая отслойка ПЭ (ГОПЭ)", "hemorrhagic_ped"),
            10: ("Фиброваскулярная отслойка ПЭ (ФВОПЭ)", "fibrovascular_ped"),
            11: ("Друзеноидная отслойка ПЭ", "drusenoid_ped"),
        }

        for class_id, (name, measure_id) in DETACHMENTS.items():
            print(f"Checking detachment class {class_id}: {name}")
            width, height, area, left_x, right_x, max_perp_point, location, choroid_segment = detect_and_measure_detachments(
                self.mask, smooth_upper, spline, class_id, self.fovea_mask, self.scale_x, self.scale_y
            )
            print(f"Result: width={width}, height={height}, area={area}, location={location}")
            
            if width is None or height is None:
                print(f"Skipping {name} - no detachment found")
                continue
            
            # Добавляем ширину
            if width is not None:
                rows.append({
                    "parameter": f"{name} (ширина)",
                    "measurement_id": f"{measure_id}_width",
                    "value": f"{width:.2f}",
                    "unit": "мкм",
                    "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                })
            
            # Добавляем высоту
            if height is not None:
                rows.append({
                    "parameter": f"{name} (высота)",
                    "measurement_id": f"{measure_id}_height",
                    "value": f"{height:.2f}",
                    "unit": "мкм",
                    "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                })
                
            # Добавляем площадь отслойки
            if area is not None:
                rows.append({
                    "parameter": f"{name} (площадь)",
                    "measurement_id": f"{measure_id}_area",
                    "value": f"{area:.2f}",
                    "unit": "мкм²",
                    "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                })
            
            # Добавляем локализацию отслойки
            if location:
                rows.append({
                    "parameter": f"{name} (локализация)",
                    "measurement_id": f"{measure_id}_location",
                    "value": location,
                    "unit": "",
                    "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                })
                print(f"Added {name} with location: {location}")

        # --- 6а. Отслойка нейроэпителия (СРЖ) - класс 6 ---
        print("Checking neuroepithelial detachment class 6")
        neuro_width, neuro_height, neuro_area, neuro_left_x, neuro_right_x, neuro_max_perp_point, neuro_location, neuro_upper_segment = measure_neuroepithelial_detachment(
            self.mask, smooth_upper, spline, 6, self.fovea_mask, self.scale_x, self.scale_y
        )
        print(f"Neuroepithelial detachment result: width={neuro_width}, height={neuro_height}, area={neuro_area}, location={neuro_location}")
        
        if neuro_width is not None and neuro_height is not None:
            # Добавляем ширину
            if neuro_width is not None:
                rows.append({
                    "parameter": "Отслойка нейроэпителия (ширина)",
                    "measurement_id": "neuroepithelial_detachment_width",
                    "value": f"{neuro_width:.2f}",
                    "unit": "мкм",
                    "detachment_bounds": (neuro_left_x, neuro_right_x, neuro_max_perp_point, neuro_upper_segment),
                })
            
            # Добавляем высоту
            if neuro_height is not None:
                rows.append({
                    "parameter": "Отслойка нейроэпителия (высота)",
                    "measurement_id": "neuroepithelial_detachment_height",
                    "value": f"{neuro_height:.2f}",
                    "unit": "мкм",
                    "detachment_bounds": (neuro_left_x, neuro_right_x, neuro_max_perp_point, neuro_upper_segment),
                })
            
            # Добавляем площадь
            if neuro_area is not None:
                rows.append({
                    "parameter": "Отслойка нейроэпителия (площадь)",
                    "measurement_id": "neuroepithelial_detachment_area",
                    "value": f"{neuro_area:.2f}",
                    "unit": "мкм²",
                    "detachment_bounds": (neuro_left_x, neuro_right_x, neuro_max_perp_point, neuro_upper_segment),
                })
            
            # Добавляем локализацию
            if neuro_location:
                rows.append({
                    "parameter": "Отслойка нейроэпителия (локализация)",
                    "measurement_id": "neuroepithelial_detachment_location",
                    "value": neuro_location,
                    "unit": "",
                    "detachment_bounds": (neuro_left_x, neuro_right_x, neuro_max_perp_point, neuro_upper_segment),
                })
                print(f"Added neuroepithelial detachment with location: {neuro_location}")

        # --- 6б. Sub-Bruch's Fluid (SBF) - класс 7 ---
        sbf_mask = (self.mask == 7).astype(np.uint8)
        
        if sbf_mask.sum() > 0:
            # Вычисляем площадь
            sbf_area_px = sbf_mask.sum()
            if self.scale_x is not None and self.scale_y is not None:
                sbf_area_um2 = sbf_area_px * self.scale_x * self.scale_y
            else:
                sbf_area_um2 = None
            
            # Определяем локализацию
            sbf_location = None
            if self.fovea_mask is not None:
                in_foveola = False
                in_fovea = False
                in_macula = False
                
                ys, xs = np.where(sbf_mask == 1)
                for y, x in zip(ys, xs):
                    zone = self.fovea_mask[y, x]
                    if zone == 1:
                        in_foveola = True
                    elif zone == 2:
                        in_fovea = True
                    else:
                        in_macula = True
                
                # Алгоритм 0-3
                if in_foveola:
                    sbf_location = "0 – Фовеола"
                elif in_fovea:
                    sbf_location = "1 – Фовеа (без фовеолы)"
                elif in_macula:
                    sbf_location = "2 – Макула (без фовеолы и фовеа)"
                else:
                    sbf_location = "3 – Вне макулы"
            
            # Добавляем площадь
            if sbf_area_um2 is not None:
                rows.append({
                    "parameter": "Sub-Bruch's Fluid (площадь)",
                    "measurement_id": "sbf_area",
                    "value": f"{sbf_area_um2:.2f}",
                    "unit": "мкм²",
                })
            
            # Добавляем локализацию
            if sbf_location is not None:
                rows.append({
                    "parameter": "Sub-Bruch's Fluid (локализация)",
                    "measurement_id": "sbf_location",
                    "value": sbf_location,
                    "unit": "",
                })
            
            print(f"Sub-Bruch's Fluid: area={sbf_area_um2}, location={sbf_location}")

        # --- 6в. Гиперрефлективный материал - классы 4 и 5 ---
        subretinal_mask = (self.mask == 4).astype(np.uint8)  # Субретинальный
        intraretinal_mask = (self.mask == 5).astype(np.uint8)  # Интраретинальный
        
        subretinal_present = subretinal_mask.sum() > 0
        intraretinal_present = intraretinal_mask.sum() > 0
        
        print(f"=== ГИПЕРРЕФЛЕКТИВНЫЙ МАТЕРИАЛ ===")
        print(f"Субретинальный (класс 4): {subretinal_mask.sum()} пикселей, present={subretinal_present}")
        print(f"Интраретинальный (класс 5): {intraretinal_mask.sum()} пикселей, present={intraretinal_present}")
        
        # Определяем локализацию
        if not subretinal_present and not intraretinal_present:
            hyperreflective_location = "Отсутствует"
        elif subretinal_present and intraretinal_present:
            hyperreflective_location = "Субретинальный + интраретинальный"
        elif subretinal_present:
            hyperreflective_location = "Субретинальный"
        else:
            hyperreflective_location = "Интраретинальный"
        
        print(f"Локализация: {hyperreflective_location}")
        
        # Добавляем локализацию
        rows.append({
            "parameter": "Гиперрефлективный материал (локализация)",
            "measurement_id": "hyperreflective_material_location",
            "value": hyperreflective_location,
            "unit": "",
        })
        
        # Вычисляем площадь субретинального материала
        if subretinal_present:
            subretinal_area_px = subretinal_mask.sum()
            if self.scale_x is not None and self.scale_y is not None:
                subretinal_area_um2 = subretinal_area_px * self.scale_x * self.scale_y
                print(f"Субретинальный: {subretinal_area_px} пикселей = {subretinal_area_um2:.2f} мкм²")
                rows.append({
                    "parameter": "Гиперрефлективный материал субретинальный (площадь)",
                    "measurement_id": "hyperreflective_subretinal_area",
                    "value": f"{subretinal_area_um2:.2f}",
                    "unit": "мкм²",
                })
        
        # Вычисляем площадь интраретинального материала
        if intraretinal_present:
            intraretinal_area_px = intraretinal_mask.sum()
            if self.scale_x is not None and self.scale_y is not None:
                intraretinal_area_um2 = intraretinal_area_px * self.scale_x * self.scale_y
                print(f"Интраретинальный: {intraretinal_area_px} пикселей = {intraretinal_area_um2:.2f} мкм²")
                rows.append({
                    "parameter": "Гиперрефлективный материал интраретинальный (площадь)",
                    "measurement_id": "hyperreflective_intraretinal_area",
                    "value": f"{intraretinal_area_um2:.2f}",
                    "unit": "мкм²",
                })
        
        # Общая площадь (если есть оба типа)
        if subretinal_present and intraretinal_present:
            if self.scale_x is not None and self.scale_y is not None:
                total_hyperreflective_area = subretinal_area_um2 + intraretinal_area_um2
                print(f"Общая площадь: {total_hyperreflective_area:.2f} мкм²")
                rows.append({
                    "parameter": "Гиперрефлективный материал общая площадь",
                    "measurement_id": "hyperreflective_total_area",
                    "value": f"{total_hyperreflective_area:.2f}",
                    "unit": "мкм²",
                })
        
        print(f"Hyperreflective material: location={hyperreflective_location}")
        print(f"=================================\n")

        # --- 7. Объекты по сегментации (друзы и т.п.) ---
        for class_cfg in self.class_configs:
            class_id = class_cfg["id"]
            class_name = class_cfg["name"]

            contours = self._find_objects_by_class(class_id)
            
            # Специальная обработка для друз (исключаем "друзеноидную отслойку")
            is_drusen = (("Drusen" in class_name or "друз" in class_name.lower()) 
                        and "друзеноид" not in class_name.lower())

            if is_drusen and len(contours) > 0:
                # Для друз измеряем относительно сглаженной линии хориоидеи
                for i, cnt in enumerate(contours):
                    # Используем специальную функцию для друз (передаем контур конкретной друзы)
                    width, height, area, left_x, right_x, max_perp_point, location, choroid_segment = measure_drusen(
                        self.mask, smooth_upper, spline, cnt, self.fovea_mask, self.scale_x, self.scale_y
                    )
                    
                    if width is None or height is None:
                        continue
                    
                    # Определяем локализацию друзы (как для дефектов РПЭ)
                    drusen_location = self._get_drusen_location(cnt)
                    
                    rows.append({
                        "parameter": f"{class_name} #{i+1} (ширина)",
                        "measurement_id": f"drusen_{i+1}_width",
                        "value": f"{width:.2f}",
                        "unit": "мкм",
                        "contour": cnt,
                        "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                    })
                    
                    rows.append({
                        "parameter": f"{class_name} #{i+1} (высота)",
                        "measurement_id": f"drusen_{i+1}_height",
                        "value": f"{height:.2f}",
                        "unit": "мкм",
                        "contour": cnt,
                        "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                    })
                    
                    if area is not None:
                        rows.append({
                            "parameter": f"{class_name} #{i+1} (площадь)",
                            "measurement_id": f"drusen_{i+1}_area",
                            "value": f"{area:.2f}",
                            "unit": "мкм²",
                            "contour": cnt,
                            "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                        })
                    
                    if drusen_location:
                        rows.append({
                            "parameter": f"{class_name} #{i+1} (локализация)",
                            "measurement_id": f"drusen_{i+1}_location",
                            "value": drusen_location,
                            "unit": "",
                            "contour": cnt,
                            "detachment_bounds": (left_x, right_x, max_perp_point, choroid_segment),
                        })

        return rows

    def to_patient_exam_data(self, rows: list[dict]) -> PatientExamData:
        """Преобразует результаты анализа в объект PatientExamData (делегирование к DataConverter)."""
        return self.data_converter.to_patient_exam_data(rows)
    
    # Методы для обратной совместимости - делегирование к DataConverter
    def _parse_float(self, value) -> float | None:
        """Парсит значение в float."""
        return self.data_converter._parse_float(value)
    
    def _extract_location(self, row: dict) -> str | None:
        """Извлекает локализацию из данных отслойки."""
        return self.data_converter._extract_location(row)
    
    def _calculate_area(self, width: float | None, height: float | None) -> float | None:
        """Вычисляет площадь из ширины и высоты."""
        return self.data_converter._calculate_area(width, height)
    
    def _parse_measurement_from_value_string(self, value_str: str) -> Measurement:
        """Парсит строку вида 'w=123.45; h=67.89; area=890.12' в Measurement."""
        return self.data_converter._parse_measurement_from_value_string(value_str)
    
    def _extract_drusen_location(self, row: dict) -> str | None:
        """Извлекает локализацию друзы из контура."""
        return self.data_converter._extract_drusen_location(row)
    
    def _get_zone_defects_location(self, contours):
        """Определяет локализацию дефектов зоны по маске фовеа."""
        return self.data_converter.get_zone_defects_location(contours)
    
    def _get_drusen_location(self, contour) -> str | None:
        """Определяет локализацию друзы по алгоритму 0-3."""
        return self.data_converter.get_drusen_location(contour, self.mask.shape)


    def _find_objects_by_class(self, class_id: int) -> list:
        """Ищем объекты для определенного класса на маске"""
        
        class_mask = (self.mask == class_id).astype(np.uint8)
        objects, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return objects

def find_L_in_image(image: np.ndarray) -> np.ndarray | None:
    """
    Находит белый L-образный уголок на темном фоне.
    Сначала ищет самый большой контур, потом левый нижний угол внутри него.
    Возвращает контур L или None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(main_contour)

    corner_w, corner_h = max(1, int(0.07 * w)), max(1, int(0.2 * h))
    corner_x = x
    corner_y = y + h - corner_h

    corner = gray[corner_y:corner_y+corner_h, corner_x:corner_x+corner_w]

    _, corner_bin = cv2.threshold(corner, 100, 255, cv2.THRESH_BINARY)

    corner_contours, _ = cv2.findContours(corner_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not corner_contours:
        return None

    L_contour = max(corner_contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])

    L_contour += np.array([[[corner_x, corner_y]]], dtype=np.int32)

    return L_contour
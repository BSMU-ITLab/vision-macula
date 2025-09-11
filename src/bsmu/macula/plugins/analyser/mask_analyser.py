from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
import numpy as np
import cv2

from PySide6.QtCore import Qt
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindowPlugin, MainWindow
from bsmu.macula.plugins.db.SQLiteTableViewer import TableWidgetExample
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder

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
        mask_layer = layered_image_viewer.layer_by_name('masks')
        mask_pixels = mask_layer.image_pixels
        image_pixels = layered_image_viewer.layer_by_name('images').image_pixels

        classes = self.config_value("classes", [])

        mask_analyser = MaskAnalyser(image_pixels, mask_pixels, classes)

        # Выполняем анализ
        results = mask_analyser.analyze()
        print(results)


    def _find_contours_by_class(self, mask: np.ndarray, class_id: int) -> list:
        
        class_mask = (mask == class_id).astype(np.uint8)

        objects, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return objects

    def _disable(self):
        self._main_window = None



class MaskAnalyser:
    def __init__(self, image:np.ndarray, mask: np.ndarray, class_configs: List[Dict]):

        self.mask = mask
        self.class_configs = class_configs
        self.image = image

    def analyze(self) -> dict:
        results = {}

        L_shape = find_L_in_image(self.image)

        image_with_L = self.image.copy()
        if L_shape is not None:
            x, y, w, h = cv2.boundingRect(L_shape)
            xw, xh = float(200/w), (200/h)
        else:
            xw, xh = 1, 1


        for class_config in self.class_configs:
            class_id = class_config['id']
            name = class_config['name']
            attributes = class_config['attributes']

            objects = self._find_objects_by_class(class_id)
            class_results = []

            for obj in objects:
                rect = cv2.minAreaRect(obj)
                width, height = rect[1]
                width *= xw
                height *= xh
                area = cv2.contourArea(obj)
                area *= xw * xh

                object_data = {}
                if any(attr['name'] == "Width" for attr in attributes):
                    object_data["width"] = width
                if any(attr['name'] == "Height" for attr in attributes):
                    object_data["height"] = height
                if any(attr['name'] == "Area" for attr in attributes):
                    object_data["area"] = area

                class_results.append(object_data)

            results[class_id] = {
                'name': name,
                'objects': class_results
            }

        return results

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
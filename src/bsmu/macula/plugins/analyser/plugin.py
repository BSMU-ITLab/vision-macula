"""Плагин Vision Framework: пункт меню «Process Mask» и окно результатов."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from PySide6.QtWidgets import QMessageBox

from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindow, MainWindowPlugin
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewerHolder

from bsmu.macula.plugins.analyser.analyzed_table import MeasurementsSubWindow, TableWindow
from bsmu.macula.plugins.analyser.analyzer import ANALYSIS_SIZE, MaskAnalyser
from bsmu.macula.plugins.analyser.highlight import HighlightRenderer

if TYPE_CHECKING:
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi, MdiPlugin

#: Имена слоёв, которые читает анализатор.
IMAGE_LAYER_NAME = "images"
MASK_LAYER_NAME = "masks"
MASK_FOVEA_LAYER_NAME = "mask-fovea"


class MaskAnalyserPlugin(Plugin):
    """Добавляет в меню алгоритмов команду «Process Mask»."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        "main_window_plugin": "bsmu.vision.plugins.windows.main.MainWindowPlugin",
        "mdi_plugin": "bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin",
    }

    def __init__(self, main_window_plugin: MainWindowPlugin, mdi_plugin: MdiPlugin):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._main_window: MainWindow | None = None
        self._mdi: Mdi | None = None
        self._table_window: TableWindow | None = None
        self._table_sub_window: MeasurementsSubWindow | None = None

    @property
    def main_window(self) -> MainWindow | None:
        return self._main_window

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window
        self._mdi = self._mdi_plugin._mdi
        self._main_window.add_menu_action(
            AlgorithmsMenu, self.tr("Process Mask"), self._process_mask
        )

    def _disable(self) -> None:
        self._main_window = None

    # --- основной сценарий ---

    def _process_mask(self) -> None:
        prepared = self._read_layers()
        if prepared is None:
            return

        image_pixels, mask_pixels, fovea_pixels, original_size, layered_image = prepared
        analyser = MaskAnalyser(image_pixels, mask_pixels, fovea_pixels,
                                self.config_value("classes", []))
        rows = analyser.analyze()
        if not rows:
            return

        patient_exam_data = analyser.to_patient_exam_data(rows)
        renderer = HighlightRenderer(
            layered_image, mask_pixels.shape[:2], original_size, ANALYSIS_SIZE
        )
        self._show_table(rows, patient_exam_data, renderer, analyser)

    def _read_layers(self):
        """Читает слои снимка и приводит их к рабочему размеру анализа.

        Если какого-то слоя нет или он без пикселей — говорим об этом
        пользователю и анализ не начинаем.

        :return: ``(image, mask, fovea_mask, исходный_размер, layered_image)``
            либо ``None``.
        """
        sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        if sub_window is None:
            return None

        viewer = sub_window.layered_image_viewer
        layers = {}
        for layer_name in (IMAGE_LAYER_NAME, MASK_LAYER_NAME, MASK_FOVEA_LAYER_NAME):
            layer = viewer.layer_by_name(layer_name)
            if layer is None or layer.image_pixels is None:
                QMessageBox.warning(
                    self._main_window,
                    self.tr("Process Mask"),
                    self.tr(
                        'No layer with name "{}".\nAnalysis cannot be started.'
                    ).format(layer_name),
                )
                return None
            layers[layer_name] = layer

        image_pixels = layers[IMAGE_LAYER_NAME].image_pixels
        mask_pixels = layers[MASK_LAYER_NAME].image_pixels
        fovea_pixels = layers[MASK_FOVEA_LAYER_NAME].image_pixels

        source_height, source_width = mask_pixels.shape[:2]
        original_size = (source_width, source_height)
        if original_size != ANALYSIS_SIZE:
            ratio_x = ANALYSIS_SIZE[0] / float(source_width)
            ratio_y = ANALYSIS_SIZE[1] / float(source_height)
            image_interp = cv2.INTER_AREA if (ratio_x < 1 or ratio_y < 1) else cv2.INTER_LINEAR
            image_pixels = cv2.resize(image_pixels, ANALYSIS_SIZE, interpolation=image_interp)
            mask_pixels = cv2.resize(mask_pixels, ANALYSIS_SIZE, interpolation=cv2.INTER_NEAREST)
            fovea_pixels = cv2.resize(fovea_pixels, ANALYSIS_SIZE, interpolation=cv2.INTER_NEAREST)

        return image_pixels, mask_pixels, fovea_pixels, original_size, viewer.data

    def _show_table(self, rows, patient_exam_data, renderer, analyser) -> None:
        """Показывает результаты как MDI-подокно, а не отдельное окно.

        Иначе окно результатов перекрывается главным окном программы при его
        активации, и нельзя видеть снимок и измерения одновременно.
        """
        self._table_window = TableWindow(
            rows,
            patient_exam_data,
            highlight_callback=lambda row_data: renderer.render(row_data, analyser),
        )

        # Повторный запуск анализа не должен плодить копии подокна.
        if self._table_sub_window is not None:
            try:
                self._table_sub_window.close()
            except RuntimeError:
                pass  # окно уже уничтожено Qt
            self._table_sub_window = None

        self._table_sub_window = MeasurementsSubWindow(self._table_window)
        self._mdi.add_sub_window(self._table_sub_window)
        self._table_sub_window.show()

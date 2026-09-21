"""Загрузка слоя маски фовеа (зоны фовеола/фовеа/макула) из файла."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox

from bsmu.vision.core.image import FlatImage
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.core.visibility import Visibility
from bsmu.vision.plugins.windows.main import FileMenu

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.windows.main import MainWindow, MainWindowPlugin


#: Имя слоя с маской зон фовеа (1 — фовеола, 2 — фовеа, 3 — макула).
MASK_FOVEA_LAYER_NAME = "mask-fovea"


class FoveaMaskLoaderPlugin(Plugin):
    """Пункт меню «Select Fovea Mask»: загрузка маски зон фовеа из файла."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        "main_window_plugin": "bsmu.vision.plugins.windows.main.MainWindowPlugin",
        "mdi_plugin": "bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin",
        "palette_pack_settings_plugin": "bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin",
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
    ):
        super().__init__()

        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._palette_pack_settings_plugin = palette_pack_settings_plugin

        self._main_window: MainWindow | None = None

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window

        self._main_window.add_menu_action(
            FileMenu,
            self.tr("Select Fovea Mask"),
            self._select_fovea_mask_file,
        )

    def _disable(self) -> None:
        self._main_window = None

    # --- выбор и загрузка файла ---

    def _active_layered_image(self) -> LayeredImage | None:
        """LayeredImage активного окна или ``None`` (с сообщением пользователю)."""
        active_sub_window = self._mdi_plugin.mdi.activeSubWindow()
        if active_sub_window is None:
            self._warn(self.tr("Откройте изображение, чтобы загрузить маску фовеа."))
            return None

        widget = active_sub_window.widget()
        layered_image = getattr(widget, "data", None)
        if layered_image is None:
            self._warn(self.tr("Активное окно не содержит изображения."))
            return None

        return layered_image

    def _select_fovea_mask_file(self) -> None:
        # Активное окно получаем ДО диалога выбора файла, иначе фокус теряется.
        layered_image = self._active_layered_image()
        if layered_image is None:
            return

        file_name, _ = QFileDialog.getOpenFileName(
            parent=self._main_window,
            caption=self.tr("Select Fovea Mask"),
            filter=self.tr("Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"),
        )
        if not file_name:
            return

        mask_fovea_pixels = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        if mask_fovea_pixels is None:
            self._warn(self.tr("Не удалось прочитать файл маски фовеа:\n{}").format(file_name))
            return

        # Цветную маску приводим к одному каналу: класс зоны хранится в значении.
        if mask_fovea_pixels.ndim == 3:
            mask_fovea_pixels = mask_fovea_pixels[..., 0]

        image_shape = self._image_shape(layered_image)
        if image_shape is not None and mask_fovea_pixels.shape[:2] != image_shape:
            mask_fovea_pixels = cv2.resize(
                mask_fovea_pixels,
                (image_shape[1], image_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        self._add_fovea_mask_layer(layered_image, mask_fovea_pixels.astype(np.uint8))

    @staticmethod
    def _image_shape(layered_image: LayeredImage) -> tuple[int, int] | None:
        if not layered_image.layers:
            return None
        return layered_image.layers[0].image.pixels.shape[:2]

    def _add_fovea_mask_layer(self, layered_image: LayeredImage, mask_fovea: np.ndarray) -> None:
        layered_image.add_layer_or_modify_pixels(
            MASK_FOVEA_LAYER_NAME,
            mask_fovea,
            FlatImage,
            palette=self._palette_pack_settings_plugin.settings.main_palette,
            visibility=Visibility(True, 0.5),
        )

    def _warn(self, text: str) -> None:
        QMessageBox.warning(self._main_window, self.tr("Fovea Mask"), text)


__all__ = ["FoveaMaskLoaderPlugin", "MASK_FOVEA_LAYER_NAME"]

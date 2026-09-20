from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, FileMenu
from bsmu.vision.core.image import MaskDrawMode
from bsmu.macula.infervis.mdi_ensemble_segmenter import EnsembleMdiSegmenter
from PySide6.QtWidgets import QFileDialog

if TYPE_CHECKING:
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from bsmu.macula.plugins.ensemble_segmenter import BinaryEnsemblePlugin


class EnsembleSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'ensemble_segmenter_plugin': 'bsmu.macula.plugins.ensemble_segmenter.BinaryEnsemblePlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            ensemble_segmenter_plugin: BinaryEnsemblePlugin,
    ):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._ensemble_segmenter_plugin = ensemble_segmenter_plugin

        self._ensemble_segmenter_gui: BinaryEnsemblePlugin | None = None
        self._main_window: MainWindow | None = None
        self._fovea_mask_path: str | None = None

    @property
    def ensemble_segmenter_gui(self) -> BinaryEnsemblePlugin | None:
        return self._ensemble_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window
        mdi = self._mdi_plugin.mdi

        self._ensemble_segmenter_gui = EnsembleMdiSegmenter(
            self._ensemble_segmenter_plugin.binary_segmenter,
            mdi,
        )

        self._main_window.add_menu_action(
            FileMenu,
            self.tr('Select Fovea Mask'),
            self._select_fovea_mask_file,
        )

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Ensemble Segmentation'),
            partial(
                self._ensemble_segmenter_gui.segment_async,
                mask_layer_name='masks',
                mask_draw_mode=MaskDrawMode.OVERLAY_FOREGROUND,
            ),
        )

    def _select_fovea_mask_file(self):
        """Открывает диалог выбора файла маски fovea"""
        # Получаем активное окно MDI ДО открытия диалога (чтобы не потерять фокус)
        mdi = self._mdi_plugin.mdi
        active_sub_window = mdi.activeSubWindow()
        if active_sub_window is None:
            print("Ошибка: нет активного окна")
            return
        
        # Получаем widget и данные напрямую
        widget = active_sub_window.widget()
        if widget is None or not hasattr(widget, 'data'):
            print("Ошибка: активное окно не содержит данных")
            return
        
        layered_image = widget.data
        if layered_image is None:
            print("Ошибка: данные layered image не загружены")
            return
        
        # Теперь открываем диалог выбора файла
        file_name, _ = QFileDialog.getOpenFileName(
            parent=self._main_window,
            caption='Select Fovea Mask',
            filter='Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)'
        )
        
        if file_name:
            self._fovea_mask_path = file_name
            self._ensemble_segmenter_gui.set_fovea_mask_path(file_name)
            
            # Инициализируем слой (используем сохраненную ссылку на layered_image)
            self._ensemble_segmenter_gui.init_fovea_mask_layer(layered_image)
            
            # Загружаем маску fovea из выбранного файла
            self._ensemble_segmenter_gui.load_and_add_fovea_mask_layer_from_active_image()
            print(f"Fovea mask path selected: {file_name}")

    def _disable(self):
        self._ensemble_segmenter_gui = None
        self._main_window = None
        self._fovea_mask_path = None

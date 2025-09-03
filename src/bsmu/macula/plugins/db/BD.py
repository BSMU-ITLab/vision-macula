from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu, MainWindowPlugin, MainWindow

from bsmu.macula.plugins.db.SQLiteTableViewer import TableWidgetExample
from bsmu.macula.plugins.ensemble_segmenter import BinaryEnsemblePlugin

if TYPE_CHECKING:
    pass
class BD(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin'
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin
    ):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin

        self._ensemble_segmenter_gui: BinaryEnsemblePlugin | None = None
        self._main_window: MainWindow | None = None

    @property
    def ensemble_segmenter_gui(self) -> BinaryEnsemblePlugin | None:
        return self._ensemble_segmenter_gui

    def _enable_gui(self):
        
        self._main_window = self._main_window_plugin.main_window

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('BD'),
            self._re
        )
    def _re(self):
        self.window = TableWidgetExample()
        self.window.setWindowModality(Qt.ApplicationModal)
        self.window.show()

    def _disable(self):
        self._ensemble_segmenter_gui = None
        self._main_window = None
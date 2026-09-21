from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu
from bsmu.vision.core.image import MaskDrawMode
from bsmu.macula.infervis.mdi_ensemble_segmenter import EnsembleMdiSegmenter

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
            AlgorithmsMenu,
            self.tr('Ensemble Segmentation'),
            partial(
                self._ensemble_segmenter_gui.segment_async,
                mask_layer_name='masks',
                mask_draw_mode=MaskDrawMode.OVERLAY_FOREGROUND,
            ),
        )

    def _disable(self):
        self._ensemble_segmenter_gui = None
        self._main_window = None

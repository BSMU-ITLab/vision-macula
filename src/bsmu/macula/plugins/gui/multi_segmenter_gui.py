from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu
from bsmu.vision.core.image import MaskDrawMode
from bsmu.macula.infervis.mdi_multi_segmenter import MultiMdiSegmenter

if TYPE_CHECKING:
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.windows.main import MainWindowPlugin, MainWindow
    from bsmu.macula.plugins.multiclass_segmenter import MultiSegmPlugin


class MultiSegmenterGuiPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'multi_segmenter_plugin': 'bsmu.macula.plugins.multiclass_segmenter.MultiSegmPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            multi_segmenter_plugin:MultiSegmPlugin,
    ):
        
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._multi_segmenter_plugin = multi_segmenter_plugin
        
        self._multi_segmenter_gui: MultiSegmPlugin | None = None
        self._main_window: MainWindow | None = None

    @property
    def multi_segmenter_gui(self) -> MultiSegmPlugin | None:
        return self._multi_segmenter_gui

    def _enable_gui(self):
        self._main_window = self._main_window_plugin.main_window
        mdi = self._mdi_plugin.mdi

        self._multi_segmenter_gui = MultiMdiSegmenter(
            self._multi_segmenter_plugin.multi_segmenter,
            mdi,
        )

        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Multi Segmentation'),
            partial(
                self._multi_segmenter_gui.segment_async,
                mask_layer_name='masks',
                mask_draw_mode=MaskDrawMode.OVERLAY_FOREGROUND,
            ),
        )

    def _disable(self):
        self._multi_segmenter_gui = None
        self._main_window = None

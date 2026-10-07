"""Algorithms menu item for retinal keypoint prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.vision.plugins.windows.main import AlgorithmsMenu

from bsmu.macula.infervis.mdi_keypoint_predictor import MdiKeypointPredictor

if TYPE_CHECKING:
    from bsmu.vision.plugins.doc_interfaces.mdi import MdiPlugin
    from bsmu.vision.plugins.windows.main import MainWindow, MainWindowPlugin
    from bsmu.macula.plugins.keypoint_predictor import KeypointPredictorPlugin


class KeypointPredictorGuiPlugin(Plugin):
    """Add keypoint prediction to the Algorithms menu."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'main_window_plugin': 'bsmu.vision.plugins.windows.main.MainWindowPlugin',
        'mdi_plugin': 'bsmu.vision.plugins.doc_interfaces.mdi.MdiPlugin',
        'keypoint_predictor_plugin': 'bsmu.macula.plugins.keypoint_predictor.KeypointPredictorPlugin',
    }

    def __init__(
            self,
            main_window_plugin: MainWindowPlugin,
            mdi_plugin: MdiPlugin,
            keypoint_predictor_plugin: KeypointPredictorPlugin,
    ):
        super().__init__()
        self._main_window_plugin = main_window_plugin
        self._mdi_plugin = mdi_plugin
        self._keypoint_predictor_plugin = keypoint_predictor_plugin
        self._mdi_predictor: MdiKeypointPredictor | None = None
        self._main_window: MainWindow | None = None

    def _enable_gui(self) -> None:
        self._main_window = self._main_window_plugin.main_window
        self._mdi_predictor = MdiKeypointPredictor(
            self._keypoint_predictor_plugin.predictor,
            self._mdi_plugin.mdi,
        )
        self._main_window.add_menu_action(
            AlgorithmsMenu,
            self.tr('Keypoint Prediction'),
            self._mdi_predictor.predict_async,
        )

    def _disable(self) -> None:
        self._mdi_predictor = None
        self._main_window = None

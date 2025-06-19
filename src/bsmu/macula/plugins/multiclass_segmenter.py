from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.macula.inference.multi_segmenter import MultiSegmenter
from bsmu.macula.inference.inferecer import ImageModelParams

if TYPE_CHECKING:
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.storages.task import TaskStoragePlugin


class MultiSegmPlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    MODEL_DIR_NAME = 'dnn-models'

    def __init__(
            self,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()

        self._palette_pack_settings_plugin = palette_pack_settings_plugin
        self._task_storage_plugin = task_storage_plugin

        self._multi_segmenter: MultiSegmenter | None = None

    @property
    def multi_segmenter(self) -> MultiSegmenter | None:
        return self._multi_segmenter

    def _enable(self):
        model_params = ImageModelParams.from_config(
            self.config_value('multi_segmenter_model'), self.data_path(self.MODEL_DIR_NAME))

        main_palette = self._palette_pack_settings_plugin.settings.main_palette
        task_storage = self._task_storage_plugin.task_storage
        num_classes = self.config_value('multi_segmenter_model')["numclasses"]
        self._multi_segmenter = MultiSegmenter(
            model_params, main_palette, task_storage, num_classes=num_classes)

    def _disable(self):
        self._multi_segmenter = None
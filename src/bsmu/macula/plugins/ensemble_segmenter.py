from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin
from bsmu.macula.inference.enseble import EnsembleSegmenter
from bsmu.macula.inference.inferecer import EnsembleImageModelParams

if TYPE_CHECKING:
    from bsmu.vision.plugins.palette.settings import PalettePackSettingsPlugin
    from bsmu.vision.plugins.storages.task import TaskStoragePlugin


class BinaryEnsemblePlugin(Plugin):
    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'palette_pack_settings_plugin': 'bsmu.vision.plugins.palette.settings.PalettePackSettingsPlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    _DNN_MODELS_DIR_NAME = 'dnn-models'
    _DATA_DIRS = (_DNN_MODELS_DIR_NAME,)

    def __init__(
            self,
            palette_pack_settings_plugin: PalettePackSettingsPlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()

        self._palette_pack_settings_plugin = palette_pack_settings_plugin
        self._task_storage_plugin = task_storage_plugin

        self._binary_segmenter: EnsembleSegmenter | None = None

    @property
    def binary_segmenter(self) -> EnsembleSegmenter | None:
        return self._binary_segmenter

    def _enable(self):
        model_dir = self.data_path(self._DNN_MODELS_DIR_NAME)
        ensemble_model_params = EnsembleImageModelParams.from_config(
            self.config_value('binary_ensemble_models'), model_dir)

        missing_models = []
        if not ensemble_model_params.class_models:
            missing_models.append('class_models (no class models configured)')
        else:
            missing_models.extend(
                model_name
                for model_name in ensemble_model_params.class_models.values()
                if not model_name or not (model_dir / model_name).is_file()
            )

        if not ensemble_model_params.boundary_model:
            missing_models.append('boundary_model (not configured)')
        elif not (model_dir / ensemble_model_params.boundary_model).is_file():
            missing_models.append(ensemble_model_params.boundary_model)

        class_ids = set(ensemble_model_params.class_models)
        if {1, 4}.issubset(class_ids) and not ensemble_model_params.ped_model:
            missing_models.append('ped_model (required for classes 1 and 4)')
        if {3, 6}.issubset(class_ids) and not ensemble_model_params.three_and_six_model:
            missing_models.append('three_and_six_model (required for classes 3 and 6)')

        for model_name in (
                ensemble_model_params.ped_model,
                ensemble_model_params.three_and_six_model):
            if model_name and not (model_dir / model_name).is_file():
                missing_models.append(model_name)

        if missing_models:
            logging.warning(
                'Ensemble segmentation is unavailable; model files are missing: %s',
                ', '.join(sorted(set(missing_models))),
            )
            self._binary_segmenter = None
            return

        main_palette = self._palette_pack_settings_plugin.settings.main_palette
        task_storage = self._task_storage_plugin.task_storage
        self._binary_segmenter = EnsembleSegmenter(
            ensemble_model_params, main_palette, task_storage)

    def _disable(self):
        self._binary_segmenter = None

"""Plugin that loads the retinal keypoint ONNX model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bsmu.vision.core.plugins import Plugin

from bsmu.macula.inference.keypoint import KeypointModelParams, KeypointPredictor

if TYPE_CHECKING:
    from bsmu.vision.plugins.storages.task import TaskStoragePlugin
    from bsmu.macula.plugins.ensemble_segmenter import BinaryEnsemblePlugin


class KeypointPredictorPlugin(Plugin):
    """Own the keypoint model and its background-mask dependency."""

    _DEFAULT_DEPENDENCY_PLUGIN_FULL_NAME_BY_KEY = {
        'ensemble_segmenter_plugin': 'bsmu.macula.plugins.ensemble_segmenter.BinaryEnsemblePlugin',
        'task_storage_plugin': 'bsmu.vision.plugins.storages.task.TaskStoragePlugin',
    }

    _DNN_MODELS_DIR_NAME = 'dnn-models'
    _DATA_DIRS = (_DNN_MODELS_DIR_NAME,)

    def __init__(
            self,
            ensemble_segmenter_plugin: BinaryEnsemblePlugin,
            task_storage_plugin: TaskStoragePlugin,
    ):
        super().__init__()
        self._ensemble_segmenter_plugin = ensemble_segmenter_plugin
        self._task_storage_plugin = task_storage_plugin
        self._predictor: KeypointPredictor | None = None

    @property
    def predictor(self) -> KeypointPredictor | None:
        return self._predictor

    def _enable(self) -> None:
        config = self.config_value('keypoint_model')
        model_params = KeypointModelParams(
            input_height=config.get('input_height', 256),
            input_width=config.get('input_width', 512),
            longest_max_size=config.get('longest_max_size', 512),
            normalize_mean=config.get('normalize_mean', 0.5),
            normalize_std=config.get('normalize_std', 0.5),
            normalize_max_pixel_value=config.get('normalize_max_pixel_value', 255.0),
            keypoint_radius=config.get('keypoint_radius', 3),
        )
        model_path = self.data_path(self._DNN_MODELS_DIR_NAME) / config.get(
            'file_name', 'Keypoints.onnx')
        self._predictor = KeypointPredictor(
            model_path,
            model_params,
            self._ensemble_segmenter_plugin.binary_segmenter,
            self._task_storage_plugin.task_storage,
        )

    def _disable(self) -> None:
        self._predictor = None

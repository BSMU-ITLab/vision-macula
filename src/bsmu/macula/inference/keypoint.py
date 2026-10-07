"""ONNX inference for retinal fovea keypoint localization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject

from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.task import DnnTask

from bsmu.macula.inference.enseble import EnsembleSegmenter


@dataclass(frozen=True)
class KeypointModelParams:
    """Model and preprocessing settings recorded with the ClearML task."""

    input_height: int = 256
    input_width: int = 512
    longest_max_size: int = 512
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    normalize_max_pixel_value: float = 255.0
    keypoint_radius: int = 3


class KeypointPredictor(QObject):
    """Predict five X coordinates and estimate their Y positions from a mask."""

    def __init__(
            self,
            model_path: Path,
            model_params: KeypointModelParams,
            segmenter: EnsembleSegmenter,
            task_storage=None,
    ):
        super().__init__()

        import onnxruntime as ort

        self._model_params = model_params
        self._segmenter = segmenter
        self._task_storage = task_storage
        self._session = ort.InferenceSession(
            str(model_path), providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict_async(
            self,
            image: np.ndarray,
            background_mask: np.ndarray | None,
            on_finished: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        task = KeypointPredictionTask(
            image,
            background_mask,
            self,
            name='Retinal Keypoint Prediction',
        )
        task.on_finished = on_finished
        if self._task_storage is not None:
            self._task_storage.add_item(task)
        ThreadPool.run_async_task(task)

    def predict(
            self,
            image: np.ndarray,
            background_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Return a class-indexed overlay with fovea and foveola points."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if background_mask is None:
            background_mask = self._segmenter.predict_background_mask(image)
        else:
            background_mask = self._retina_mask_from_segmentation(background_mask)
            if not np.any(background_mask):
                background_mask = self._segmenter.predict_background_mask(image)

        if background_mask.shape != image.shape[:2]:
            background_mask = cv2.resize(
                background_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        tensor, transform = self._preprocess(image)
        logits = self._session.run([self._output_name], {self._input_name: tensor})[0]
        normalized_x = 1.0 / (1.0 + np.exp(-np.clip(logits.reshape(-1), -80.0, 80.0)))
        transformed_x = normalized_x * max(self._model_params.input_width - 1, 1)
        original_x = self._reverse_x_transform(transformed_x, transform)

        point_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for keypoint_index, x in enumerate(original_x[:4]):
            x_pixel = int(np.clip(np.rint(x), 0, image.shape[1] - 1))
            y_pixel = self._top_mask_pixel(background_mask, x_pixel)
            if y_pixel is not None:
                cv2.circle(
                    point_mask,
                    (x_pixel, y_pixel),
                    self._model_params.keypoint_radius,
                    1 if keypoint_index < 2 else 2,
                    thickness=-1,
                )

        return point_mask

    @staticmethod
    def _retina_mask_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
        """Convert the ensemble's multiclass mask to a retinal foreground mask."""
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        if np.any(segmentation == 255):
            return segmentation != 255
        return segmentation != 0

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
        """Apply the task's LongestMaxSize, pad, resize, and Normalize transforms."""
        params = self._model_params
        original_height, original_width = image.shape[:2]
        scale = params.longest_max_size / max(original_height, original_width)
        resized_width = min(params.longest_max_size, max(1, int(round(original_width * scale))))
        resized_height = min(params.longest_max_size, max(1, int(round(original_height * scale))))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_top = max(0, (params.input_height - resized_height) // 2)
        pad_left = max(0, (params.input_width - resized_width) // 2)
        pad_bottom = max(0, params.input_height - resized_height - pad_top)
        pad_right = max(0, params.input_width - resized_width - pad_left)
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
        padded_width = image.shape[1]
        image = cv2.resize(
            image,
            (params.input_width, params.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(np.float32) / params.normalize_max_pixel_value
        image = (image - params.normalize_mean) / params.normalize_std
        tensor = image[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)
        transform = {
            'original_width': original_width,
            'resized_width': resized_width,
            'padded_width': padded_width,
            'input_width': params.input_width,
            'pad_left': pad_left,
        }
        return tensor, transform

    @staticmethod
    def _reverse_x_transform(
            transformed_x: np.ndarray,
            transform: dict[str, int | float],
    ) -> np.ndarray:
        """Map keypoint X coordinates back through the task's resize and padding."""
        resized_to_padded = transform['padded_width'] / transform['input_width']
        x_before_padding = transformed_x * resized_to_padded - transform['pad_left']
        scale = transform['resized_width'] / transform['original_width']
        return x_before_padding / scale

    @staticmethod
    def _top_mask_pixel(mask: np.ndarray, x: int) -> int | None:
        """Find the retinal surface at X, using the nearest valid column if needed."""
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            return int(ys[0])

        valid_columns = np.flatnonzero(mask.any(axis=0))
        if not valid_columns.size:
            return None
        nearest_x = int(valid_columns[np.argmin(np.abs(valid_columns - x))])
        nearest_ys = np.flatnonzero(mask[:, nearest_x])
        return int(nearest_ys[0]) if nearest_ys.size else None


class KeypointPredictionTask(DnnTask):
    """Background worker for ONNX keypoint inference."""

    def __init__(
            self,
            image: np.ndarray,
            background_mask: np.ndarray | None,
            predictor: KeypointPredictor,
            name: str = '',
    ):
        super().__init__(name)
        self._image = image
        self._background_mask = background_mask
        self._predictor = predictor

    def _run(self) -> np.ndarray:
        return self._predictor.predict(
            self._image,
            self._background_mask,
        )

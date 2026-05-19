from __future__ import annotations

import cv2
import numpy as np
import copy
import logging
from typing import TYPE_CHECKING, Callable
from PySide6.QtCore import QObject
from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.task import DnnTask
from bsmu.vision.dnn.segmenter import Segmenter as DnnSegmenter
from bsmu.macula.inference.inferecer import EnsembleImageModelParams
from bsmu.vision.core.palette import Palette
from bsmu.macula.inference.utility import (
    RoiTiler, sigmoid_2d, preprocess_for_model, reverse_preprocess,
)

if TYPE_CHECKING:
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.storages.task import TaskStorage


class CurrentSOTA:
    """Ensemble segmentation pipeline with boundary-guided ROI extraction.

    Pipeline stages:
    1. RoiTiler -> crop initial ROI (threshold-based contour detection)
    2. ROI preprocessing (LongestMaxSize + PadIfNeeded + normalize)
    3. Boundary model -> extract tight ROI bbox
    4. Crop tight ROI from original (untransformed) ROI image
    5. Mask preprocessing (same transforms)
    6. Class model inference with combination rules + argmax
    7. Reverse: place back in ROI coords, add class 255 for non-retinal
    """

    DIRECT_CLASSES = (5, 9, 12, 13, 14, 15)

    def __init__(
        self,
        class_models: dict[int, DnnSegmenter],
        boundary_model: DnnSegmenter,
        ped_model: DnnSegmenter | None,
        three_and_six_model: DnnSegmenter | None,
    ):
        self.class_models = class_models
        self.boundary_model = boundary_model
        self.ped_model = ped_model
        self.three_and_six_model = three_and_six_model
        self.max_cls = max(class_models.keys()) + 1 if class_models else 1

    @staticmethod
    def _run_model(segmenter: DnnSegmenter, image: np.ndarray) -> np.ndarray:
        """Run a single model and apply sigmoid activation."""
        outputs = segmenter.segment(image)
        return sigmoid_2d(outputs)

    @staticmethod
    def _extract_roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        """Extract bounding box of non-zero region in a binary mask.

        Returns (y_min, y_max, x_min, x_max) — inclusive-exclusive like numpy slicing.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return (0, mask.shape[0], 0, mask.shape[1])
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (y_min, y_max + 1, x_min, x_max + 1)

    def _predict_direct_classes(
        self, pp_roi: np.ndarray, content_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Run direct-threshold models and return a per-class prediction stack."""
        stack = np.zeros((*target_shape, self.max_cls), dtype=np.float32)
        for cls_idx in self.DIRECT_CLASSES:
            if cls_idx not in self.class_models:
                continue
            pred = self._run_model(self.class_models[cls_idx], pp_roi)
            pred = reverse_preprocess(pred, content_shape, target_shape)
            stack[:, :, cls_idx] = pred * (pred > 0.5)
        return stack

    def _predict_combined_1_4(
        self, pp_roi: np.ndarray, content_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Combine models 1, 4, and 4PED for classes 1 and 4."""
        stack = np.zeros((*target_shape, self.max_cls), dtype=np.float32)
        if not (1 in self.class_models and 4 in self.class_models and self.ped_model is not None):
            return stack

        p_all = reverse_preprocess(
            self._run_model(self.ped_model, pp_roi), content_shape, target_shape)
        p_1 = reverse_preprocess(
            self._run_model(self.class_models[1], pp_roi), content_shape, target_shape)
        p_4 = reverse_preprocess(
            self._run_model(self.class_models[4], pp_roi), content_shape, target_shape)

        p_ped = p_1 > p_4
        p_1_result = p_all * p_ped
        p_4_result = p_all * (~p_ped)
        stack[:, :, 1] = p_1_result * (p_1_result > 0.5)
        stack[:, :, 4] = p_4_result * (p_4_result > 0.5)
        return stack

    def _predict_combined_3_6(
        self, pp_roi: np.ndarray, content_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Combine models 3, 6, and 3and6 for classes 3 and 6."""
        stack = np.zeros((*target_shape, self.max_cls), dtype=np.float32)
        if not (3 in self.class_models and 6 in self.class_models and self.three_and_six_model is not None):
            return stack

        p_all = reverse_preprocess(
            self._run_model(self.three_and_six_model, pp_roi), content_shape, target_shape)
        p_3 = reverse_preprocess(
            self._run_model(self.class_models[3], pp_roi), content_shape, target_shape)
        p_6 = reverse_preprocess(
            self._run_model(self.class_models[6], pp_roi), content_shape, target_shape)

        p_3p = p_3 > p_6
        p_3_result = p_all * p_3p
        p_6_result = p_all * (~p_3p)
        stack[:, :, 3] = p_3_result * (p_3_result > 0.5)
        stack[:, :, 6] = p_6_result * (p_6_result > 0.5)
        return stack

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run the full ensemble segmentation pipeline."""
        # 1: Initial ROI via RoiTiler (cropping apps info)
        roi_tiler = RoiTiler()
        roi_image = next(roi_tiler.split(image))

        # 2: ROI preprocessing
        pp_image, bnd_content_shape = preprocess_for_model(roi_image)

        # 3: Boundary model -> unpad -> resize to roi_image dimensions
        bnd_pred = self._run_model(self.boundary_model, pp_image)
        bnd_mask_processed = bnd_pred > 0.5
        content_h, content_w = bnd_content_shape
        bnd_mask = cv2.resize(
            bnd_mask_processed[:content_h, :content_w].astype(np.uint8),
            roi_image.shape[::-1],
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # 4: Extract retinal ROI from roi_image
        y_min, y_max, x_min, x_max = self._extract_roi_bbox(bnd_mask)
        tight_roi = roi_image[y_min:y_max, x_min:x_max]

        # 5: Mask preprocessing
        pp_roi, content_shape = preprocess_for_model(tight_roi)

        # 6: Class model inference with combination rules
        mask_stack = self._predict_direct_classes(pp_roi, content_shape, tight_roi.shape)
        mask_stack += self._predict_combined_1_4(pp_roi, content_shape, tight_roi.shape)
        mask_stack += self._predict_combined_3_6(pp_roi, content_shape, tight_roi.shape)

        result = np.argmax(mask_stack, axis=2).astype(np.uint8)

        # place tight ROI result back in roi_image coordinates
        full_roi_mask = np.zeros(roi_image.shape, dtype=np.uint8)
        full_roi_mask[y_min:y_max, x_min:x_max] = result

        full_roi_mask[~bnd_mask] = 255

        # 7: Assemble back into original image
        roi_tiler.update(full_roi_mask)
        return roi_tiler.assemble()


class EnsembleSegmenter(QObject):
    def __init__(
        self,
        ensemble_model_params: EnsembleImageModelParams,
        mask_palette: Palette,
        task_storage: TaskStorage = None,
    ):
        super().__init__()
        self._ensemble_model_params = ensemble_model_params
        self._mask_palette = mask_palette
        self._task_storage = task_storage

        model_dir = ensemble_model_params.path

        def _create_segmenter(model_name: str) -> DnnSegmenter:
            """Create a DnnSegmenter for a single model file."""
            model_params = copy.deepcopy(ensemble_model_params)
            model_params.path = model_dir / model_name
            return DnnSegmenter(model_params)

        class_models = {
            cls_idx: _create_segmenter(model_name)
            for cls_idx, model_name in ensemble_model_params.class_models.items()
        }

        boundary_model = _create_segmenter(ensemble_model_params.boundary_model)

        ped_model = (
            _create_segmenter(ensemble_model_params.ped_model)
            if ensemble_model_params.ped_model
            else None
        )
        three_and_six_model = (
            _create_segmenter(ensemble_model_params.three_and_six_model)
            if ensemble_model_params.three_and_six_model
            else None
        )

        self._sota_inference = CurrentSOTA(
            class_models, boundary_model, ped_model, three_and_six_model,
        )

    @property
    def mask_palette(self) -> Palette:
        return self._mask_palette

    def segment_async(self, image: Image, on_finished: Callable[[np.ndarray], None] | None = None):
        task_name = f"SOTA Segmentation [{image.path_name}]"
        segmentation_task = EnsembleSegmentationTask(image.pixels, self._sota_inference, task_name)
        segmentation_task.on_finished = on_finished
        if self._task_storage is not None:
            self._task_storage.add_item(segmentation_task)
        ThreadPool.run_async_task(segmentation_task)

class EnsembleSegmentationTask(DnnTask):
    def __init__(self, image: np.ndarray, sota_inference: CurrentSOTA, name: str = ''):
        super().__init__(name)
        if len(image.shape) == 3:
            self._image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            self._image = image
        self._sota_inference = sota_inference

    def _run(self) -> np.ndarray:
        logging.info("Starting SOTA segmentation task.")
        mask = self._sota_inference(self._image)
        logging.info("SOTA segmentation task completed.")
        return mask


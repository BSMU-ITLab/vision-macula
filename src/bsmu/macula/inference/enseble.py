from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Dict, Callable

import numpy as np
from PySide6.QtCore import QObject

from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.task import DnnTask
from bsmu.vision.dnn.segmenter import Segmenter as DnnSegmenter

if TYPE_CHECKING:
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.storages.task import TaskStorage

from bsmu.macula.inference.inferecer import EnsembleImageModelParams
from bsmu.vision.core.palette import Palette
import cv2

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

        self._segmenters: Dict[str, DnnSegmenter] = {}
        for model_name, mask_class in ensemble_model_params.name_to_mask_class.items():
            model_path = ensemble_model_params.path.parent / model_name
            model_params = copy.deepcopy(ensemble_model_params)
            model_params.path = model_path
            self._segmenters[model_name] = DnnSegmenter(model_params)

    @property
    def mask_palette(self) -> Palette:
        return self._mask_palette

    def segment_async(
            self,
            image: Image,
            on_finished: Callable[[np.ndarray, Dict[int, int]], None] | None = None,
    ):

        task_name = f"Ensemble Segmentation [{image.path_name}]"
        segmentation_task = EnsembleSegmentationTask(
            image.pixels,
            self._segmenters,
            self._ensemble_model_params.name_to_mask_class,
            task_name,
        )
        segmentation_task.on_finished = on_finished

        if self._task_storage is not None:
            self._task_storage.add_item(segmentation_task)
        ThreadPool.run_async_task(segmentation_task)


class EnsembleSegmentationTask(DnnTask):

    def __init__(
            self,
            image: np.ndarray,
            segmenters: Dict[str, DnnSegmenter],
            name_to_mask_class: Dict[str, int],
            name: str = '',
    ):
        super().__init__(name)
        self._image = image
        self._segmenters = segmenters
        self._name_to_mask_class = name_to_mask_class

    def _run(self) -> np.ndarray:
        logging.info("Starting ensemble segmentation task.")
        # Preparing image here, to avoid excessive preprocessing in DnnSegmenter
        prepared_image, cords = list(self._segmenters.values())[0].model_params.preprocessed_input(self._image, False)
        x, y, w, h = cords
        empty_mask = np.zeros(self._image.shape[:2], dtype=np.uint8) # to handel images with software info
        num_models = len(self._segmenters)
        model_predictions = np.zeros((num_models, h, w), dtype=np.float32)
        labels = np.zeros(num_models, dtype=np.uint8)

        for i, (model_name, segmenter) in enumerate(self._segmenters.items()):
            logging.info(f"Segmenting with model: {model_name}")
            mask = segmenter.segment(prepared_image)

            # Code from DnnSegmenter.segment method
            if (w, h) != mask.shape[:2]:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
                im = cv2.resize(prepared_image, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)

            model_predictions[i] = np.where(mask > segmenter.model_params.mask_binarization_threshold, mask, 0)
            labels[i] = self._name_to_mask_class[model_name]

        max_indices = np.argmax(model_predictions, axis=0)
        zero_mask = np.all(model_predictions==0, axis=0)
        final_mask = labels[max_indices]
        final_mask[zero_mask] = 0
        empty_mask[y:y + h, x:x + w] = final_mask
        # ---------- Добавлено: расчет площадей ----------
        class_areas: Dict[int, int] = {}
        for label_id in np.unique(final_mask):
            if label_id == 0:  # фон
                continue
            area_px = int(np.count_nonzero(final_mask == label_id))
            class_areas[label_id] = area_px
            logging.info(f"Class {label_id} area = {area_px} px")

        self.class_areas = class_areas
        # -----------------------------------------------

        logging.info("Ensemble segmentation task completed.")
        return empty_mask, im, cords, class_areas

from __future__ import annotations

import cv2
import numpy as np
import copy
import logging
from typing import TYPE_CHECKING, Dict, Callable
from PySide6.QtCore import QObject
from bsmu.vision.core.concurrent import ThreadPool
from bsmu.vision.core.task import DnnTask
from bsmu.vision.dnn.segmenter import Segmenter as DnnSegmenter
from bsmu.macula.inference.inferecer import EnsembleImageModelParams
from bsmu.vision.core.palette import Palette
from bsmu.macula.inference.utility import BoundaryTiler, RoiTiler, tiled_inference, sigmoid_2d

if TYPE_CHECKING:
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.storages.task import TaskStorage

class CurrentSOTA:
    def __init__(self, tile_dict, fullsize_dict, boundary_model):
        self.seg_model_dict = tile_dict
        self.fullsize_model_dict = fullsize_dict
        self.boundary_model = boundary_model
        self.max_cls = max(map(lambda x: max(x.keys()), (tile_dict, fullsize_dict))) + 1

    def run_model(self, segmenter: DnnSegmenter, image: np.ndarray) -> np.ndarray:
        outputs = segmenter.segment(image)
        outputs = sigmoid_2d(outputs)
        return outputs

    def preprocess(self, image):
        b_mask = self.run_model(self.boundary_model, image)
        b_mask = (b_mask > 0.5).astype(np.uint8)
        return image, b_mask

    @tiled_inference(lambda self: BoundaryTiler(kernel=(512, 512), stride=(256, 256), cls_num=self.max_cls))
    def model_call(self, image):
        stack = np.zeros((512, 512, self.max_cls), dtype=np.float32)
        for cls_idx, model in self.seg_model_dict.items():
            pred = self.run_model(model, image)
            pred = (pred >= 0.5) * pred
            stack[:, :, cls_idx] = pred
        return stack

    def fullsize_model_call(self, image):
        oH, oW = image.shape
        image = cv2.resize(image, (512, 256))
        stack = np.zeros((oH, oW, self.max_cls), dtype=np.float32)
        for cls_idx, model in self.fullsize_model_dict.items():
            pred = self.run_model(model, image)
            pred = cv2.resize(pred, (oW, oH), interpolation=cv2.INTER_LINEAR)
            pred = (pred >= 0.5) * pred
            stack[:, :, cls_idx] = pred
        return stack

    @tiled_inference(lambda self: RoiTiler())
    def __call__(self, image):
        image, b_mask = self.preprocess(image)
        fullsize_mask = self.fullsize_model_call(image)
        # tiled_mask = self.model_call(image, b_mask)
        # result = np.argmax(tiled_mask + fullsize_mask, axis=2).astype(np.uint8)
        result = np.argmax(fullsize_mask, axis=2).astype(np.uint8)
        return result

class EnsembleSegmenter(QObject):
    def __init__(self, ensemble_model_params: EnsembleImageModelParams, mask_palette: Palette, task_storage: TaskStorage = None):
        super().__init__()
        self._ensemble_model_params = ensemble_model_params
        self._mask_palette = mask_palette
        self._task_storage = task_storage
        tile_dict = {}
        fullseg_dict = {}
        model_dir = ensemble_model_params.path
        for cls_idx, model_name in ensemble_model_params.tiler_name.items():
            path = model_dir / model_name
            model_params = copy.deepcopy(ensemble_model_params)
            model_params.path = path
            tile_dict[cls_idx] = DnnSegmenter(model_params)
        for cls_idx, model_name in ensemble_model_params.fullseg_name.items():
            path = model_dir / model_name
            model_params = copy.deepcopy(ensemble_model_params)
            model_params.path = path
            fullseg_dict[cls_idx] = DnnSegmenter(model_params)
        boundary_path = model_dir / ensemble_model_params.boundary_model
        boundary_params = copy.deepcopy(ensemble_model_params)
        boundary_params.path = boundary_path
        boundary_model = DnnSegmenter(boundary_params)
        self._sota_inference = CurrentSOTA(tile_dict, fullseg_dict, boundary_model)

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
        self._image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self._sota_inference = sota_inference

    def _run(self) -> np.ndarray:
        logging.info("Starting SOTA segmentation task.")
        mask = self._sota_inference(self._image)
        logging.info("SOTA segmentation task completed.")
        return mask


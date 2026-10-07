"""MDI adapter that displays predicted retinal keypoints as an overlay."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bsmu.vision.core.image import FlatImage, MaskDrawMode
from bsmu.vision.core.palette import Palette
from bsmu.vision.core.visibility import Visibility

from bsmu.macula.inference.keypoint import KeypointPredictor
from bsmu.macula.infervis.mdi import MdiSegmenter

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi


MASK_LAYER_NAME = 'masks'
KEYPOINT_LAYER_NAME = 'keypoints'
KEYPOINT_PALETTE = Palette.from_row_by_name({
    'background': [0, 0, 0, 0, 0],
    'fovea': [1, 0, 0, 255, 255],
    'foveola': [2, 0, 255, 0, 255],
})


class MdiKeypointPredictor(MdiSegmenter):
    """Run keypoint inference for the active image and add a dot layer."""

    def __init__(self, predictor: KeypointPredictor, mdi: Mdi):
        super().__init__(mdi)
        self._predictor = predictor

    def predict_async(self) -> None:
        layered_image, image = self._check_duplicate_mask_and_get_active_layered_image(
            KEYPOINT_LAYER_NAME,
            mask_draw_mode=MaskDrawMode.REDRAW_ALL,
        )
        if layered_image is None or image is None:
            return

        background_mask = self._existing_background_mask(layered_image)
        on_finished = partial(self._on_prediction_finished, layered_image=layered_image)
        self._predictor.predict_async(
            image.pixels,
            background_mask,
            on_finished=on_finished,
        )

    @staticmethod
    def _existing_background_mask(layered_image: LayeredImage) -> np.ndarray | None:
        mask_layer = layered_image.layer_by_name(MASK_LAYER_NAME)
        if mask_layer is None or not mask_layer.is_image_pixels_valid:
            return None
        return mask_layer.image_pixels

    def _on_prediction_finished(
            self,
            keypoint_mask: np.ndarray,
            *,
            layered_image: LayeredImage,
    ) -> None:
        layered_image.add_layer_or_modify_pixels(
            KEYPOINT_LAYER_NAME,
            keypoint_mask,
            FlatImage,
            palette=KEYPOINT_PALETTE,
            visibility=Visibility(True, 0.9),
        )

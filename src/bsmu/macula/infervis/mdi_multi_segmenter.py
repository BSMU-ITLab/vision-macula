from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bsmu.macula.infervis.mdi import MdiSegmenter
from bsmu.vision.core.visibility import Visibility
from bsmu.macula.inference.multi_segmenter import MultiSegmenter
from bsmu.vision.core.image import MaskDrawMode

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.core.image import FlatImage
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi

class MultiMdiSegmenter(MdiSegmenter):
    def __init__(self, segmenter: MultiSegmenter, mdi: Mdi):
        super().__init__(mdi)
        self._segmenter = segmenter

    @property
    def mask_foreground_class(self) -> int:
        return self._segmenter.mask_palette.row_index_by_name('foreground')

    @property
    def mask_background_class(self) -> int:
        return self._segmenter.mask_palette.row_index_by_name('background')

    def segment_async(
            self,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        layered_image, image = self._check_duplicate_mask_and_get_active_layered_image(
            mask_layer_name,
            mask_draw_mode=mask_draw_mode,
        )

        if image is None:
            return

        on_finished = partial(
            self._on_segmentation_finished,
            layered_image=layered_image,
            mask_layer_name=mask_layer_name,
            mask_draw_mode=mask_draw_mode,
        )
        self._segmenter.segment_async(image, on_finished=on_finished)

    def _expand_mask_to_full_image(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            crop_coords: tuple[int, int, int, int],
    ) -> np.ndarray:
        y1, y2, x1, x2 = crop_coords
        full_shape = layered_image.layer_by_name('images').image_pixels.shape[:2]  # (height, width)
        full_mask = np.zeros(full_shape, dtype=mask.dtype)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    def _on_segmentation_finished(
            self,
            mask_with_coords: tuple[np.ndarray, tuple[int, int, int, int]],
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        mask, crop_coords = mask_with_coords
        #full_mask = self._expand_mask_to_full_image(mask, layered_image, crop_coords)
        self.update_mask_layer(mask, layered_image, mask_layer_name, mask_draw_mode)

    def update_mask_layer(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
        from bsmu.vision.core.image import FlatImage
        mask_layer = layered_image.layer_by_name(mask_layer_name)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if (mask_draw_mode == MaskDrawMode.REDRAW_ALL
                or mask_layer is None
                or not mask_layer.is_image_pixels_valid
                or mask_draw_mode == MaskDrawMode.OVERLAY_FOREGROUND):

            layered_image.add_layer_or_modify_pixels(
                mask_layer_name,
                mask,
                FlatImage,
                palette=self._segmenter.mask_palette,
                visibility=Visibility(True, 0.75),
            )
        elif mask_draw_mode == MaskDrawMode.FILL_BACKGROUND:
            is_modified = mask_layer.image_pixels == self.mask_background_class
            mask_layer.image_pixels[is_modified] = mask[is_modified]
            mask_layer.image.emit_pixels_modified()
        else:
            raise ValueError(f'Invalid MaskDrawMode: {mask_draw_mode}')

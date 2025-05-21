from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from bsmu.macula.infervis.mdi import MdiSegmenter
from bsmu.vision.core.visibility import Visibility
from bsmu.macula.inference.enseble import EnsembleSegmenter
from bsmu.vision.core.image import MaskDrawMode

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.core.image import FlatImage
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi

class EnsembleMdiSegmenter(MdiSegmenter):
    def __init__(self, segmenter: EnsembleSegmenter, mdi: Mdi):
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

    def _on_segmentation_finished(
            self,
            mask: np.ndarray,
            layered_image: LayeredImage,
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):
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
        # Temp fix to redraw the entire mask even for MaskDrawMode.OVERLAY_FOREGROUND mode
        if mask_draw_mode == MaskDrawMode.REDRAW_ALL or mask_layer is None or not mask_layer.is_image_pixels_valid or MaskDrawMode.OVERLAY_FOREGROUND:
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

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from PySide6.QtCore import QObject
from bsmu.vision.core.image import MaskDrawMode
from bsmu.vision.widgets.viewers.image.layered import LayeredImageViewer, LayeredImageViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi


class MdiInferencer(QObject):
    def __init__(self, mdi: Mdi):
        super().__init__()

        self._mdi = mdi

    def _active_layered_image(self) -> LayeredImage | None:
        viewer = self._active_layered_image_viewer()
        if viewer is not None:
            return viewer.data

    def _active_layered_image_viewer(self) -> LayeredImageViewer | None:
        layered_image_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredImageViewerHolder)
        return layered_image_viewer_sub_window and layered_image_viewer_sub_window.layered_image_viewer


class MdiSegmenter(MdiInferencer):
    def _check_duplicate_mask_and_get_active_layered_image(
            self,
            mask_layer_name: str,
            show_repaint_confirmation: bool = True,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ) -> tuple[LayeredImage | None, Image | None]:

        layered_image_viewer = self._active_layered_image_viewer()
        if layered_image_viewer is None or (layered_image := layered_image_viewer.data) is None:
            return None, None

        if (show_repaint_confirmation and
                not layered_image_viewer.is_confirmed_repaint_duplicate_mask_layer(mask_layer_name, mask_draw_mode)):
            return None, None

        image_layer = layered_image.layers[0]
        return layered_image, image_layer.image

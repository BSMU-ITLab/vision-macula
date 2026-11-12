from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject

from bsmu.vision.core.image import MaskDrawMode
from bsmu.vision.core.layers import RasterLayer
from bsmu.vision.widgets.viewers.layered import LayeredDataViewerHolder

if TYPE_CHECKING:
    from bsmu.vision.core.data.layered import LayeredData
    from bsmu.vision.core.data.raster import Raster
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi
    from bsmu.vision.widgets.viewers.layered import LayeredDataViewer


class MdiInferencer(QObject):
    def __init__(self, mdi: Mdi):
        super().__init__()

        self._mdi = mdi

    def _active_layered_data_viewer(self) -> LayeredDataViewer | None:
        layered_data_viewer_sub_window = self._mdi.active_sub_window_with_type(LayeredDataViewerHolder)
        return layered_data_viewer_sub_window and layered_data_viewer_sub_window.layered_data_viewer


class MdiSegmenter(MdiInferencer):
    def _check_duplicate_mask_and_get_active_layered_image(
        self,
        mask_layer_name: str,
        show_repaint_confirmation: bool = True,
        mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ) -> tuple[LayeredData | None, Raster | None]:

        layered_data_viewer = self._active_layered_data_viewer()
        if layered_data_viewer is None or (layered_data := layered_data_viewer.data) is None:
            return None, None

        if show_repaint_confirmation and not layered_data_viewer.is_confirmed_repaint_duplicate_mask_layer(
            mask_layer_name, mask_draw_mode
        ):
            return None, None

        raster_layer = layered_data.layers[0]
        if not isinstance(raster_layer, RasterLayer):
            return layered_data, None

        return layered_data, raster_layer.data

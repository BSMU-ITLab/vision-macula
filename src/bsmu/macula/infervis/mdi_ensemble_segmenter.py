from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from bsmu.macula.infervis.mdi import MdiSegmenter
from bsmu.vision.core.visibility import Visibility
from bsmu.macula.inference.enseble import EnsembleSegmenter
from bsmu.macula.records.eye_info_data import Measurement
from bsmu.vision.core.image import MaskDrawMode

if TYPE_CHECKING:
    from bsmu.vision.core.image.layered import LayeredImage
    from bsmu.vision.core.image import FlatImage
    from bsmu.vision.plugins.doc_interfaces.mdi import Mdi
import cv2

class EnsembleMdiSegmenter(MdiSegmenter):
    def __init__(self, segmenter: EnsembleSegmenter, mdi: Mdi):
        super().__init__(mdi)
        self._segmenter = segmenter
        self._fovea_mask_path: str | None = None
        self._current_layered_image: 'LayeredImage | None' = None

    @property
    def mask_foreground_class(self) -> int:
        return self._segmenter.mask_palette.row_index_by_name('foreground')

    @property
    def mask_background_class(self) -> int:
        return self._segmenter.mask_palette.row_index_by_name('background')

    def set_fovea_mask_path(self, path: str):
        """Устанавливает путь к файлу маски fovea"""
        self._fovea_mask_path = path

    def init_fovea_mask_layer(self, layered_image: 'LayeredImage'):
        """Инициализирует пустой слой mask-fovea при загрузке изображения"""
        try:
            # Сохраняем ссылку на текущее изображение
            self._current_layered_image = layered_image
            
            # Получаем размер первого слоя (image)
            if not layered_image.layers:
                print("Ошибка: нет слоёв в layered_image")
                return
            
            image_layer = layered_image.layers[0]
            image_size = image_layer.image.pixels.shape[:2]  # (height, width)
            
            # Создаём пустой слой маски fovea
            empty_fovea_mask = np.zeros(image_size, dtype=np.uint8)
            
            from bsmu.vision.core.image import FlatImage
            layered_image.add_layer_or_modify_pixels(
                'mask-fovea',
                empty_fovea_mask,
                FlatImage,
                palette=self._segmenter.mask_palette,
                visibility=Visibility(True, 0.5),
            )
            print(f"Пустой слой mask-fovea создан с размером {image_size}")
        except Exception as e:
            print(f"Ошибка при инициализации слоя mask-fovea: {e}")

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
            prepared_image: np.ndarray,
            cords: tuple,
            class_areas: Optional[Dict[int, int]] = None,  # <-- второй позиционный, опционален
            *,
            layered_image: 'LayeredImage',
            mask_layer_name: str,
            mask_draw_mode: MaskDrawMode = MaskDrawMode.REDRAW_ALL,
    ):

        # Обновляем маску с найденным L
        self.update_mask_layer(mask, layered_image, mask_layer_name, mask_draw_mode)
        
        # Загружаем и добавляем маску fovea
        self._load_and_add_fovea_mask_layer(layered_image)

    def update_mask_layer(
            self,
            mask: np.ndarray,
            layered_image: 'LayeredImage',
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

    def add_fovea_mask_layer(
            self,
            mask_fovea: np.ndarray,
            layered_image: 'LayeredImage',
            mask_fovea_layer_name: str = 'mask-fovea',
    ):
        """Добавляет слой маски фовеа в layered image"""
        from bsmu.vision.core.image import FlatImage
        layered_image.add_layer_or_modify_pixels(
            mask_fovea_layer_name,
            mask_fovea,
            FlatImage,
            palette=self._segmenter.mask_palette,
            visibility=Visibility(True, 0.5),
        )

    def _load_and_add_fovea_mask_layer(
            self,
            layered_image: 'LayeredImage',
            mask_fovea_layer_name: str = 'mask-fovea',
    ):
        """Загружает маску фовеа из файла и добавляет её в layered image"""
        try:
            # Если пользователь не выбрал файл, используем hardcoded путь
            mask_path = self._fovea_mask_path or "C:/Users/Elena_Himbitskaya/Desktop/2026/images-masks-v7/set01-v7/masks-fovea/01-001-0_0.png"
            
            mask_fovea_pixels = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if mask_fovea_pixels is None:
                print(f"Ошибка: не удалось загрузить маску фовеа из {mask_path}")
                return
            
            self.add_fovea_mask_layer(mask_fovea_pixels, layered_image, mask_fovea_layer_name)
            print(f"Маска фовеа загружена из: {mask_path}")
        except Exception as e:
            print(f"Ошибка при загрузке маски фовеа: {e}")

    def load_and_add_fovea_mask_layer_from_active_image(
            self,
            mask_fovea_layer_name: str = 'mask-fovea',
    ):
        """Загружает маску фовеа и добавляет её в активное изображение"""
        # Используем сохранённую ссылку на layered_image
        if self._current_layered_image is not None:
            self._load_and_add_fovea_mask_layer(self._current_layered_image, mask_fovea_layer_name)
            return
        
        # Если нет сохранённой ссылки, показываем ошибку
        raise RuntimeError("No layered image initialized. Please open an image first.")

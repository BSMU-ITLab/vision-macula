from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Tuple, List
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

from bsmu.vision.core.bbox import BBox
from bsmu.vision.core.palette import Palette
from bsmu.vision.dnn.inferencer import ImageModelParams

if TYPE_CHECKING:
    from bsmu.vision.core.image import Image
    from bsmu.vision.plugins.storages.task import TaskStorage


def get_zone(imagem) -> List[int]:
    if len(imagem.shape) == 3 and imagem.shape[2] == 3:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagem
    image = cv2.bitwise_not(gray)
    contours, _ = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = sorted_contours[0]
    x, y, w, h = cv2.boundingRect(largest_contour)
    return [y, y + h, x, x + w]

def apply_segmentation(image: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    
    # Если цветное, конвертировать в grayscale
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = image.astype(np.float32) / 255.0
    
    # Теперь image shape: (H, W) -> добавить channel dimension
    image = np.expand_dims(image, axis=0)  # (1, H, W)
    image = np.expand_dims(image, axis=0)  # (1, 1, H, W)
    
    onnx_output = session.run(None, {input_name: image})
    mask = np.argmax(onnx_output[0], axis=1).squeeze()
    return mask


def apply_segmentation_to_tiles(tiles: List[np.ndarray], session: ort.InferenceSession) -> List[np.ndarray]:
    return [apply_segmentation(tile, session) for tile in tiles]

def tile_image(image: np.ndarray, tile_height=512, tile_width=64, shift=32) -> List[np.ndarray]:
    img_height, img_width = image.shape[:2]
    tiles = []
    for x in range(0, img_width - tile_width + 1, shift):
        tile = image[0:tile_height, x:x + tile_width]
        tiles.append(tile)
    return tiles

def combine_tiles(
    masks: List[np.ndarray],
    image_shape: Tuple[int, int],
    tile_height=512,
    tile_width=64,
    shift=32,
    threshold_ratio=0.8,
    num_classes=10
) -> np.ndarray:
    mask_image = np.zeros((*image_shape, num_classes), dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.float32)
    for idx, mask in enumerate(masks):
        x_start = idx * shift
        mask_resized = cv2.resize(mask, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
        for cls in range(num_classes):
            mask_image[0:tile_height, x_start:x_start + tile_width, cls] += (mask_resized == cls) * 0.5
        counts[0:tile_height, x_start:x_start + tile_width] += 0.5
    mask_image /= np.maximum(counts[:, :, None], 1)
    final_mask = np.argmax(mask_image, axis=2)
    for cls in range(1, num_classes):
        class_mask = mask_image[:, :, cls] >= (threshold_ratio * 0.5 * counts)
        final_mask[class_mask] = cls
    final_mask[final_mask == 1] = 0
    logging.info("Sorted unique values in mask: %s", np.unique(final_mask))
    logging.info(f"shape of final mask  {final_mask.shape}")
    logging.info(f"shape of image  {image_shape}")
    return final_mask

def expand_mask_to_full(
    mask_cropped: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    full_shape: tuple[int, int]
) -> np.ndarray:
    y1, y2, x1, x2 = crop_coords
    full_mask = np.zeros(full_shape, dtype=mask_cropped.dtype)
    full_mask[y1:y2, x1:x2] = mask_cropped
    return full_mask



class MultiSegmenter:
    def __init__(
        self,
        model_params: ImageModelParams,
        mask_palette: Palette,
        task_storage: TaskStorage | None = None,
        num_classes: int = 10,
        tile_height: int = 512,
        tile_width: int = 64,
        tile_shift: int = 32,
        threshold_ratio: float = 0.8,
    ):
        self._model_params = model_params
        self._mask_palette = mask_palette
        self._task_storage = task_storage
        self._num_classes = num_classes
        self._tile_height = tile_height
        self._tile_width = tile_width
        self._tile_shift = tile_shift
        self._threshold_ratio = threshold_ratio

        # Создаем сессию onnxruntime из пути к модели
        model_path = Path(model_params.path)  # или model_params.path если это уже Path
        self._session = ort.InferenceSession(str(model_path))

    @property
    def mask_palette(self) -> Palette:
        return self._mask_palette

    def segment(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        # Get the cropping coordinates
        y1, y2, x1, x2 = get_zone(image)
        cropped_image = image[y1:y2, x1:x2]
        #cv2.imwrite("cropped_image.jpg", cropped_image)
        print(f"Type of cropped_image: {type(cropped_image)}")
        
        # Resize the cropped image
        resized_cropped_image = cv2.resize(cropped_image, (1024, 512)).astype(np.uint8)
        
        # Process tiles and apply segmentation
        tiles = tile_image(resized_cropped_image, self._tile_height, self._tile_width, self._tile_shift)
        segmented_tiles = apply_segmentation_to_tiles(tiles, self._session)
        
        # Combine the segmented tiles into one mask
        combined_mask = combine_tiles(
            segmented_tiles,
            image_shape=resized_cropped_image.shape[:2],
            tile_height=self._tile_height,
            tile_width=self._tile_width,
            shift=self._tile_shift,
            threshold_ratio=self._threshold_ratio,
            num_classes=self._num_classes,
        )
        
        print(f"Type of combined_mask: {type(combined_mask)}")
        # Resize combined_mask to match the cropped_image size
        resized_combined_mask = cv2.resize(combined_mask.astype(np.uint8), (cropped_image.shape[1], cropped_image.shape[0]))
        # Create a full mask the same size as the original image, filled with zeros
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Place the resized_combined_mask into the corresponding region of the full_mask
        full_mask[y1:y2, x1:x2] = resized_combined_mask

        # Return the full mask and the coordinates
        return full_mask, (y1, y2, x1, x2)


    def segment_async(
        self,
        image: Image,
        on_finished: Callable[[np.ndarray], None] | None = None,
    ):
        import threading

        def task():
            logging.info("Starting async multi-segmentation task.")
            mask = self.segment(image.pixels)
            logging.info("Async multi-segmentation task finished.")
            if on_finished:
                on_finished(mask)

        threading.Thread(target=task, daemon=True).start()

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from bsmu.vision.dnn.inferencer import ImageModelParams

@dataclass
class EnsembleImageModelParams(ImageModelParams):
    """
    Ensemble model where multiple model names map to different mask classes.
    """
    name_to_mask_class: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config_data: dict, model_dir: Path) -> "EnsembleImageModelParams":
        model_mapping = config_data.get("name", {})  # Expecting dictionary like {"model_1": 1, "model_5": 5}
        if not isinstance(model_mapping, dict):
            raise ValueError("'name' in config should be a dictionary mapping model names to mask classes.")

        field_names = {f.name for f in fields(cls)}
        SENTINEL = object()
        field_name_to_config_value = {
            field_name: config_value for field_name in field_names
            if (config_value := config_data.get(field_name, SENTINEL)) != SENTINEL}

        return cls(
            path=model_dir / list(model_mapping.keys())[0],
            name_to_mask_class = model_mapping,
            **field_name_to_config_value)

    def preprocessed_input(self, image: np.ndarray, skip_preprocessing=True) -> (np.ndarray, Tuple[int]):

        if skip_preprocessing:   # Quick fix to avoid DnnSegmenter preprocessing
            image = image[np.newaxis, :, :]
            return image

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Crop software info
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        image = image[y:y + h, x:x + w]

        if image.shape != self.input_size[1:]:
            image = cv2.resize(image[1:], self.input_size[1:], interpolation=cv2.INTER_AREA)

        if self.normalize:
            image = image.astype(np.float32)
            image /= 255.0
            image -= self.IMAGENET_MEAN.mean()
            image /= self.IMAGENET_STD.mean()

        return image, (x, y, w, h)

    def preprocessed_input_batch(self, src_batch: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return super().preprocessed_input_batch(src_batch)

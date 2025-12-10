from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Sequence
from bsmu.vision.dnn.inferencer import ImageModelParams
import numpy as np

@dataclass
class EnsembleImageModelParams(ImageModelParams):
    tiler_name: Dict[int, str] = field(default_factory=dict)
    fullseg_name: Dict[int, str] = field(default_factory=dict)
    boundary_model: str = ""

    @classmethod
    def from_config(cls, config_data: dict, model_dir: Path) -> "SOTAImageModelParams":
        tiler = config_data.get("tiler_name", {})
        fullseg = config_data.get("fullseg_name", {})
        boundary = config_data.get("boundary_model", "")

        if not isinstance(tiler, dict) or not isinstance(fullseg, dict):
            raise ValueError("'tiler_name' and 'fullseg_name' in config should be dicts mapping class ids to model names.")

        field_names = {f.name for f in fields(cls)}
        SENTINEL = object()
        field_name_to_config_value = {
            field_name: config_value
            for field_name in field_names
            if field_name not in ("tiler_name", "fullseg_name", "boundary_model")
               and (config_value := config_data.get(field_name, SENTINEL)) != SENTINEL
        }

        return cls(
            path=model_dir,
            tiler_name={int(k): v for k, v in tiler.items()},
            fullseg_name={int(k): v for k, v in fullseg.items()},
            boundary_model=boundary,
            **field_name_to_config_value,
        )

    def preprocessed_input(self, image: np.ndarray, skip_preprocessing=True) -> np.ndarray:
        if self.normalize:
            image = image.astype(np.float32)
            image /= 255.0
            image -= self.IMAGENET_MEAN.mean()
            image /= self.IMAGENET_STD.mean()
        return image[np.newaxis, :, :]

    # def preprocessed_input(self, image: np.ndarray, skip_preprocessing=True) -> np.ndarray:
    #     if self.normalize:
    #         image = image.astype(np.float32)
    #         image /= 255.0
    #         image = (image - 0.5) / 0.5
    #     return image[np.newaxis, :, :]


    def preprocessed_input_batch(self, src_batch: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return super().preprocessed_input_batch(src_batch)
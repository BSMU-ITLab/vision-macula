from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Sequence
from bsmu.vision.dnn.inferencer import ImageModelParams
import numpy as np

@dataclass
class EnsembleImageModelParams(ImageModelParams):
    class_models: Dict[int, str] = field(default_factory=dict)
    boundary_model: str = ""
    ped_model: str = ""
    three_and_six_model: str = ""

    @classmethod
    def from_config(cls, config_data: dict, model_dir: Path) -> "EnsembleImageModelParams":
        class_models_raw = config_data.get("class_models", {})
        boundary = config_data.get("boundary_model", "")
        ped = config_data.get("ped_model", "")
        three_and_six = config_data.get("three_and_six_model", "")

        if not isinstance(class_models_raw, dict):
            raise ValueError("'class_models' in config should be a dict mapping class ids to model names.")

        field_names = {f.name for f in fields(cls)}
        special_fields = {"class_models", "boundary_model", "ped_model", "three_and_six_model"}
        SENTINEL = object()
        field_name_to_config_value = {
            field_name: config_value
            for field_name in field_names
            if field_name not in special_fields
               and (config_value := config_data.get(field_name, SENTINEL)) != SENTINEL
        }

        return cls(
            path=model_dir,
            class_models={int(k): v for k, v in class_models_raw.items()},
            boundary_model=boundary,
            ped_model=ped,
            three_and_six_model=three_and_six,
            **field_name_to_config_value,
        )

    def preprocessed_input(self, image: np.ndarray, skip_preprocessing: bool = True) -> np.ndarray:
        """Minimal preprocessing: just add the channel/batch axis.

        All resize/pad/normalize transforms happen externally in the
        inference pipeline (CurrentSOTA), not here.
        """
        return image[np.newaxis, :, :]

    def preprocessed_input_batch(self, src_batch: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return super().preprocessed_input_batch(src_batch)
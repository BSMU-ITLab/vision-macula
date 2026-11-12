from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bsmu.vision.dnn.inferencer import ImageModelConfig

if TYPE_CHECKING:
    from typing import Sequence


@dataclass
class EnsembleImageModelConfig(ImageModelConfig):
    class_models: dict[int, str] = field(default_factory=dict)
    boundary_model: str = ''
    ped_model: str = ''
    three_and_six_model: str = ''
    tr_model: str = ''
    catboost_model: str = ''

    def preprocessed_input(self, image: np.ndarray, _skip_preprocessing: bool = True) -> np.ndarray:
        """Minimal preprocessing: just add the channel/batch axis.

        All resize/pad/normalize transforms happen externally in the
        inference pipeline (CurrentSOTA), not here.
        """
        return image[np.newaxis, :, :]

    def preprocessed_input_batch(self, src_batch: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return super().preprocessed_input_batch(src_batch)

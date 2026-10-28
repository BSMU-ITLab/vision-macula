"""Shared colors for keypoint overlays and exported masks."""

from bsmu.vision.core.palette import Palette


# Palette row values are (class index, red, green, blue, alpha).
KEYPOINT_COLORS = {
    1: (0, 0, 255),  # fovea
    2: (0, 255, 0),  # foveola
}

KEYPOINT_PALETTE = Palette.from_row_by_name({
    'background': [0, 0, 0, 0, 0],
    'fovea': [1, *KEYPOINT_COLORS[1], 255],
    'foveola': [2, *KEYPOINT_COLORS[2], 255],
})

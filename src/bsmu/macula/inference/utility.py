import cv2
import numpy as np
from typing import Any, Callable, Iterable, Optional

def smooth_boundary(boundary):
    """
    Fix NaNs by interpolation and smooth with Gaussian filter.
    """
    x = np.arange(len(boundary))
    y = boundary.copy()
    nans = np.isnan(y)
    if np.any(nans):
        not_nans = np.where(~nans)[0]
        y[nans] = np.interp(x[nans], x[not_nans], y[~nans])
    return y


def extract_boundary(mask, tp="min", smooth=True, d_type=np.int16):
    h, w = mask.shape
    top_row = np.full(w, np.nan, dtype=np.float32)
    for col in range(w):
        rows = np.where(mask[:, col] > 0)[0]
        if rows.size:
            top_row[col] = rows.min() if tp == "min" else rows.max()
    return smooth_boundary(top_row).astype(d_type) if smooth else top_row.astype(d_type)

def sigmoid_2d(array_2d: np.ndarray) -> np.ndarray:
    if array_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array_2d.ndim}D")
    return 1 / (1 + np.exp(-array_2d))

def get_resize_value(length, kernel_size, stride_size):
    i = (length - kernel_size + stride_size) // stride_size
    excess = length - (kernel_size + (i - 1) * stride_size)
    return (kernel_size + i * stride_size) if excess > kernel_size // 2 else length - excess

def longest_max_size(image: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Resize image to fit within target_size, preserving aspect ratio."""
    h, w = image.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def pad_if_needed(image: np.ndarray, target_size: tuple[int, int] = (512, 512),
                  pad_value: float = 0) -> np.ndarray:
    """Pad image to target_size, adding padding to bottom and right."""
    h, w = image.shape
    target_w, target_h = target_size
    if h > target_h or w > target_w:
        raise ValueError(f"Image size ({h}, {w}) larger than target ({target_h}, {target_w})")
    return cv2.copyMakeBorder(image, 0, target_h - h, 0, target_w - w,
                              cv2.BORDER_CONSTANT, value=pad_value)


def preprocess_for_model(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply ROI/mask preprocessing transforms.

    Pipeline: LongestMaxSize(512,512) -> PadIfNeeded(512,512, zeros)
              -> Normalize(mean=0.5, std=0.5, max_value=255)
    Maps pixel values from [0, 255] to [-1, 1].

    Returns:
        Preprocessed image (always 512x512) and the (h, w) shape
        before padding — needed to correctly reverse the transform.
    """
    image = longest_max_size(image, (512, 512))
    content_shape = image.shape  # (h, w) before padding
    image = pad_if_needed(image, (512, 512), pad_value=0)
    image = image.astype(np.float32)
    image /= 255.0
    image = (image - 0.5) / 0.5
    return image, content_shape


def reverse_preprocess(
    pred: np.ndarray,
    content_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reverse the preprocess_for_model transform on a model prediction.

    Crops out the zero-padded region (content_shape), then resizes to
    target_shape (the original image dimensions before any preprocessing).

    Args:
        pred: Model output at 512x512.
        content_shape: (h, w) of valid content within the 512x512, as returned
                       by preprocess_for_model.
        target_shape: (h, w) to resize the unpadded content to.
    """
    content_h, content_w = content_shape
    unpadded = pred[:content_h, :content_w]
    target_h, target_w = target_shape
    return cv2.resize(unpadded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class BaseTiler:
    def __init__(self):
        self.original_shape = None
        self.buffer = None

class BoundaryTiler(BaseTiler):
    def __init__(self, kernel: tuple[int, int], stride: tuple[int, int], cls_num: int):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.cls_num = cls_num
        self._coords: Optional[tuple[int, int, int, int]] = None

    def split(self, sample: np.ndarray, info: Optional[np.ndarray] = None) -> Iterable[np.ndarray]:
        self.original_shape = sample.shape
        H, W = sample.shape
        nW = get_resize_value(W, self.kernel[1], self.stride[1])
        sample_resized = cv2.resize(sample, (nW, H))
        boundary_mask = cv2.resize(info, (nW, H)) if info is not None else np.zeros((H, nW))
        boundary_top = extract_boundary(boundary_mask)
        self.buffer = np.zeros((*sample_resized.shape, self.cls_num), dtype=np.float32)
        for left in range(0, nW - self.kernel[1] + 1, self.stride[1]):
            right = left + self.kernel[1]
            top = int(np.min(boundary_top[left:right]))
            bottom = min(top + self.kernel[0], H)
            self._coords = (top, bottom, left, right)
            tile = np.zeros(self.kernel, dtype=sample_resized.dtype)
            tile[: bottom - top, :] = sample_resized[top:bottom, left:right]
            yield tile

    def update(self, processed_tile: np.ndarray):
        top, bottom, left, right = self._coords
        self.buffer[top:bottom, left:right] += processed_tile[: bottom - top, :]

    def assemble(self) -> np.ndarray:
        return cv2.resize(self.buffer, self.original_shape[::-1], cv2.INTER_NEAREST)

class RoiTiler(BaseTiler):
    def __init__(self):
        super().__init__()
        self._coords: Optional[tuple[int, int, int, int]] = None

    def split(self, sample: np.ndarray, info: Optional[Any] = None) -> Iterable[np.ndarray]:
        self.original_shape = sample.shape
        self.buffer = np.zeros_like(sample, dtype=np.float32)
        H, W = sample.shape
        _, binary = cv2.threshold(sample, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._coords = (0, 0, W, H)
            yield sample

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        self._coords = (x, y, w, h)
        yield sample[y:y + h, x:x + w]

    def update(self, processed_tile: np.ndarray):
        x, y, w, h = self._coords
        self.buffer[y:y + h, x:x + w] = processed_tile

    def assemble(self) -> np.ndarray:
        return self.buffer

def tiled_inference(tiler_factory: Callable[[Any], Any]):
    def decorator(model_fn: Callable):
        def wrapper(self, image: np.ndarray, info=None):
            tiler = tiler_factory(self)
            for tile in tiler.split(image, info):
                processed = model_fn(self, tile)
                tiler.update(processed)
            return tiler.assemble()
        return wrapper
    return decorator

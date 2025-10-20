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

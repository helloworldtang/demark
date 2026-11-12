"""Image loading, watermark inpainting, and comparison rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demark.errors import DemarkError
from demark.region import Region

_INPAINT_METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}

# cv2 的函数签名返回松类型的 ndarray，别名放宽到 Any 以免在每个调用点 cast。
BGRImage = np.ndarray[Any, Any]
GrayImage = np.ndarray[Any, Any]


def load_image(path: Path) -> BGRImage:
    """Load an image from disk in BGR.

    Args:
        path: Image file path (png/jpg/webp/...).

    Returns:
        BGR image array.

    Raises:
        DemarkError: If the file is missing or not decodable.
    """
    if not path.is_file():
        raise DemarkError(f"image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise DemarkError(f"cannot decode image: {path}")
    return image


def save_image(path: Path, image: BGRImage) -> None:
    """Save a BGR image, creating parent directories as needed.

    Args:
        path: Destination path; the extension picks the encoder.
        image: BGR image array.

    Raises:
        DemarkError: If encoding or writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(f".{path.suffix.lstrip('.') or 'png'}", image)
    if not ok:
        raise DemarkError(f"cannot encode image to {path.suffix}")
    buf.tofile(str(path))


def build_mask(shape: tuple[int, ...], regions: list[Region], dilate_px: int = 3) -> GrayImage:
    """Rasterize regions into a binary inpaint mask.

    Args:
        shape: Image shape (h, w[, c]).
        regions: Rectangles to paint white on the mask.
        dilate_px: Grow each rectangle by this many pixels so inpainting
            also eats the watermark's anti-aliased fringe.

    Returns:
        uint8 mask, 255 inside regions and 0 elsewhere.
    """
    h, w = shape[0], shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        x0 = max(region.x - dilate_px, 0)
        y0 = max(region.y - dilate_px, 0)
        x1 = min(region.x + region.w + dilate_px, w)
        y1 = min(region.y + region.h + dilate_px, h)
        mask[y0:y1, x0:x1] = 255
    return mask


def inpaint_regions(image: BGRImage, regions: list[Region], method: str = "telea", radius: int = 5) -> BGRImage:
    """Remove watermark regions via OpenCV inpainting.

    Args:
        image: BGR source image.
        regions: Watermark rectangles.
        method: ``telea`` (fast, smooth) or ``ns`` (slower, fluid-like).
        radius: Inpainting neighborhood radius in pixels.

    Returns:
        New BGR image with regions inpainted; the input is not modified.

    Raises:
        DemarkError: If the method name is unknown.
    """
    if method not in _INPAINT_METHODS:
        raise DemarkError(f"unknown inpaint method {method!r}, expected one of {sorted(_INPAINT_METHODS)}")
    mask = build_mask(image.shape, regions)
    return cv2.inpaint(image, mask, radius, _INPAINT_METHODS[method])


def make_compare(original: BGRImage, result: BGRImage) -> BGRImage:
    """Render original and result side by side with a divider.

    Args:
        original: Source image.
        result: Processed image of the same size.

    Returns:
        Wide BGR image: original on the left, result on the right,
        separated by a 3px white line.
    """
    divider = np.full((original.shape[0], 3, 3), 255, dtype=np.uint8)
    return np.hstack([original, divider, result])

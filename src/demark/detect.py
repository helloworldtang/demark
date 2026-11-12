"""Heuristic watermark detection for images and videos.

Design goal is "prefer missing a watermark over flagging content": every
auto-detected region is a candidate the user can preview via ``demark detect``
and override with explicit ``-r`` rectangles. Two independent heuristics:

- Images: text/logo-like blobs (high morphological-gradient response) whose
  centers sit in corner bands or the bottom strip — where conference name
  bars, "recording" badges, and AI-platform marks live.
- Videos: spatio-temporal — sample frames, compute per-pixel temporal std and
  a median frame; a watermark is a region that never changes over time yet
  carries strong edges. Large static regions (tripod shots, letterboxing)
  are rejected by the area cap.
"""

from __future__ import annotations

import cv2
import numpy as np

from demark.image_ops import BGRImage, GrayImage
from demark.region import Region

# --- image heuristics -------------------------------------------------------
CORNER_WIDTH_RATIO = 0.4  # corner band width as a fraction of frame width
CORNER_HEIGHT_RATIO = 0.28  # corner band height as a fraction of frame height
BOTTOM_STRIP_RATIO = 0.75  # y above this fraction counts as "bottom strip"
MIN_AREA_FRAC = 0.0005  # blob smaller than this fraction of the frame is noise
MAX_AREA_FRAC = 0.06  # blob larger than this fraction is scenery, not a mark
BLOB_PAD_PX = 4  # padding added around each detected bbox

# --- video heuristics -------------------------------------------------------
VIDEO_SAMPLES = 30
TEMPORAL_STD_MAX = 8.0  # per-pixel time std below this counts as "static"
VIDEO_MAX_AREA_FRAC = 0.05  # static blob larger than this is background/scenery
VIDEO_MIN_AREA_FRAC = 0.001  # smaller static blobs are compression speckle
VIDEO_MIN_FILL = 0.15  # contour area / bbox area; scattered edges score low
VIDEO_CENTER_REJECT = 0.3  # blob centered in the middle 40%x40% of the frame is scenery


def _gradient_blobs(gray: GrayImage, merge_kernel: tuple[int, int]) -> list[Region]:
    """Find text/logo-like blobs via morphological gradient + contours.

    Args:
        gray: Grayscale image.
        merge_kernel: Dilation kernel (w, h) used to merge glyphs into blobs;
            wide-flat kernels suit horizontal text lines.

    Returns:
        Bounding-box Regions of candidate blobs.
    """
    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    merged = cv2.dilate(binary, np.ones(merge_kernel, np.uint8), iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        regions.append(Region(x=int(x), y=int(y), w=int(w), h=int(h)))
    return regions


def _in_mark_zone(cx: float, cy: float, width: int, height: int) -> bool:
    """Check whether a blob center lies in a corner band or the bottom strip.

    Args:
        cx: Blob center x in pixels.
        cy: Blob center y in pixels.
        width: Frame width.
        height: Frame height.

    Returns:
        True if the location is a plausible watermark spot.
    """
    band_w = width * CORNER_WIDTH_RATIO
    band_h = height * CORNER_HEIGHT_RATIO
    in_top = cy < band_h
    in_bottom = cy > height * BOTTOM_STRIP_RATIO
    in_left = cx < band_w
    in_right = cx > width - band_w
    return (in_top or in_bottom) and (in_left or in_right)


def detect_image(image: BGRImage) -> list[Region]:
    """Detect likely watermark regions in a screenshot-style image.

    Args:
        image: BGR image (conference screenshot, AI-generated picture, ...).

    Returns:
        Candidate Regions, possibly empty; padded by ``BLOB_PAD_PX``.
    """
    height, width = image.shape[0], image.shape[1]
    frame_area = width * height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    regions = []
    for blob in _gradient_blobs(gray, merge_kernel=(9, 3)):
        area_frac = blob.w * blob.h / frame_area
        if not MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
            continue
        cx = blob.x + blob.w / 2
        cy = blob.y + blob.h / 2
        if not _in_mark_zone(cx, cy, width, height):
            continue
        regions.append(
            Region(
                x=max(blob.x - BLOB_PAD_PX, 0),
                y=max(blob.y - BLOB_PAD_PX, 0),
                w=blob.w + 2 * BLOB_PAD_PX,
                h=blob.h + 2 * BLOB_PAD_PX,
            )
        )
    return regions


def detect_video(frames: list[BGRImage], std_max: float = TEMPORAL_STD_MAX) -> list[Region]:
    """Detect static overlay regions across sampled video frames.

    Args:
        frames: Evenly sampled BGR frames (>= 2 recommended).
        std_max: Per-pixel temporal std at or below this counts as static.

    Returns:
        Candidate Regions, possibly empty.
    """
    if len(frames) < 2:
        return []
    height, width = frames[0].shape[0], frames[0].shape[1]
    frame_area = width * height
    stack = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) for frame in frames])
    temporal_std = stack.std(axis=0)
    median_frame = np.median(stack, axis=0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    gradient = cv2.morphologyEx(median_frame, cv2.MORPH_GRADIENT, kernel)
    _, edges = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    static = (temporal_std <= std_max).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(static, edges)
    candidate = cv2.dilate(candidate, np.ones((7, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_frac = w * h / frame_area
        if not VIDEO_MIN_AREA_FRAC <= area_frac <= VIDEO_MAX_AREA_FRAC:
            continue
        fill = cv2.contourArea(contour) / (w * h)
        if fill < VIDEO_MIN_FILL:
            continue
        # 位置先验：真水印贴边/贴角（台标、角标、名字条），悬在画面中央的
        # 静止块几乎总是场景内容（静止背景、缓慢动画的稳定区）。
        cx, cy = x + w / 2, y + h / 2
        in_center_x = VIDEO_CENTER_REJECT * width < cx < (1 - VIDEO_CENTER_REJECT) * width
        in_center_y = VIDEO_CENTER_REJECT * height < cy < (1 - VIDEO_CENTER_REJECT) * height
        if in_center_x and in_center_y:
            continue
        regions.append(
            Region(
                x=max(int(x) - BLOB_PAD_PX, 0),
                y=max(int(y) - BLOB_PAD_PX, 0),
                w=int(w) + 2 * BLOB_PAD_PX,
                h=int(h) + 2 * BLOB_PAD_PX,
            )
        )
    return regions


def annotate(image: BGRImage, regions: list[Region]) -> BGRImage:
    """Draw detection boxes on a copy of the image for previewing.

    Args:
        image: BGR image.
        regions: Regions to draw.

    Returns:
        Annotated copy; the input is not modified.
    """
    canvas = image.copy()
    for region in regions:
        top_left = (region.x, region.y)
        bottom_right = (region.x + region.w, region.y + region.h)
        cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(
            canvas,
            str(region),
            (region.x, max(region.y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas

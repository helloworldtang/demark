"""Shared fixtures: synthetic watermarked images and videos."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from demark.ffmpeg_utils import ffmpeg_path

W, H = 640, 480
WATERMARK_X, WATERMARK_Y = 540, 430  # bottom-right corner, like an AI badge
VIDEO_WM_X = 425  # video patch is ~200px wide; keep it fully inside the frame


@pytest.fixture
def watermarked_image() -> np.ndarray:
    """A gradient photo-like image with a text watermark in the corner."""
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for x in range(W):
        image[:, x] = (x * 255 // W, 128, 255 - x * 255 // W)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.putText(
        image,
        "AI GEN",
        (WATERMARK_X, WATERMARK_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def watermarked_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, tuple[int, int, int, int]]:
    """A short mandelbrot video with a static corner watermark.

    brew 的 ffmpeg 8.1.1 未编入 drawtext（No such filter），所以用 cv2 生成
    水印 PNG 再走 overlay 滤镜，效果等价且不依赖 freetype。背景用 mandelbrot
    动画源——整幅画面持续运动，唯一静止的就是水印，贴近真实网课/网络视频。

    Returns:
        (path, (x, y, w, h)) — the watermark rectangle in frame coordinates.
    """
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not installed")
    tmp = tmp_path_factory.mktemp("video")
    path = tmp / "marked.mp4"
    # 60x220 RGBA patch, white text with baseline y=40.
    patch = np.zeros((60, 220, 4), dtype=np.uint8)
    cv2.putText(patch, "WATERMARK", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255, 255), 2, cv2.LINE_AA)
    patch_path = tmp / "wm.png"
    cv2.imwrite(str(patch_path), patch)
    subprocess.run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"mandelbrot=s={W}x{H}:rate=10",
            "-i",
            str(patch_path),
            "-filter_complex",
            f"[0:v][1:v]overlay=x={VIDEO_WM_X}:y={WATERMARK_Y}",
            "-pix_fmt",
            "yuv420p",
            "-t",
            "3",  # 输出侧限时：mandelbrot 源无限流，不限制会跑飞
            str(path),
        ],
        check=True,
    )
    # Glyphs occupy roughly patch y 15..40 at the overlay position.
    return str(path), (VIDEO_WM_X, WATERMARK_Y + 15, 190, 28)

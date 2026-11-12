"""ffmpeg delogo pipeline for videos."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from demark import ffmpeg_utils
from demark.errors import DemarkError
from demark.image_ops import BGRImage
from demark.region import Region


def build_delogo_filter(regions: list[Region], width: int, height: int) -> str:
    """Build an ffmpeg filter chain of delogo filters, one per region.

    ffmpeg ``delogo`` suppresses a TV-station-style logo by interpolating the
    pixels just outside the rectangle, and requires the rectangle to stay at
    least 1px away from the frame edges — hence the clamping here.

    Args:
        regions: Watermark rectangles in frame pixel coordinates.
        width: Frame width.
        height: Frame height.

    Returns:
        Filter graph string such as ``delogo=x=10:y=10:w=80:h=30,delogo=...``.

    Raises:
        DemarkError: If no region survives clamping.
    """
    parts = []
    for region in regions:
        clamped = region.clamped_to(width, height, inset=1)
        if clamped.x + clamped.w >= width or clamped.y + clamped.h >= height:
            # Still touching the far edge after clamping means the region is
            # too large for delogo; shrink it by one pixel on that side.
            clamped = Region(
                x=clamped.x,
                y=clamped.y,
                w=min(clamped.w, width - 1 - clamped.x),
                h=min(clamped.h, height - 1 - clamped.y),
            )
        if clamped.w < 1 or clamped.h < 1:
            continue
        parts.append(f"delogo=x={clamped.x}:y={clamped.y}:w={clamped.w}:h={clamped.h}")
    if not parts:
        raise DemarkError("no usable delogo region remains after clamping")
    return ",".join(parts)


def remove_video_watermarks(
    input_path: Path,
    regions: list[Region],
    output_path: Path,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Run ffmpeg delogo over a video, copying audio untouched.

    Args:
        input_path: Source video.
        regions: Watermark rectangles.
        output_path: Destination video.
        crf: libx264 quality (lower = better; 18 is visually lossless-ish).
        preset: libx264 speed/size preset.

    Returns:
        The output path.

    Raises:
        DemarkError: On probing or ffmpeg failure.
    """
    width, height = ffmpeg_utils.video_size(input_path)
    video_filter = build_delogo_filter(regions, width, height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_utils.run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    return output_path


def sample_frames(input_path: Path, count: int = 30) -> list[BGRImage]:
    """Read up to ``count`` frames evenly spaced across a video.

    Args:
        input_path: Source video.
        count: Target number of samples.

    Returns:
        BGR frames (possibly fewer if the video is short).

    Raises:
        DemarkError: If the video cannot be opened.
    """
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise DemarkError(f"cannot open video: {input_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        positions = np.linspace(0, total - 1, num=min(count, total), dtype=int)
        frames = []
        for pos in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def read_first_frame(input_path: Path) -> BGRImage:
    """Read the first frame of a video.

    Args:
        input_path: Source video.

    Returns:
        BGR frame.

    Raises:
        DemarkError: If the video cannot be opened or has no frames.
    """
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise DemarkError(f"cannot open video: {input_path}")
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise DemarkError(f"no frames in video: {input_path}")
        return frame
    finally:
        capture.release()


def has_audio_stream(input_path: Path) -> bool:
    """Check whether a video file carries an audio stream.

    Args:
        input_path: Source video.

    Returns:
        True if at least one audio stream exists.
    """
    cmd = [
        ffmpeg_utils.ffprobe_path(),
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return bool(proc.stdout.strip())

"""Locate and invoke the ffmpeg/ffprobe system binaries."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from demark.errors import DemarkError

_INSTALL_HINT = (
    "ffmpeg/ffprobe are system dependencies, not pip packages. "
    "Install them first: macOS `brew install ffmpeg`, "
    "Ubuntu `sudo apt install ffmpeg`, Windows https://ffmpeg.org/download.html"
)


def _locate(binary: str) -> str:
    """Return the absolute path of a binary, or raise DemarkError.

    Args:
        binary: Binary name such as ``ffmpeg`` or ``ffprobe``.

    Returns:
        Absolute path to the binary.

    Raises:
        DemarkError: If the binary is not on PATH.
    """
    found = shutil.which(binary)
    if found is None:
        raise DemarkError(f"{binary} not found on PATH. {_INSTALL_HINT}")
    return found


def ffmpeg_path() -> str:
    """Return the absolute path of the ``ffmpeg`` binary.

    Raises:
        DemarkError: If ffmpeg is not installed.
    """
    return _locate("ffmpeg")


def ffprobe_path() -> str:
    """Return the absolute path of the ``ffprobe`` binary.

    Raises:
        DemarkError: If ffprobe is not installed.
    """
    return _locate("ffprobe")


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with the given argument list, raising on failure.

    Args:
        args: Arguments after the binary name, e.g. ``["-i", "in.mp4", ...]``.

    Raises:
        DemarkError: If ffmpeg exits non-zero; the message carries the tail
            of stderr for diagnosis.
    """
    cmd = [ffmpeg_path(), *args, "-hide_banner", "-loglevel", "error", "-y"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().splitlines()[-5:]
        raise DemarkError(f"ffmpeg failed (exit {proc.returncode}): {' | '.join(stderr_tail)}")


def probe_video(path: Path) -> dict[str, Any]:
    """Probe a video file and return its first video stream as a dict.

    Args:
        path: Video file path.

    Returns:
        The stream dict from ``ffprobe -show_streams`` (width/height/codec...).

    Raises:
        DemarkError: If ffprobe fails or no video stream exists.
    """
    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise DemarkError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams:
        raise DemarkError(f"no video stream found in {path}")
    return dict(streams[0])


def video_size(path: Path) -> tuple[int, int]:
    """Return the (width, height) of the first video stream.

    Args:
        path: Video file path.

    Raises:
        DemarkError: If probing fails or dimensions are missing.
    """
    stream = probe_video(path)
    width, height = stream.get("width"), stream.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise DemarkError(f"cannot determine frame size of {path}")
    return width, height

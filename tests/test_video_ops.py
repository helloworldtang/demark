"""Tests for the ffmpeg delogo pipeline."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from demark.errors import DemarkError
from demark.region import Region
from demark.video_ops import build_delogo_filter, remove_video_watermarks, sample_frames

ffmpeg_missing = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class TestBuildDelogoFilter:
    def test_single_region(self) -> None:
        chain = build_delogo_filter([Region(10, 10, 80, 30)], 640, 480)
        assert chain == "delogo=x=10:y=10:w=80:h=30"

    def test_multiple_regions_chain(self) -> None:
        chain = build_delogo_filter([Region(10, 10, 80, 30), Region(500, 400, 100, 50)], 640, 480)
        assert chain == "delogo=x=10:y=10:w=80:h=30,delogo=x=500:y=400:w=100:h=50"

    def test_clamps_to_one_pixel_inset(self) -> None:
        chain = build_delogo_filter([Region(0, 0, 640, 480)], 640, 480)
        assert "x=1" in chain and "y=1" in chain
        # w/h must fit within width-1 / height-1
        match = re.search(r"w=(\d+):h=(\d+)", chain)
        assert match is not None
        w, h = int(match.group(1)), int(match.group(2))
        assert 1 + w <= 639
        assert 1 + h <= 479

    def test_empty_regions_raise(self) -> None:
        with pytest.raises(DemarkError, match="no usable delogo region"):
            build_delogo_filter([], 640, 480)


@ffmpeg_missing
class TestEndToEnd:
    def test_delogo_produces_playable_output(
        self, watermarked_video: tuple[str, tuple[int, int, int, int]], tmp_path: Path
    ) -> None:
        src, (wx, wy, ww, wh) = watermarked_video
        out = tmp_path / "clean.mp4"
        remove_video_watermarks(Path(src), [Region(wx, wy, ww, wh)], out)
        assert out.is_file() and out.stat().st_size > 0

        # Output must be re-readable and have the same geometry.
        frames = sample_frames(out, 5)
        assert len(frames) == 5
        assert frames[0].shape[:2] == (480, 640)

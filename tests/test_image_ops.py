"""Tests for the image inpainting pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from demark.errors import DemarkError
from demark.image_ops import build_mask, inpaint_regions, make_compare
from demark.region import Region


class TestBuildMask:
    def test_paints_regions_white(self) -> None:
        mask = build_mask((100, 100, 3), [Region(x=10, y=10, w=20, h=20)], dilate_px=0)
        assert mask.shape == (100, 100)
        assert mask[20, 20] == 255
        assert mask[5, 5] == 0

    def test_dilation_covers_fringe(self) -> None:
        mask = build_mask((100, 100, 3), [Region(x=50, y=50, w=10, h=10)], dilate_px=3)
        assert mask[47, 47] == 255
        assert mask[46, 46] == 0

    def test_clips_at_frame_edge(self) -> None:
        mask = build_mask((100, 100, 3), [Region(x=90, y=90, w=20, h=20)], dilate_px=3)
        assert mask[99, 99] == 255  # no out-of-bounds indexing


class TestInpaint:
    def test_changes_watermark_area_only(self) -> None:
        rng = np.random.default_rng(11)
        image = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        region = Region(x=150, y=150, w=40, h=40)
        result = inpaint_regions(image, [region])
        assert not np.array_equal(result[160:180, 160:180], image[160:180, 160:180]), "watermark area must change"
        assert np.array_equal(result[0:100, 0:100], image[0:100, 0:100]), "outside area must be untouched"

    def test_unknown_method_raises(self) -> None:
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(DemarkError, match="unknown inpaint method"):
            inpaint_regions(image, [Region(0, 0, 10, 10)], method="magic")


class TestCompare:
    def test_side_by_side_shape(self) -> None:
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        result = np.full((50, 100, 3), 200, dtype=np.uint8)
        compare = make_compare(image, result)
        assert compare.shape == (50, 100 + 3 + 100, 3)

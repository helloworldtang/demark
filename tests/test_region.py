"""Tests for Region parsing and clamping."""

from __future__ import annotations

import pytest

from demark.region import Region


class TestParse:
    def test_parses_four_integers(self) -> None:
        assert Region.parse("12,34,120,40") == Region(x=12, y=34, w=120, h=40)

    def test_tolerates_spaces(self) -> None:
        assert Region.parse(" 1 , 2 , 3 , 4 ") == Region(x=1, y=2, w=3, h=4)

    @pytest.mark.parametrize("bad", ["", "1,2,3", "1,2,3,4,5", "a,b,c,d", "-1,2,3,4"])
    def test_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(ValueError, match="invalid region"):
            Region.parse(bad)

    def test_rejects_zero_size(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            Region.parse("1,2,0,4")

    def test_rejects_negative_origin(self) -> None:
        with pytest.raises(ValueError, match="x/y must be >= 0"):
            Region(x=-1, y=0, w=10, h=10)


class TestClamp:
    def test_noop_when_inside(self) -> None:
        region = Region(10, 10, 50, 50).clamped_to(640, 480)
        assert (region.x, region.y, region.w, region.h) == (10, 10, 50, 50)

    def test_clamps_overflowing_region(self) -> None:
        region = Region(100, 100, 600, 500).clamped_to(640, 480)
        assert region.x + region.w <= 640
        assert region.y + region.h <= 480

    def test_inset_keeps_region_off_edges(self) -> None:
        region = Region(0, 0, 640, 480).clamped_to(640, 480, inset=1)
        assert region.x >= 1
        assert region.y >= 1
        assert region.x + region.w <= 639
        assert region.y + region.h <= 479

    def test_str_roundtrip(self) -> None:
        assert str(Region(1, 2, 3, 4)) == "1,2,3,4"

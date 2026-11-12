"""Region primitives shared by detection and removal pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass

_REGION_RE = re.compile(r"^(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)$")


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle in pixel coordinates.

    Attributes:
        x: Left edge (pixels from the left of the frame).
        y: Top edge (pixels from the top of the frame).
        w: Width in pixels (must be >= 1).
        h: Height in pixels (must be >= 1).
    """

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w < 1 or self.h < 1:
            raise ValueError(f"region width/height must be >= 1, got {self.w}x{self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"region x/y must be >= 0, got ({self.x}, {self.y})")

    @classmethod
    def parse(cls, spec: str) -> Region:
        """Parse a ``X,Y,W,H`` string such as ``"12,34,120,40"``.

        Args:
            spec: Four comma-separated non-negative integers.

        Returns:
            The parsed Region.

        Raises:
            ValueError: If the string is malformed or values are invalid.
        """
        match = _REGION_RE.match(spec.strip())
        if match is None:
            raise ValueError(f"invalid region {spec!r}: expected X,Y,W,H (e.g. 12,34,120,40)")
        x, y, w, h = (int(part) for part in match.groups())
        return cls(x=x, y=y, w=w, h=h)

    def clamped_to(self, width: int, height: int, inset: int = 0) -> Region:
        """Return the region clipped to a frame of the given size.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            inset: Shrink the usable area by this many pixels on every side
                (ffmpeg ``delogo`` requires the rectangle not to touch edges).

        Returns:
            A Region fully inside ``[inset, width-inset) x [inset, height-inset)``.
            May be zero-sized if nothing remains, in which case callers should
            drop the region.
        """
        max_w = width - 2 * inset
        max_h = height - 2 * inset
        if max_w < 1 or max_h < 1:
            raise ValueError(f"frame {width}x{height} too small for inset {inset}")
        new_x = min(max(self.x, inset), inset + max_w - 1)
        new_y = min(max(self.y, inset), inset + max_h - 1)
        new_w = min(self.w, inset + max_w - new_x)
        new_h = min(self.h, inset + max_h - new_y)
        return Region(x=new_x, y=new_y, w=max(new_w, 1), h=max(new_h, 1))

    def __str__(self) -> str:
        return f"{self.x},{self.y},{self.w},{self.h}"

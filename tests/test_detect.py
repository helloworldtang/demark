"""Tests for heuristic watermark detection."""

from __future__ import annotations

import cv2
import numpy as np

from demark.detect import detect_image, detect_video

WATERMARK_X, WATERMARK_Y = 540, 430  # keep in sync with conftest


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _bbox_recall(detected: tuple[int, int, int, int], truth: tuple[int, int, int, int]) -> float:
    """Fraction of the truth box covered by the detected box.

    Temporal detection tends to merge the watermark with adjacent static
    scenery into one big blob; recall measures "did we cover the mark at all"
    where IoU would unfairly punish that merge.
    """
    ax, ay, aw, ah = detected
    bx, by, bw, bh = truth
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (bw * bh) if bw * bh else 0.0


class TestDetectImage:
    def test_finds_corner_text_watermark(self, watermarked_image: np.ndarray) -> None:
        regions = detect_image(watermarked_image)
        assert regions, "auto-detect should find the corner watermark"
        # putText 的 y 是基线（文字底部），字形实际分布在基线上方 ~35px。
        truth = (WATERMARK_X - 6, WATERMARK_Y - 32, 96, 40)
        best = max(regions, key=lambda r: _bbox_iou((r.x, r.y, r.w, r.h), truth))
        assert _bbox_iou((best.x, best.y, best.w, best.h), truth) > 0.3

    def test_clean_image_yields_nothing(self) -> None:
        rng = np.random.default_rng(7)
        noisy = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        for _ in range(60):  # scramble with blur+noise so no text-like blobs survive
            noisy = cv2.GaussianBlur(noisy, (11, 11), 0)
            noisy = rng.integers(0, 256, noisy.shape, dtype=np.uint8)
        assert detect_image(noisy) == []


class TestDetectVideo:
    def test_finds_static_overlay(self, watermarked_video: tuple[str, tuple[int, int, int, int]]) -> None:
        from demark.video_ops import sample_frames

        path, (wx, wy, ww, wh) = watermarked_video
        frames = sample_frames(path, 30)
        assert len(frames) >= 10
        regions = detect_video(frames)
        assert regions, "auto-detect should find the static watermark"
        boxes = [(r.x, r.y, r.w, r.h) for r in regions]
        recall = max(_bbox_recall(box, (wx, wy, ww, wh)) for box in boxes)
        assert recall > 0.5, f"best recall {recall:.2f} for truth {(wx, wy, ww, wh)} vs {boxes}"

    def test_needs_multiple_frames(self) -> None:
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        assert detect_video([frame]) == []

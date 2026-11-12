"""Command-line entry point for demark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from demark import __version__, detect, image_ops, video_ops
from demark.errors import DemarkError
from demark.image_ops import BGRImage
from demark.region import Region

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _default_output(path: Path) -> Path:
    """Return ``<stem>.demark<ext>`` next to the input."""
    return path.with_name(f"{path.stem}.demark{path.suffix}")


def _parse_regions(specs: list[str]) -> list[Region]:
    """Parse ``-r`` specs, turning parse errors into DemarkError."""
    try:
        return [Region.parse(spec) for spec in specs]
    except ValueError as exc:
        raise DemarkError(str(exc)) from exc


def _select_regions_interactively(image: BGRImage) -> list[Region]:
    """Let the user drag rectangles in a GUI window (ESC to finish)."""
    rois = cv2.selectROIs("demark: drag boxes, ENTER to confirm each, ESC when done", image, False, False)
    return [Region(x=int(x), y=int(y), w=int(w), h=int(h)) for x, y, w, h in rois]


def _collect_image_regions(image: BGRImage, args: argparse.Namespace) -> list[Region]:
    """Combine manual, interactive, and auto-detected regions for one image."""
    regions = list(_parse_regions(getattr(args, "region", None) or []))
    if getattr(args, "select", False):
        regions.extend(_select_regions_interactively(image))
    if getattr(args, "auto", False):
        regions.extend(detect.detect_image(image))
    # Deduplicate exact duplicates while keeping order.
    seen: set[tuple[int, int, int, int]] = set()
    unique = []
    for region in regions:
        key = (region.x, region.y, region.w, region.h)
        if key not in seen:
            seen.add(key)
            unique.append(region)
    return unique


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def cmd_img(args: argparse.Namespace) -> int:
    """Handle ``demark img``."""
    inputs = [Path(p) for p in args.inputs]
    if len(inputs) > 1 and (args.output or args.in_place):
        raise DemarkError("-o/--in-place only work with a single input")
    for path in inputs:
        image = image_ops.load_image(path)
        regions = _collect_image_regions(image, args)
        if not regions:
            raise DemarkError(f"no watermark region for {path.name}: pass -r, --select, or --auto")
        result = image_ops.inpaint_regions(image, regions, method=args.method)
        if args.compare:
            compare_path = Path(args.compare)
            image_ops.save_image(compare_path, image_ops.make_compare(image, result))
            print(f"compare  -> {compare_path}")
        out = path if args.in_place else Path(args.output) if args.output else _default_output(path)
        image_ops.save_image(out, result)
        print(f"{path.name}: {len(regions)} region(s) inpainted -> {out}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    """Handle ``demark video``."""
    input_path = Path(args.input)
    if not input_path.is_file():
        raise DemarkError(f"video not found: {input_path}")
    regions = _parse_regions(args.region or [])
    if args.auto:
        frames = video_ops.sample_frames(input_path, detect.VIDEO_SAMPLES)
        auto_regions = detect.detect_video(frames)
        if not auto_regions:
            print("auto-detect found nothing; continuing with manual regions only", file=sys.stderr)
        regions.extend(auto_regions)
    if not regions:
        raise DemarkError("no watermark region: pass -r or --auto")

    if args.preview_frame:
        first = video_ops.read_first_frame(input_path)
        preview = image_ops.make_compare(first, image_ops.inpaint_regions(first, regions))
        preview_path = Path(args.preview_frame)
        image_ops.save_image(preview_path, preview)
        print(f"preview  -> {preview_path} (left: original, right: after delogo-like inpaint)")

    out = Path(args.output) if args.output else _default_output(input_path)
    video_ops.remove_video_watermarks(input_path, regions, out, crf=args.crf, preset=args.preset)
    print(f"{input_path.name}: {len(regions)} region(s) delogo -> {out}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Handle ``demark detect``: dry-run detection with an annotated preview."""
    path = Path(args.input)
    if path.suffix.lower() in IMAGE_EXTS:
        image = image_ops.load_image(path)
        regions = detect.detect_image(image)
        annotated = detect.annotate(image, regions)
    elif _is_video(path):
        frames = video_ops.sample_frames(path, detect.VIDEO_SAMPLES)
        regions = detect.detect_video(frames)
        if frames:
            annotated = detect.annotate(frames[0], regions)
        else:
            raise DemarkError(f"no frames could be sampled from {path}")
    else:
        raise DemarkError(
            f"unrecognized file type: {path.suffix!r} (image: {sorted(IMAGE_EXTS)}, video: {sorted(VIDEO_EXTS)})"
        )

    for region in regions:
        print(f"{path.name}: -r {region}")
    if not regions:
        print(f"{path.name}: nothing auto-detected; specify regions manually with -r X,Y,W,H")
        return 1
    out = path.with_name(f"{path.stem}.detect{'.png' if _is_video(path) else path.suffix}")
    image_ops.save_image(out, annotated)
    print(f"annotated -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(prog="demark", description="去水印 CLI：图片 inpaint + 视频 delogo")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    img = sub.add_parser("img", help="remove watermarks from image(s)")
    img.add_argument("inputs", nargs="+", help="image file(s)")
    img.add_argument(
        "-r", "--region", action="append", default=[], metavar="X,Y,W,H", help="watermark rectangle, repeatable"
    )
    img.add_argument("--auto", action="store_true", help="auto-detect corner/bottom marks")
    img.add_argument("--select", action="store_true", help="interactively drag boxes (GUI)")
    img.add_argument("--method", choices=["telea", "ns"], default="telea", help="inpaint algorithm (default telea)")
    img.add_argument("-o", "--output", help="output path (single input only)")
    img.add_argument("--in-place", action="store_true", help="overwrite the input file")
    img.add_argument("--compare", metavar="PATH", help="also write a side-by-side before/after image")
    img.set_defaults(func=cmd_img)

    video = sub.add_parser("video", help="remove watermarks from a video")
    video.add_argument("input", help="video file")
    video.add_argument(
        "-r", "--region", action="append", default=[], metavar="X,Y,W,H", help="watermark rectangle, repeatable"
    )
    video.add_argument("--auto", action="store_true", help="auto-detect static overlay regions")
    video.add_argument("--crf", type=int, default=18, help="libx264 crf (default 18)")
    video.add_argument("--preset", default="medium", help="libx264 preset (default medium)")
    video.add_argument("-o", "--output", help="output path")
    video.add_argument(
        "--preview-frame", metavar="PATH", help="write a first-frame before/after preview before the full run"
    )
    video.set_defaults(func=cmd_video)

    det = sub.add_parser("detect", help="dry-run: show auto-detected regions")
    det.add_argument("input", help="image or video file")
    det.set_defaults(func=cmd_detect)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DemarkError as exc:
        print(f"demark: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

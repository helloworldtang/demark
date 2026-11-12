"""CLI-level tests exercising the argparse wiring end to end."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from demark.cli import main
from demark.image_ops import load_image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


@pytest.fixture
def clean_image_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    path = tmp_path / "shot.png"
    save_image(path, rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    return path


class TestVersion:
    def test_prints_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "demark 0." in capsys.readouterr().out


class TestImgCommand:
    def test_inpaints_manual_region(self, clean_image_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.png"
        code = main(["img", str(clean_image_file), "-r", "100,100,80,40", "-o", str(out)])
        assert code == 0
        original = load_image(clean_image_file)
        result = load_image(out)
        assert not np.array_equal(result[110:130, 110:170], original[110:130, 110:170])

    def test_no_region_is_an_error(self, clean_image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["img", str(clean_image_file)])
        assert code == 1
        assert "no watermark region" in capsys.readouterr().err

    def test_bad_region_spec_is_an_error(self, clean_image_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["img", str(clean_image_file), "-r", "oops"])
        assert code == 1
        assert "invalid region" in capsys.readouterr().err

    def test_multi_input_rejects_output(self, clean_image_file: Path, tmp_path: Path) -> None:
        other = tmp_path / "b.png"
        save_image(other, np.zeros((480, 640, 3), dtype=np.uint8))
        code = main(["img", str(clean_image_file), str(other), "-o", str(tmp_path / "x.png")])
        assert code == 1


class TestDetectCommand:
    def test_reports_regions_for_watermarked_image(self, tmp_path: Path) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(640):
            image[:, x] = (x * 255 // 640, 128, 255 - x * 255 // 640)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.putText(image, "REC", (540, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        path = tmp_path / "marked.png"
        save_image(path, image)
        code = main(["detect", str(path)])
        assert code == 0
        annotated = tmp_path / "marked.detect.png"
        assert annotated.is_file()

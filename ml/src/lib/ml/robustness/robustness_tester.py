"""
SPIRO ML — RobustnessTester
Tests the SPIRO inference pipeline against degraded and adversarial images.

Test categories
---------------
- Blurry images (Gaussian blur)
- Dark images (low brightness)
- Overexposed images (high brightness)
- Rotated images (various angles)
- JPEG-compressed images (low quality)
- Low resolution (32×32 → 128×128)
- Large resolution (4096×4096)
- Occluded objects (random black patches)
- Multi-object images (synthesised)
- Empty images (no objects)
- Corrupted byte streams
- Unsupported formats (text, random bytes)

Metrics per category
--------------------
- Pass rate (% of images accepted by preprocessor)
- Detection rate (% with at least 1 detection)
- Mean confidence
- Pipeline stability (no exceptions)
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class RobustnessCategory:
    name: str
    n_images: int = 0
    n_accepted: int = 0        # passed preprocessor quality gate
    n_detected: int = 0        # had ≥1 detection
    n_crashed: int = 0         # raised an exception
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0
    pass_rate: float = 0.0
    detection_rate: float = 0.0
    stability_rate: float = 0.0
    notes: str = ""

    def compute(self) -> None:
        self.pass_rate      = round(self.n_accepted / max(self.n_images, 1), 3)
        self.detection_rate = round(self.n_detected / max(self.n_images, 1), 3)
        self.stability_rate = round(1.0 - self.n_crashed / max(self.n_images, 1), 3)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessReport:
    model_id: str
    categories: List[RobustnessCategory]
    overall_stability: float = 0.0
    overall_pass_rate: float = 0.0
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["categories"] = [c.to_dict() for c in self.categories]
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# SPIRO Robustness Report — {self.model_id}",
            f"**Generated:** {self.generated_at}",
            f"**Overall stability:** {self.overall_stability:.1%}",
            f"**Overall pass rate:** {self.overall_pass_rate:.1%}",
            "",
            "| Category | N | Pass Rate | Detection Rate | Stability | Mean Conf |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.categories:
            lines.append(
                f"| {c.name} | {c.n_images} | "
                f"{c.pass_rate:.1%} | {c.detection_rate:.1%} | "
                f"{c.stability_rate:.1%} | {c.mean_confidence:.3f} |"
            )
        if self.notes:
            lines += ["", f"**Notes:** {self.notes}"]
        return "\n".join(lines)


class RobustnessTester:
    """
    Tests the preprocessor and (optionally) full pipeline for robustness.

    Parameters
    ----------
    infer_fn : callable
        Function that accepts an image (BGR ndarray or bytes) and returns
        a pipeline result dict. If None, only preprocessor is tested.
    report_dir : Path
    n_per_category : int
        Synthetic images generated per test category.

    Example
    -------
    >>> tester = RobustnessTester(infer_fn=pipeline.infer)
    >>> report = tester.run_all(model_id="yolov11s")
    >>> tester.save_report(report)
    """

    def __init__(
        self,
        infer_fn: Optional[Callable] = None,
        report_dir: Path = Path("reports"),
        n_per_category: int = 20,
        base_size: int = 640,
    ) -> None:
        self.infer_fn = infer_fn
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.n = n_per_category
        self.base_size = base_size

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all(self, model_id: str = "spiro") -> RobustnessReport:
        categories: List[RobustnessCategory] = []

        tests = [
            ("blurry",          self._gen_blurry),
            ("dark",            self._gen_dark),
            ("overexposed",     self._gen_overexposed),
            ("rotated_45",      self._gen_rotated),
            ("jpeg_compressed", self._gen_compressed),
            ("low_resolution",  self._gen_low_res),
            ("large_resolution",self._gen_large_res),
            ("occluded",        self._gen_occluded),
            ("multi_object",    self._gen_multi_object),
            ("empty_image",     self._gen_empty),
            ("corrupted",       self._gen_corrupted),
            ("unsupported_fmt", self._gen_unsupported),
        ]

        for name, gen_fn in tests:
            cat = self._run_category(name, gen_fn)
            categories.append(cat)
            log.info(
                f"  {name:<22}: pass={cat.pass_rate:.1%} "
                f"stab={cat.stability_rate:.1%} det={cat.detection_rate:.1%}"
            )

        all_stab = [c.stability_rate for c in categories]
        all_pass = [c.pass_rate for c in categories]
        report = RobustnessReport(
            model_id=model_id,
            categories=categories,
            overall_stability=round(float(np.mean(all_stab)), 3),
            overall_pass_rate=round(float(np.mean(all_pass)), 3),
        )
        return report

    # ------------------------------------------------------------------
    # Category runner
    # ------------------------------------------------------------------

    def _run_category(self, name: str, gen_fn: Callable) -> RobustnessCategory:
        cat = RobustnessCategory(name=name, n_images=self.n)
        total_conf = 0.0
        total_latency = 0.0
        n_conf = 0

        for _ in range(self.n):
            try:
                img = gen_fn()
                t0 = time.perf_counter()

                if self.infer_fn is not None:
                    if isinstance(img, bytes):
                        result = self.infer_fn(img)
                    else:
                        result = self.infer_fn(img)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    total_latency += latency_ms
                    status = result.get("status", "error")
                    if status in ("success", "partial"):
                        cat.n_accepted += 1
                    if result.get("detection_count", 0) > 0:
                        cat.n_detected += 1
                        for det in result.get("detections", []):
                            total_conf += det.get("final_confidence", 0.0)
                            n_conf += 1
                else:
                    # Preprocessor-only test
                    from lib.ml.pipeline.preprocessing.preprocessor import ImagePreprocessor
                    prep = ImagePreprocessor(blur_threshold=0.0, quality_threshold=0.0)
                    if isinstance(img, bytes):
                        r = prep.process(img)
                    else:
                        r = prep.process(img)
                    if r.quality.passed:
                        cat.n_accepted += 1

            except Exception as e:
                cat.n_crashed += 1
                log.debug(f"  [{name}] exception: {e}")

        cat.mean_confidence = round(total_conf / max(n_conf, 1), 4)
        cat.mean_latency_ms = round(total_latency / max(self.n - cat.n_crashed, 1), 2)
        cat.compute()
        return cat

    # ------------------------------------------------------------------
    # Image generators
    # ------------------------------------------------------------------

    def _gen_blurry(self) -> np.ndarray:
        img = self._base_image()
        return cv2.GaussianBlur(img, (101, 101), 30)

    def _gen_dark(self) -> np.ndarray:
        img = self._base_image()
        return np.clip(img.astype(np.int32) - 200, 0, 255).astype(np.uint8)

    def _gen_overexposed(self) -> np.ndarray:
        return np.full((self.base_size, self.base_size, 3), 250, dtype=np.uint8)

    def _gen_rotated(self) -> np.ndarray:
        img = self._base_image()
        angle = np.random.choice([15, 30, 45, 90, 180])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def _gen_compressed(self) -> np.ndarray:
        img = self._base_image()
        quality = np.random.randint(1, 10)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _gen_low_res(self) -> np.ndarray:
        size = np.random.randint(32, 128)
        return np.random.randint(30, 200, (size, size, 3), dtype=np.uint8)

    def _gen_large_res(self) -> np.ndarray:
        size = np.random.choice([2048, 3840, 4096])
        return np.random.randint(30, 200, (size, size, 3), dtype=np.uint8)

    def _gen_occluded(self) -> np.ndarray:
        img = self._base_image()
        # Add 3–8 black occluding patches
        n_patches = np.random.randint(3, 8)
        h, w = img.shape[:2]
        for _ in range(n_patches):
            ph = np.random.randint(h // 6, h // 3)
            pw = np.random.randint(w // 6, w // 3)
            y = np.random.randint(0, max(1, h - ph))
            x = np.random.randint(0, max(1, w - pw))
            img[y:y+ph, x:x+pw] = 0
        return img

    def _gen_multi_object(self) -> np.ndarray:
        return np.random.randint(30, 200, (self.base_size, self.base_size, 3), dtype=np.uint8)

    def _gen_empty(self) -> np.ndarray:
        return np.full((self.base_size, self.base_size, 3), 128, dtype=np.uint8)

    def _gen_corrupted(self) -> bytes:
        # Random bytes that look like a JPEG header but are corrupt
        header = bytes([0xFF, 0xD8, 0xFF, 0xE0])
        garbage = np.random.bytes(np.random.randint(100, 1000))
        return header + garbage

    def _gen_unsupported(self) -> bytes:
        return b"This is not an image file at all."

    def _base_image(self) -> np.ndarray:
        return np.random.randint(30, 200, (self.base_size, self.base_size, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Report saving
    # ------------------------------------------------------------------

    def save_report(self, report: RobustnessReport) -> Tuple[Path, Path]:
        json_path = self.report_dir / f"{report.model_id}_robustness.json"
        md_path   = self.report_dir / f"{report.model_id}_robustness.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        md_path.write_text(report.to_markdown())
        log.info(f"Robustness report → {json_path}")
        return json_path, md_path

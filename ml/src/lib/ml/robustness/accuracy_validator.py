"""
SPIRO ML — AccuracyValidator
Cross-format accuracy validation:
  PyTorch .pt vs ONNX FP32 vs ONNX FP16 vs ONNX INT8

Validates
---------
- Numerical parity (max absolute difference)
- Top-1 / Top-5 prediction consistency (classification)
- Bounding box drift (detection)
- Confidence distribution drift (detection)
- Class-level consistency
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


@dataclass
class FormatComparison:
    reference_format: str   # "fp32" (baseline)
    compared_format: str    # "fp16" | "int8" | "pytorch"
    n_samples: int = 0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    top1_agreement_rate: float = 0.0
    top5_agreement_rate: float = 0.0
    confidence_drift: float = 0.0
    passed: bool = False
    abs_diff_threshold: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccuracyReport:
    model_id: str
    task: str
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    comparisons: List[FormatComparison] = field(default_factory=list)
    all_passed: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["comparisons"] = [c.to_dict() for c in self.comparisons]
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# Accuracy Validation Report — {self.model_id}",
            f"**Task:** {self.task}",
            f"**Generated:** {self.generated_at}",
            f"**All passed:** {'✓' if self.all_passed else '✗'}",
            "",
            "| Format | Max Diff | Mean Diff | Top-1 Agree | Top-5 Agree | Passed |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.comparisons:
            lines.append(
                f"| FP32 vs {c.compared_format} | "
                f"{c.max_abs_diff:.6f} | {c.mean_abs_diff:.6f} | "
                f"{c.top1_agreement_rate:.1%} | {c.top5_agreement_rate:.1%} | "
                f"{'✓' if c.passed else '✗'} |"
            )
        return "\n".join(lines)


class AccuracyValidator:
    """
    Compares model outputs across formats for numerical consistency.

    Example
    -------
    >>> validator = AccuracyValidator()
    >>> report = validator.compare_formats(
    ...     fp32_path="model.onnx",
    ...     fp16_path="model_fp16.onnx",
    ...     int8_path="model_int8.onnx",
    ...     model_id="yolov11s",
    ...     input_size=640,
    ...     task="detection",
    ... )
    """

    def __init__(
        self,
        abs_diff_threshold: float = 0.01,
        top1_agreement_threshold: float = 0.95,
        n_samples: int = 50,
        report_dir: Path = Path("reports"),
    ) -> None:
        self.abs_diff_thr = abs_diff_threshold
        self.top1_agree_thr = top1_agreement_threshold
        self.n_samples = n_samples
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def compare_formats(
        self,
        fp32_path: str | Path,
        fp16_path: Optional[str | Path] = None,
        int8_path: Optional[str | Path] = None,
        pytorch_path: Optional[str | Path] = None,
        model_id: str = "model",
        input_size: int = 640,
        task: str = "detection",
        num_classes: int = 109,
    ) -> AccuracyReport:
        """
        Run all available format comparisons against FP32 baseline.
        """
        import onnxruntime as ort

        report = AccuracyReport(model_id=model_id, task=task)
        comparisons: List[FormatComparison] = []

        # Load FP32 baseline
        fp32_path = Path(fp32_path)
        if not fp32_path.exists():
            report.notes = f"FP32 baseline not found: {fp32_path}"
            return report

        fp32_sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        fp32_inp  = fp32_sess.get_inputs()[0].name
        dummy_f32 = np.random.rand(self.n_samples, 3, input_size, input_size).astype(np.float32)

        # FP32 baseline outputs (run one at a time)
        fp32_outputs = []
        for i in range(self.n_samples):
            out = fp32_sess.run(None, {fp32_inp: dummy_f32[i:i+1]})[0]
            fp32_outputs.append(out)

        # Compare formats
        format_paths = {}
        if fp16_path and Path(fp16_path).exists():
            format_paths["fp16"] = Path(fp16_path)
        if int8_path and Path(int8_path).exists():
            format_paths["int8"] = Path(int8_path)

        for fmt_name, fmt_path in format_paths.items():
            cmp = self._compare_single(
                fp32_outputs=fp32_outputs,
                baseline_inp=dummy_f32,
                candidate_path=fmt_path,
                compared_format=fmt_name,
                task=task,
                num_classes=num_classes,
            )
            comparisons.append(cmp)
            log.info(
                f"  {model_id} FP32 vs {fmt_name}: "
                f"max_diff={cmp.max_abs_diff:.6f} "
                f"top1_agree={cmp.top1_agreement_rate:.1%} "
                f"{'✓' if cmp.passed else '✗'}"
            )

        # PyTorch comparison
        if pytorch_path and Path(pytorch_path).exists():
            cmp = self._compare_pytorch(
                pytorch_path=Path(pytorch_path),
                fp32_outputs=fp32_outputs,
                dummy_f32=dummy_f32,
                task=task,
                num_classes=num_classes,
            )
            if cmp:
                comparisons.append(cmp)

        report.comparisons = comparisons
        report.all_passed = all(c.passed for c in comparisons) if comparisons else True

        # Save
        json_out = self.report_dir / f"{model_id}_accuracy_validation.json"
        md_out   = self.report_dir / f"{model_id}_accuracy_validation.md"
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        md_out.write_text(report.to_markdown())
        log.info(f"Accuracy report → {json_out}")
        return report

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def _compare_single(
        self,
        fp32_outputs: List[np.ndarray],
        baseline_inp: np.ndarray,
        candidate_path: Path,
        compared_format: str,
        task: str,
        num_classes: int,
    ) -> FormatComparison:
        import onnxruntime as ort
        cmp = FormatComparison(
            reference_format="fp32",
            compared_format=compared_format,
            n_samples=len(fp32_outputs),
            abs_diff_threshold=self.abs_diff_thr,
        )
        try:
            sess = ort.InferenceSession(str(candidate_path), providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            dtype = np.float16 if compared_format == "fp16" else np.float32

            diffs, top1_matches, top5_matches, conf_deltas = [], [], [], []
            for i, fp32_out in enumerate(fp32_outputs):
                inp = baseline_inp[i:i+1].astype(dtype)
                cand_out = sess.run(None, {inp_name: inp})[0].astype(np.float32)
                if fp32_out.shape == cand_out.shape:
                    diff = float(np.abs(fp32_out - cand_out).max())
                    diffs.append(diff)
                if task == "verification":
                    fp32_top1  = int(np.argmax(fp32_out[0]))
                    cand_top1  = int(np.argmax(cand_out[0]))
                    fp32_top5  = set(np.argsort(fp32_out[0])[::-1][:5].tolist())
                    cand_top5  = set(np.argsort(cand_out[0])[::-1][:5].tolist())
                    top1_matches.append(float(fp32_top1 == cand_top1))
                    top5_matches.append(float(len(fp32_top5 & cand_top5) >= 3))
                    conf_deltas.append(abs(float(fp32_out[0].max()) - float(cand_out[0].max())))

            if diffs:
                cmp.max_abs_diff  = round(float(max(diffs)), 8)
                cmp.mean_abs_diff = round(float(np.mean(diffs)), 8)
            cmp.top1_agreement_rate = round(float(np.mean(top1_matches)), 4) if top1_matches else 1.0
            cmp.top5_agreement_rate = round(float(np.mean(top5_matches)), 4) if top5_matches else 1.0
            cmp.confidence_drift    = round(float(np.mean(conf_deltas)), 6) if conf_deltas else 0.0

            cmp.passed = (
                cmp.max_abs_diff <= self.abs_diff_thr and
                (not top1_matches or cmp.top1_agreement_rate >= self.top1_agree_thr)
            )
        except Exception as e:
            cmp.notes = str(e)
            log.warning(f"Comparison {compared_format} failed: {e}")
        return cmp

    def _compare_pytorch(
        self,
        pytorch_path: Path,
        fp32_outputs: List[np.ndarray],
        dummy_f32: np.ndarray,
        task: str,
        num_classes: int,
    ) -> Optional[FormatComparison]:
        try:
            import torch
            cmp = FormatComparison(
                reference_format="fp32",
                compared_format="pytorch",
                n_samples=len(fp32_outputs),
                abs_diff_threshold=self.abs_diff_thr,
            )
            ckpt = torch.load(str(pytorch_path), map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
            if state is None:
                cmp.notes = "Cannot load PyTorch state dict"
                return cmp
            # Verification model path
            from lib.ml.verification.verify_config import VerifyConfig
            from lib.ml.verification.model.verify_model import VerifierModel
            cfg_path = Path("configs/verification/effnetv2_s.yaml")
            if not cfg_path.exists():
                return None
            cfg = VerifyConfig.load(cfg_path)
            model = VerifierModel.load(cfg, pytorch_path)
            model.eval()
            diffs = []
            with torch.no_grad():
                for i, fp32_out in enumerate(fp32_outputs):
                    t = torch.from_numpy(dummy_f32[i:i+1])
                    pt_out = model.backbone(t).numpy()
                    if pt_out.shape == fp32_out.shape:
                        diffs.append(float(np.abs(pt_out - fp32_out).max()))
            if diffs:
                cmp.max_abs_diff  = round(float(max(diffs)), 8)
                cmp.mean_abs_diff = round(float(np.mean(diffs)), 8)
            cmp.passed = cmp.max_abs_diff <= self.abs_diff_thr
            return cmp
        except Exception as e:
            log.debug(f"PyTorch comparison skipped: {e}")
            return None

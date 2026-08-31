"""
SPIRO ML — VerifyExporter
Exports the EfficientNetV2 verifier to ONNX with:
  - Dynamic batch axes
  - Opset 17
  - Optional graph simplification
  - ONNXRuntime verification (numerical parity check)
  - Metadata embedding
  - Verification report JSON
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from lib.ml.core.logger import get_logger
from lib.ml.verification.verify_config import VerifyConfig

log = get_logger(__name__)


class VerifyExporter:
    """
    Exports VerifierModel to ONNX and verifies the export.

    Example
    -------
    >>> exporter = VerifyExporter(cfg)
    >>> onnx_path = exporter.export(model)
    >>> report = exporter.verify(onnx_path, model)
    """

    def __init__(self, cfg: VerifyConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.export.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        model: "VerifierModel",  # type: ignore
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Export model to ONNX.

        Parameters
        ----------
        model : VerifierModel
        output_name : str, optional
            Output filename. Defaults to <experiment_name>_best.onnx

        Returns
        -------
        Path to .onnx file
        """
        output_name = output_name or f"{self.cfg.experiment.name}_best_effnet.onnx"
        dst = self.output_dir / output_name

        model.eval()
        device = next(model.parameters()).device
        size = self.cfg.input_size
        dummy = torch.zeros(1, 3, size, size, device=device)

        if self.cfg.export.half:
            model = model.half()
            dummy = dummy.half()

        dynamic_axes = None
        if self.cfg.export.dynamic_axes:
            dynamic_axes = {
                "images": {0: "batch_size"},
                "probabilities": {0: "batch_size"},
            }

        log.info(f"Exporting EfficientNetV2 → ONNX: {dst}")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(dst),
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.cfg.export.opset,
                do_constant_folding=True,
                export_params=True,
            )

        # Validate ONNX graph
        onnx_model = onnx.load(str(dst))
        onnx.checker.check_model(onnx_model)

        # Simplify
        if self.cfg.export.simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                simplified, ok = onnx_simplify(onnx_model)
                if ok:
                    onnx.save(simplified, str(dst))
                    log.info("ONNX graph simplified ✓")
                else:
                    log.warning("onnxsim returned not-ok")
            except ImportError:
                log.warning("onnxsim not installed — skipping simplification")

        # Embed metadata
        self._embed_metadata(dst, {
            "architecture": "efficientnetv2",
            "variant": self.cfg.model.variant,
            "timm_name": self.cfg.model.timm_name,
            "num_classes": self.cfg.model.num_classes,
            "input_size": self.cfg.input_size,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": "verification",
        })

        log.info(f"ONNX export complete: {dst}")

        if self.cfg.export.verify:
            self.verify(dst, model)

        return dst

    def verify(
        self,
        onnx_path: Path,
        pt_model: Optional["VerifierModel"] = None,  # type: ignore
        atol: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Verify ONNX export against PyTorch model.

        Runs a random input through both and compares outputs.
        Writes a verification report JSON.

        Returns
        -------
        dict with verification results
        """
        onnx_path = Path(onnx_path)
        log.info(f"Verifying ONNX: {onnx_path}")

        size = self.cfg.input_size
        dummy_np = np.random.rand(1, 3, size, size).astype(np.float32)

        # ORT inference
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if ort.get_available_providers().__contains__("CUDAExecutionProvider") \
            else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = sess.get_inputs()[0].name

        t0 = time.perf_counter()
        ort_logits = sess.run(None, {input_name: dummy_np})[0]
        ort_latency_ms = (time.perf_counter() - t0) * 1000

        ort_probs = self._softmax(ort_logits[0])
        ort_class = int(np.argmax(ort_probs))

        report: Dict[str, Any] = {
            "onnx_path": str(onnx_path),
            "ort_top1_class": ort_class,
            "ort_latency_ms": round(ort_latency_ms, 2),
            "ort_output_shape": list(ort_logits.shape),
            "providers": providers,
            "numerical_check": "skipped",
            "max_diff": None,
            "passed": True,
        }

        # Numerical parity against PyTorch model
        if pt_model is not None:
            pt_model.eval()
            with torch.no_grad():
                dummy_torch = torch.from_numpy(dummy_np).to(
                    next(pt_model.parameters()).device
                )
                pt_logits = pt_model(dummy_torch).cpu().numpy()

            max_diff = float(np.abs(pt_logits - ort_logits).max())
            passed = max_diff < atol
            report["numerical_check"] = "performed"
            report["max_diff"] = round(max_diff, 8)
            report["atol"] = atol
            report["passed"] = passed

            if passed:
                log.info(f"  Numerical parity ✓ (max_diff={max_diff:.2e}, atol={atol})")
            else:
                log.warning(
                    f"  Numerical mismatch! max_diff={max_diff:.2e} > atol={atol}"
                )

        # Save report
        report_path = self.output_dir / f"{onnx_path.stem}_verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Verification report → {report_path}")

        return report

    def benchmark_ort(
        self,
        onnx_path: Path,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """Measure ORT inference latency statistics."""
        size = self.cfg.input_size
        dummy = np.random.rand(1, 3, size, size).astype(np.float32)
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)
        stats = {
            "mean_ms": float(times_arr.mean()),
            "std_ms": float(times_arr.std()),
            "min_ms": float(times_arr.min()),
            "p50_ms": float(np.percentile(times_arr, 50)),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "max_ms": float(times_arr.max()),
            "fps": float(1000 / times_arr.mean()),
        }
        log.info(
            f"ORT benchmark ({n_runs} runs): "
            f"mean={stats['mean_ms']:.2f}ms fps={stats['fps']:.1f}"
        )
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_metadata(onnx_path: Path, meta: Dict[str, Any]) -> None:
        model = onnx.load(str(onnx_path))
        for k, v in meta.items():
            prop = model.metadata_props.add()
            prop.key = k
            prop.value = json.dumps(v) if not isinstance(v, str) else v
        onnx.save(model, str(onnx_path))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

"""
SPIRO ML — ONNXExporter
Exports PyTorch models to ONNX format with:
  - Dynamic batch axes
  - ONNX graph simplification (onnxsim)
  - Post-export verification (shape check + numerical parity)
  - FP16 half-precision variant
  - Metadata embedding
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch

from lib.ml.core.config import ConfigManager
from lib.ml.core.device import resolve_device
from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class ONNXExporter:
    """
    Export SPIRO models to ONNX.

    Example — YOLO
    --------------
    >>> exp = ONNXExporter(cfg)
    >>> out = exp.export_yolo("models/checkpoints/best.pt")
    >>> exp.verify(out, input_shape=(1, 3, 640, 640))

    Example — EfficientNet
    ----------------------
    >>> out = exp.export_efficientnet(model, "models/exports/classifier.onnx")
    """

    def __init__(self, cfg: ConfigManager) -> None:
        self.cfg = cfg
        self.export_dir = Path(cfg.paths.exports_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.exp_cfg = cfg.export

    # ------------------------------------------------------------------
    # YOLO export (delegates to Ultralytics)
    # ------------------------------------------------------------------

    def export_yolo(
        self,
        weights_path: Union[str, Path],
        output_name: Optional[str] = None,
        half: Optional[bool] = None,
    ) -> Path:
        """
        Export YOLOv11 weights to ONNX via Ultralytics.

        Parameters
        ----------
        weights_path : Path
            Source .pt file.
        output_name : str, optional
            Filename for output .onnx (default: same stem as weights).
        half : bool, optional
            FP16 export (GPU required). Overrides config.

        Returns
        -------
        Path to exported .onnx file.
        """
        from ultralytics import YOLO

        weights_path = Path(weights_path)
        log.info(f"Exporting YOLO → ONNX: {weights_path}")

        model = YOLO(str(weights_path))
        onnx_path = model.export(
            format="onnx",
            imgsz=self.cfg.model.input_size[0],
            opset=self.exp_cfg.opset_version,
            dynamic=self.exp_cfg.dynamic_axes,
            simplify=self.exp_cfg.simplify,
            half=half if half is not None else self.exp_cfg.half_precision,
        )

        # Move to exports dir
        onnx_path = Path(onnx_path)
        out_name = output_name or onnx_path.name
        dst = self.export_dir / out_name
        if onnx_path != dst:
            import shutil
            shutil.move(str(onnx_path), str(dst))

        self._embed_metadata(dst, {
            "architecture": "yolov11",
            "num_classes": self.cfg.dataset.num_classes,
            "class_names": list(self.cfg.dataset.class_names),
            "input_size": list(self.cfg.model.input_size),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        log.info(f"YOLO ONNX export complete: {dst}")
        return dst

    # ------------------------------------------------------------------
    # EfficientNet export (torch.onnx.export)
    # ------------------------------------------------------------------

    def export_efficientnet(
        self,
        model: torch.nn.Module,
        output_name: str = "efficientnet.onnx",
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        half: bool = False,
    ) -> Path:
        """
        Export EfficientNetV2 (SPIROClassifier) to ONNX.

        Parameters
        ----------
        model : SPIROClassifier
        output_name : str
            Output filename (inside exports dir).
        input_shape : tuple
            (N, C, H, W)
        half : bool
            Export in FP16.

        Returns
        -------
        Path to exported .onnx file.
        """
        log.info(f"Exporting EfficientNet → ONNX: {output_name}")
        dst = self.export_dir / output_name
        model.eval()
        device = next(model.parameters()).device

        dummy = torch.zeros(*input_shape, device=device)
        if half:
            model = model.half()
            dummy = dummy.half()

        dynamic_axes: Optional[Dict] = None
        if self.exp_cfg.dynamic_axes:
            dynamic_axes = {
                "images": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(dst),
                input_names=["images"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=self.exp_cfg.opset_version,
                do_constant_folding=True,
                export_params=True,
            )

        # Verify ONNX graph
        onnx_model = onnx.load(str(dst))
        onnx.checker.check_model(onnx_model)

        # Simplify
        if self.exp_cfg.simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                onnx_model, ok = onnx_simplify(onnx_model)
                if ok:
                    onnx.save(onnx_model, str(dst))
                    log.info("ONNX graph simplified")
                else:
                    log.warning("ONNX simplification returned not-ok")
            except ImportError:
                log.warning("onnxsim not installed — skipping simplification")

        self._embed_metadata(dst, {
            "architecture": "efficientnetv2",
            "num_classes": self.cfg.dataset.num_classes,
            "class_names": list(self.cfg.dataset.class_names),
            "input_shape": list(input_shape),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        log.info(f"EfficientNet ONNX export complete: {dst}")
        return dst

    # ------------------------------------------------------------------
    # Post-export verification
    # ------------------------------------------------------------------

    def verify(
        self,
        onnx_path: Union[str, Path],
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        torch_model: Optional[torch.nn.Module] = None,
        atol: float = 1e-3,
    ) -> bool:
        """
        Verify the exported ONNX model:
        1. Loads with ONNXRuntime
        2. Runs a forward pass
        3. If torch_model provided, checks numerical parity

        Returns True if all checks pass.
        """
        log.info(f"Verifying ONNX export: {onnx_path}")
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name

        dummy_np = np.random.rand(*input_shape).astype(np.float32)
        ort_out = session.run(None, {input_name: dummy_np})
        log.info(f"  ORT output shapes: {[o.shape for o in ort_out]}")

        if torch_model is not None:
            torch_model.eval()
            with torch.no_grad():
                dummy_torch = torch.from_numpy(dummy_np)
                torch_out = torch_model(dummy_torch).cpu().numpy()
            # Compare first output
            if not np.allclose(torch_out, ort_out[0], atol=atol):
                log.error("Numerical mismatch between PyTorch and ORT outputs!")
                return False
            log.info(f"  Numerical parity check passed (atol={atol})")

        log.info("ONNX verification passed ✓")
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_metadata(onnx_path: Path, meta: Dict[str, Any]) -> None:
        """Embed custom metadata into the ONNX model's metadata_props."""
        model = onnx.load(str(onnx_path))
        for k, v in meta.items():
            prop = model.metadata_props.add()
            prop.key = k
            prop.value = json.dumps(v) if not isinstance(v, str) else v
        onnx.save(model, str(onnx_path))

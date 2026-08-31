"""
SPIRO ML — GradCAM Explainability
Grad-CAM and GradCAM++ for EfficientNetV2 VerifierModel.

Generates:
  - Heatmap overlays on crop images
  - Top contributing spatial regions
  - Per-prediction explanation JSON
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from lib.ml.core.logger import get_logger

log = get_logger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for VerifierModel.

    Parameters
    ----------
    model : VerifierModel
    target_layer : nn.Module or None
        Layer to hook. If None, auto-detects the last Conv2d.
    method : str
        "gradcam" or "gradcam++"

    Example
    -------
    >>> cam = GradCAM(model, method="gradcam")
    >>> heatmap, overlay = cam.explain(image_bgr, class_idx=None)
    >>> cam.save_explanation(image_bgr, heatmap, "reports/gradcam/crop.jpg")
    """

    def __init__(
        self,
        model: "VerifierModel",  # type: ignore
        target_layer: Optional[nn.Module] = None,
        method: str = "gradcam",
    ) -> None:
        self.model = model
        self.method = method
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List = []

        layer = target_layer or self._find_last_conv(model)
        if layer is None:
            raise RuntimeError("Could not find a Conv2d layer for Grad-CAM")
        self._target_layer = layer
        self._register_hooks(layer)
        log.info(f"GradCAM ({method}) hooked onto: {type(layer).__name__}")

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self, layer: nn.Module) -> None:
        def fwd_hook(_, __, output):
            self._activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._handles.append(layer.register_forward_hook(fwd_hook))
        self._handles.append(layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Core CAM generation
    # ------------------------------------------------------------------

    def explain(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate Grad-CAM heatmap and overlay.

        Parameters
        ----------
        source : image path, BGR ndarray, or PIL Image
        class_idx : int, optional
            Target class. None = argmax (predicted class).
        alpha : float
            Heatmap overlay transparency.

        Returns
        -------
        (heatmap_uint8, overlay_bgr, explanation_dict)
        """
        img_bgr = self._load(source)
        tensor = self._preprocess(img_bgr)
        tensor.requires_grad_(True)

        self.model.eval()
        logits = self.model(tensor)
        probs = F.softmax(logits.detach(), dim=1).squeeze()

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        # Backward pass
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Compute CAM
        cam = self._compute_cam()

        # Resize to image size
        h, w = img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cam_resized = np.clip(cam_resized, 0, None)
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
        heatmap = np.uint8(255 * cam_resized)

        # Colour heatmap
        heatmap_colour = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colour, alpha, 0)

        # Top contributing regions
        top_regions = self._top_regions(cam_resized, n=3)

        explanation = {
            "class_id": int(class_idx),
            "class_name": self.model.class_names[class_idx],
            "confidence": float(probs[class_idx]),
            "top5_predictions": [
                {
                    "rank": i + 1,
                    "class_id": int(idx),
                    "class_name": self.model.class_names[int(idx)],
                    "confidence": float(probs[int(idx)]),
                }
                for i, idx in enumerate(probs.argsort(descending=True)[:5])
            ],
            "top_contributing_regions": top_regions,
            "cam_method": self.method,
        }

        return heatmap, overlay, explanation

    def _compute_cam(self) -> np.ndarray:
        """Compute the CAM from stored activations and gradients."""
        acts = self._activations.squeeze(0).cpu().numpy()   # [C, H, W]
        grads = self._gradients.squeeze(0).cpu().numpy()    # [C, H, W]

        if self.method == "gradcam":
            weights = grads.mean(axis=(1, 2))               # [C]
            cam = np.einsum("c,chw->hw", weights, acts)

        elif self.method == "gradcam++":
            # GradCAM++ uses second-order gradient estimate
            relu_grads = np.maximum(grads, 0)
            sum_acts = acts.sum(axis=(1, 2), keepdims=True) + 1e-9
            alpha = relu_grads ** 2 / (2 * relu_grads ** 2 + sum_acts * relu_grads ** 3 + 1e-9)
            weights = (alpha * np.maximum(grads, 0)).sum(axis=(1, 2))
            cam = np.einsum("c,chw->hw", weights, acts)
        else:
            raise ValueError(f"Unknown CAM method: {self.method}")

        return cam

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def save_explanation(
        self,
        source: Union[str, Path, np.ndarray],
        output_dir: Union[str, Path],
        class_idx: Optional[int] = None,
        stem: str = "explanation",
    ) -> Dict[str, str]:
        """Generate and save heatmap + overlay + JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        heatmap, overlay, explanation = self.explain(source, class_idx=class_idx)

        img_bgr = self._load(source)
        paths = {}
        # Original
        orig_path = output_dir / f"{stem}_original.jpg"
        cv2.imwrite(str(orig_path), img_bgr)
        paths["original"] = str(orig_path)
        # Heatmap
        hmap_path = output_dir / f"{stem}_heatmap.jpg"
        cv2.imwrite(str(hmap_path), cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
        paths["heatmap"] = str(hmap_path)
        # Overlay
        ov_path = output_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(ov_path), overlay)
        paths["overlay"] = str(ov_path)
        # JSON
        json_path = output_dir / f"{stem}_explanation.json"
        with open(json_path, "w") as f:
            json.dump({**explanation, "files": paths}, f, indent=2)
        paths["json"] = str(json_path)
        log.info(f"Explanation saved: {output_dir / stem}.*")
        return paths

    @staticmethod
    def _top_regions(
        cam: np.ndarray, n: int = 3, patch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Extract top N contributing spatial regions from the CAM."""
        h, w = cam.shape
        regions = []
        cam_copy = cam.copy()
        for _ in range(n):
            if cam_copy.max() == 0:
                break
            y, x = np.unravel_index(cam_copy.argmax(), cam_copy.shape)
            score = float(cam_copy[y, x])
            regions.append({
                "center_x": int(x),
                "center_y": int(y),
                "score": round(score, 4),
                "bbox": [
                    max(0, int(x - patch_size // 2)),
                    max(0, int(y - patch_size // 2)),
                    min(w, int(x + patch_size // 2)),
                    min(h, int(y + patch_size // 2)),
                ],
            })
            # Suppress region
            y0 = max(0, y - patch_size // 2)
            y1 = min(h, y + patch_size // 2)
            x0 = max(0, x - patch_size // 2)
            x1 = min(w, x + patch_size // 2)
            cam_copy[y0:y1, x0:x1] = 0
        return regions

    @staticmethod
    def _find_last_conv(model: nn.Module) -> Optional[nn.Module]:
        """Auto-detect the last Conv2d in the model."""
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    @staticmethod
    def _load(source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot read: {source}")
            return img
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, Image.Image):
            return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        raise TypeError(f"Unsupported type: {type(source)}")

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        size = self.model.input_size
        resized = cv2.resize(img_bgr, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb  = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.model.device)
